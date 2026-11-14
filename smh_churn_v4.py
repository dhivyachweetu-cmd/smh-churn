#!/usr/bin/env python3
"""
SMH-Churn: Segment-Migration-aware discrete-time Hazard churn pipeline
======================================================================
Implements every experiment needed by the manuscript:
  * data loaders (UCI Online Retail II, Dunnhumby Complete Journey, generic CSV, synthetic)
  * leakage-safe rolling-origin folds (train / gap / valid / gap / test)
  * train-only segmentation (KMeans or GMM) + segment-migration features
  * discrete-time hazard GBDT (LightGBM) + isotonic calibration
  * baselines B1-B6, B8 and ablations A1-A4 (B7 GRU is not included)
  * metrics: ROC-AUC, PR-AUC, Brier, ECE, top-decile lift, value-at-risk capture, net benefit
  * statistics: Friedman, Wilcoxon (Holm), paired customer-level bootstrap
  * explanations: TreeSHAP by segment + Spearman stability across test origins
  * export of paper tables (6-10) and figures to --outdir

Usage examples
  python smh_churn.py --dataset synthetic --outdir out_synth
  python smh_churn.py --dataset online_retail_ii --path online_retail_II.xlsx --H 90 --delta 30 --outdir out_d1
  python smh_churn.py --dataset dunnhumby --path dunnhumby_dir/   # folder, or transaction_data.csv/.xlsx --H 60 --delta 28 --outdir out_d2
"""
import argparse, json, math, os, warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    H: int = 90                 # churn horizon (days)
    delta: int = 30             # origin spacing (days)
    K: int = 4                  # number of segments
    L: int = 3                  # volatility look-back (origins)
    min_history_days: int = 120 # first origin = first order + this
    n_valid: int = 3            # validation origins per fold
    n_test: int = 3             # test origins per fold
    min_train: int = 6          # minimum train origins in first fold
    fold_step: int = 3          # shift of the training block between folds
    seed: int = 42
    margin: float = 0.2         # assumed gross margin on order value
    gamma: float = 0.3          # assumed save rate
    cost: float = 1.0           # cost of one retention contact (same currency as amount)
    budgets: tuple = (0.05, 0.10, 0.20)
    prior_alpha: float = 50.0   # shrinkage for seg_prior
    max_cox_rows: int = 150000
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    tune_trials: int = 0        # >0 enables Optuna tuning on the validation block
    gmm_variant: bool = True    # run ablation A4 (GMM segmenter)
    calibration: str = "platt"   # platt | isotonic
    full_history_lag: bool = True  # v4: lag/migration features use the full past of each customer (not only the current block)

RFM = ["recency", "frequency", "monetary", "tenure", "avg_gap", "n_items", "aov", "last_gap", "freq_recent"]
SEGF = ["seg", "prev_seg", "seg_prior"]
MIGF = ["down", "dwell", "vol", "m_exp", "q_lapse"]
TRAJ = ["gap_ratio", "d_freq", "d_aov", "d_rec"]      # v4: customer trajectory features (no segments needed)
MIG2 = ["n_down", "n_up", "rank_slope"]               # v4: individual migration history
B9, SMHP = "B9 Hazard + trajectory (no segments)", "SMH-Churn+ (trajectory + migration)"

# ----------------------------------------------------------------------------- loaders
def _finish(df):
    df = df.dropna(subset=["customer_id", "order_date"]).copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_id"] = df["order_id"].astype(str)
    return df[["customer_id", "order_id", "order_date", "amount", "product_id"]]

def load_online_retail_ii(path):
    """UCI Online Retail II (xlsx with two sheets, or csv). Columns: Invoice, StockCode, Quantity,
    InvoiceDate, Price, Customer ID, Country."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None)
        raw = pd.concat(sheets.values(), ignore_index=True)
    else:
        raw = pd.read_csv(path, encoding="latin1")
    raw = raw.rename(columns={"Customer ID": "customer_id", "CustomerID": "customer_id", "Invoice": "order_id",
                              "InvoiceNo": "order_id", "InvoiceDate": "order_date", "StockCode": "product_id"})
    raw = raw[~raw["order_id"].astype(str).str.startswith("C")]           # cancellations
    raw = raw[(raw["Quantity"] > 0) & (raw["Price"] > 0)]
    raw["amount"] = raw["Quantity"] * raw["Price"]
    return _finish(raw)

def load_dunnhumby(path):
    """Dunnhumby 'The Complete Journey'. `path` may be a folder (searched recursively for transaction_data.csv/.xlsx)
    or the file itself (.csv or .xlsx, all sheets are concatenated). Column names are matched case-insensitively:
    household_key, BASKET_ID, DAY, PRODUCT_ID, SALES_VALUE. DAY (integer) is mapped to dates from 2000-01-01."""
    p = str(path)
    if os.path.isdir(p):
        hits = [os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs
                if f.lower().startswith("transaction_data") and f.lower().endswith((".csv", ".xlsx", ".xls"))]
        if not hits:
            raise FileNotFoundError(f"No transaction_data.csv/.xlsx found under {p}. Found: "
                                    f"{[f for _, _, fs in os.walk(p) for f in fs][:20]}")
        p = sorted(hits)[0]
    if p.lower().endswith((".xlsx", ".xls")):
        raw = pd.concat(pd.read_excel(p, sheet_name=None).values(), ignore_index=True)
    else:
        raw = pd.read_csv(p)
    low = {c.lower().strip(): c for c in raw.columns}
    need = {"household_key": "customer_id", "basket_id": "order_id", "product_id": "product_id",
            "sales_value": "amount", "day": "DAY"}
    miss = [k for k in need if k not in low]
    if miss:
        raise KeyError(f"Missing columns {miss}; file has {list(raw.columns)}")
    raw = raw.rename(columns={low[k]: v for k, v in need.items()})
    raw["order_date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(raw["DAY"], unit="D")
    raw = raw[raw["amount"] > 0]
    if raw["DAY"].max() < 700:
        print(f"WARNING: DAY only reaches {raw['DAY'].max()} (full data reaches 711). "
              f"The file may be truncated (Excel caps at 1,048,576 rows). Use the original CSV.")
    return _finish(raw)

def load_hm(path, price_scale=1000.0):
    """H&M Personalized Fashion Recommendations: transactions_train.csv (or a customer sample of it) with columns
    t_dat, customer_id, article_id (optional), price. All items bought by a customer on one day form one order.
    Prices are normalised in the original file, so they are multiplied by price_scale (a pure change of units;
    the contact cost in Cfg is expressed in the same scaled units)."""
    raw = pd.read_csv(path, usecols=lambda c: c in ("t_dat", "customer_id", "article_id", "price"))
    raw = raw.rename(columns={"t_dat": "order_date", "article_id": "product_id"})
    if "product_id" not in raw.columns: raw["product_id"] = np.arange(len(raw))
    raw = raw[raw["price"] > 0].copy()
    raw["amount"] = raw["price"] * price_scale
    raw["order_date"] = pd.to_datetime(raw["order_date"])
    raw["order_id"] = raw["customer_id"].astype(str) + "_" + raw["order_date"].dt.strftime("%Y%m%d")
    return _finish(raw)

def load_generic(path, mapping):
    """CSV with arbitrary names: mapping = {'customer_id': 'colA', 'order_id': ..., 'order_date': ..., 'amount': ..., 'product_id': ...}"""
    raw = pd.read_csv(path).rename(columns={v: k for k, v in mapping.items()})
    return _finish(raw)

def make_synthetic(n=2500, days=1100, seed=0):
    """Latent-state generator: purchase rate falls through states before lapsing, so migration carries signal."""
    rng = np.random.default_rng(seed)
    rates = np.array([0.14, 0.07, 0.03, 0.01])
    T = np.array([[.80, .17, .03, 0.], [.10, .75, .13, .02], [.04, .14, .70, .12], [.01, .04, .15, .80]])
    p_quit = np.array([.003, .01, .04, .12])
    rows, oid = [], 0
    start0 = pd.Timestamp("2021-01-01")
    for c in range(n):
        s = rng.choice(4, p=[.35, .35, .2, .1]); t0 = rng.integers(0, days // 2); alive = True
        for e in range(t0, days, 30):
            if not alive: break
            k = rng.poisson(rates[s] * 30)
            for _ in range(k):
                rows.append((f"C{c}", f"O{oid}", start0 + pd.Timedelta(days=float(e + rng.uniform(0, 30))),
                             float(rng.gamma(2, 20 * (1 + (3 - s) * .3))), int(rng.integers(0, 300)))); oid += 1
            if rng.random() < p_quit[s]: alive = False
            s = rng.choice(4, p=T[s])
    df = pd.DataFrame(rows, columns=["customer_id", "order_id", "order_date", "amount", "product_id"])
    return df[df.order_date < start0 + pd.Timedelta(days=days)]

def to_order_level(df):
    """Collapse line items to one row per order (customer, order_id)."""
    g = df.groupby(["customer_id", "order_id"], sort=False)
    o = g.agg(order_date=("order_date", "min"), amount=("amount", "sum"), n_items=("product_id", "size")).reset_index()
    o = o.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    o["gap"] = o.groupby("customer_id")["order_date"].diff().dt.days
    return o

# ----------------------------------------------------------------------------- snapshots
def make_origins(orders, cfg):
    t0 = orders.order_date.min() + pd.Timedelta(days=cfg.min_history_days)
    t_end = orders.order_date.max() - pd.Timedelta(days=cfg.H)
    return list(pd.date_range(t0, t_end, freq=f"{cfg.delta}D"))

def snapshot(orders, tau, cfg):
    """Features of every customer using orders with order_date <= tau ONLY, plus label y over (tau, tau+H]."""
    hist = orders[orders.order_date <= tau]
    g = hist.groupby("customer_id")
    f = pd.DataFrame({
        "recency": (tau - g.order_date.max()).dt.days,
        "frequency": g.size(),
        "monetary": g.amount.sum(),
        "tenure": (tau - g.order_date.min()).dt.days,
        "n_items": g.n_items.sum(),
    })
    f["avg_gap"] = f["tenure"] / f["frequency"].clip(lower=1)
    f["aov"] = f["monetary"] / f["frequency"]
    last = hist.groupby("customer_id").tail(1).set_index("customer_id")["gap"]
    f["last_gap"] = last.reindex(f.index).fillna(f["avg_gap"])
    recent = hist[hist.order_date > tau - pd.Timedelta(days=cfg.H)].groupby("customer_id").size()
    f["freq_recent"] = recent.reindex(f.index).fillna(0)
    f = f[f["recency"] <= cfg.H].copy()                                   # at-risk set
    fut = orders[(orders.order_date > tau) & (orders.order_date <= tau + pd.Timedelta(days=cfg.H))]
    bought = set(fut.customer_id)
    f["y"] = [0 if c in bought else 1 for c in f.index]
    return f

def build_base_panel(orders, origins, cfg):
    rows = []
    for k, tau in enumerate(origins):
        f = snapshot(orders, tau, cfg)
        f["period"], f["tau"] = k, tau
        rows.append(f.reset_index().rename(columns={"index": "customer_id"}))
    panel = pd.concat(rows, ignore_index=True).sort_values(["customer_id", "period"]).reset_index(drop=True)
    # first churn event is terminal: drop rows after the first y == 1
    before = panel.groupby("customer_id")["y"].cumsum() - panel["y"]
    panel = panel[before == 0].reset_index(drop=True)
    # expected value over next H days (assumed margin): AOV * margin * expected orders
    exp_orders = (cfg.H / panel["avg_gap"].clip(lower=1)).clip(upper=20)
    panel["V"] = panel["aov"] * cfg.margin * exp_orders
    return panel

# ----------------------------------------------------------------------------- folds
def make_folds(n_origins, cfg):
    gap = math.ceil(cfg.H / cfg.delta)
    folds, a = [], cfg.min_train
    while a + gap + cfg.n_valid + gap + cfg.n_test <= n_origins:
        tr = list(range(0, a)); va = list(range(a + gap, a + gap + cfg.n_valid))
        te = list(range(va[-1] + 1 + gap, va[-1] + 1 + gap + cfg.n_test))
        folds.append(dict(train=tr, valid=va, test=te)); a += cfg.fold_step
    return folds

# ----------------------------------------------------------------------------- segmentation + migration
class Segmenter:
    cols = ["recency", "frequency", "monetary", "avg_gap", "freq_recent"]
    def __init__(self, K, method="kmeans", seed=42): self.K, self.method, self.seed = K, method, seed
    def _x(self, f): return np.log1p(f[self.cols].clip(lower=0).values)
    def fit(self, feats):
        X = self._x(feats); self.sc = StandardScaler().fit(X); Z = self.sc.transform(X)
        if self.method == "gmm":
            self.m = GaussianMixture(self.K, covariance_type="full", random_state=self.seed, n_init=3).fit(Z)
        else:
            self.m = KMeans(self.K, n_init=10, random_state=self.seed).fit(Z)
        raw = self.m.predict(Z)
        order = pd.Series(feats["monetary"].values).groupby(raw).mean().sort_values()
        self.rank = {c: r for r, c in enumerate(order.index)}              # 0 = lowest value
        self.sil = silhouette_score(Z, raw, sample_size=min(5000, len(Z)), random_state=self.seed) if len(set(raw)) > 1 else np.nan
        return self
    def predict(self, feats):
        raw = self.m.predict(self.sc.transform(self._x(feats)))
        return np.array([self.rank[c] for c in raw])

def add_migration(panel, K, L, P=None, prior=None, alpha=50.0, base_rate=None, fit_periods=None):
    """Adds seg-derived features. P and prior are estimated from the TRAIN rows only (fit_periods) and reused for valid/test.
    All lag features look only at earlier origins of the same customer, so nothing from the future enters a row."""
    p = panel.sort_values(["customer_id", "period"]).copy()
    g = p.groupby("customer_id")["seg"]
    p["has_prev"] = g.shift(1).notna().astype(int)
    p["prev_seg"] = g.shift(1).fillna(p["seg"]).astype(int)
    p["changed"] = (p["seg"] != p["prev_seg"]).astype(int)
    p["down"] = (p["seg"] < p["prev_seg"]).astype(int)
    p["up"] = (p["seg"] > p["prev_seg"]).astype(int)
    p["vol"] = p.groupby("customer_id")["changed"].transform(lambda s: s.rolling(L, min_periods=1).sum())
    p["n_down"] = p.groupby("customer_id")["down"].transform(lambda s: s.rolling(L, min_periods=1).sum())
    p["n_up"] = p.groupby("customer_id")["up"].transform(lambda s: s.rolling(L, min_periods=1).sum())
    p["rank_slope"] = (p["seg"] - g.shift(L)).fillna(0)
    run = p.groupby("customer_id")["changed"].cumsum()
    p["dwell"] = p.groupby(["customer_id", run]).cumcount() + 1
    g2 = p.groupby("customer_id")
    p["gap_ratio"] = p["recency"] / p["avg_gap"].clip(lower=1)
    p["d_freq"] = (p["freq_recent"] - g2["freq_recent"].shift(1)).fillna(0)
    p["d_aov"] = (np.log1p(p["aov"]) - np.log1p(g2["aov"].shift(1))).fillna(0)
    p["d_rec"] = (p["recency"] - g2["recency"].shift(1)).fillna(0)
    fit = p if fit_periods is None else p[p["period"].isin(fit_periods)]
    if P is None:
        ft = fit[fit["has_prev"] == 1]
        P = pd.crosstab(ft["prev_seg"], ft["seg"]).reindex(index=range(K), columns=range(K), fill_value=0).astype(float)
        P = P.div(P.sum(axis=1).replace(0, 1), axis=0)
        for s_ in range(K):
            if P.loc[s_].sum() == 0: P.loc[s_, s_] = 1.0
    if prior is None:
        br = fit["y"].mean()
        agg = fit.groupby("seg")["y"].agg(["sum", "count"]).reindex(range(K), fill_value=0)
        prior = ((agg["sum"] + alpha * br) / (agg["count"] + alpha)).values
    ranks = np.arange(K)
    exp_change = P.values @ ranks - ranks
    lapse = P.values[:, : max(1, K // 2)].sum(axis=1)
    p["m_exp"] = exp_change[p["seg"].values]; p["q_lapse"] = lapse[p["seg"].values]
    p["seg_prior"] = prior[p["seg"].values]
    return p.drop(columns=["changed", "up", "has_prev"]), P, prior

# ----------------------------------------------------------------------------- models
def lgb_params(cfg):
    return dict(n_estimators=cfg.n_estimators, learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
                random_state=cfg.seed, verbose=-1)

def fit_lgb(tr, va, feats, cfg, params=None):
    m = lgb.LGBMClassifier(**(params or lgb_params(cfg)))
    m.fit(tr[feats], tr["y"], eval_set=[(va[feats], va["y"])], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m

def tune_lgb(tr, va, feats, cfg):
    try:
        import optuna
    except ImportError:
        return lgb_params(cfg)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def obj(t):
        pr = lgb_params(cfg)
        pr.update(num_leaves=t.suggest_int("num_leaves", 8, 96), learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                  min_child_samples=t.suggest_int("min_child_samples", 10, 100), colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                  reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True))
        m = fit_lgb(tr, va, feats, cfg, pr)
        return average_precision_score(va["y"], m.predict_proba(va[feats])[:, 1])
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    st.optimize(obj, n_trials=cfg.tune_trials)
    pr = lgb_params(cfg); pr.update(st.best_params); return pr

class _Platt:
    """Monotone Platt scaling on logit (probabilities) or log (positive scores, e.g. Cox partial hazard)."""
    def fit(self, s, y):
        s = np.asarray(s, float); self.prob = bool(np.nanmin(s) >= 0 and np.nanmax(s) <= 1)
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(s), y); return self
    def _x(self, s):
        s = np.asarray(s, float)
        z = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))) if self.prob else np.log(np.clip(s, 1e-9, None))
        return z.reshape(-1, 1)
    def predict(self, s): return self.lr.predict_proba(self._x(s))[:, 1]

CAL = {"kind": "platt"}          # set to "isotonic" via --calibration isotonic (isotonic creates tied scores)
def calibrate(score_va, y_va):
    if CAL["kind"] == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(score_va, y_va)
    return _Platt().fit(score_va, y_va)

def run_fold(panel, fold, cfg, orders_snap_cache=None):
    """Fit all models on one fold. Returns dict(name -> dict(te=index-aligned probs, va=...)), segmenter info."""
    tr0, va0, te0 = (panel[panel.period.isin(fold[k])].copy() for k in ("train", "valid", "test"))
    out, info = {}, {}
    last_tr = panel[panel.period == fold["train"][-1]]
    feats_seg = {}

    def with_seg(method):
        sg = Segmenter(cfg.K, method, cfg.seed).fit(last_tr)
        if cfg.full_history_lag:
            # segmenter, P and prior are fitted on TRAIN rows only; lag features look back over every earlier origin
            hist = panel[panel.period <= fold["test"][-1]].copy(); hist["seg"] = sg.predict(hist)
            hist, P, prior = add_migration(hist, cfg.K, cfg.L, alpha=cfg.prior_alpha, fit_periods=set(fold["train"]))
            sel = lambda per: hist[hist.period.isin(per)]
            return sel(fold["train"]), sel(fold["valid"]), sel(fold["test"]), P, prior, sg
        res = []
        for d in (tr0, va0, te0):
            d = d.copy(); d["seg"] = sg.predict(d); res.append(d)
        tr, P, prior = add_migration(res[0], cfg.K, cfg.L, alpha=cfg.prior_alpha)
        va, _, _ = add_migration(res[1], cfg.K, cfg.L, P, prior); te, _, _ = add_migration(res[2], cfg.K, cfg.L, P, prior)
        return tr, va, te, P, prior, sg

    tr, va, te, P, prior, sg = with_seg("kmeans")
    info.update(P=P, prior=prior, seg=sg, te=te, tr=tr, sil=sg.sil)
    te_idx = te.index

    def add(name, model_va, model_te, cal=True):
        if cal:
            iso = calibrate(model_va, va["y"].values); model_te = iso.predict(model_te)
        out[name] = pd.Series(np.asarray(model_te), index=te_idx)

    _pc = {}
    def prm(key, trd, vad, fs):
        """Per-model Optuna tuning on the validation block (only when cfg.tune_trials > 0); same budget for every model."""
        if cfg.tune_trials <= 0: return None
        if key not in _pc: _pc[key] = tune_lgb(trd, vad, fs, cfg)
        return _pc[key]

    # ---- static baselines: one row per customer at the last train origin ----
    st_tr = tr[tr.period == fold["train"][-1]]
    sc = StandardScaler().fit(np.log1p(st_tr[RFM].clip(lower=0)))
    X = lambda d, cols=RFM: np.log1p(d[cols].clip(lower=0).values)
    lr = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(X(st_tr)), st_tr["y"])
    add("B1 Logistic regression", lr.predict_proba(sc.transform(X(va)))[:, 1], lr.predict_proba(sc.transform(X(te)))[:, 1])
    rf = RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=cfg.seed).fit(st_tr[RFM], st_tr["y"])
    add("B2 Random forest", rf.predict_proba(va[RFM])[:, 1], rf.predict_proba(te[RFM])[:, 1])
    if len(st_tr) > 50:
        f4 = RFM + ["seg"]
        if cfg.tune_trials > 0:      # tuned static baselines get the same trial budget as the hazard models
            m3 = fit_lgb(st_tr, va, RFM, cfg, prm("B3", st_tr, va, RFM)); m4 = fit_lgb(st_tr, va, f4, cfg, prm("B4", st_tr, va, f4))
        else:
            m3 = lgb.LGBMClassifier(**{**lgb_params(cfg), "n_estimators": 200}).fit(st_tr[RFM], st_tr["y"])
            m4 = lgb.LGBMClassifier(**{**lgb_params(cfg), "n_estimators": 200}).fit(st_tr[f4], st_tr["y"])
        add("B3 LightGBM (static)", m3.predict_proba(va[RFM])[:, 1], m3.predict_proba(te[RFM])[:, 1])
        mp = fit_lgb(tr, va, RFM, cfg, prm("B3p", tr, va, RFM)); add("B3p LightGBM (pooled snapshots)", mp.predict_proba(va[RFM])[:, 1], mp.predict_proba(te[RFM])[:, 1])
        add("B4 Cluster label + LightGBM", m4.predict_proba(va[f4])[:, 1], m4.predict_proba(te[f4])[:, 1])

    # ---- B5 BG/NBD (optional) ----
    try:
        import io, contextlib
        from lifetimes import BetaGeoFitter
        s = st_tr[st_tr["tenure"] > 0].copy()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bg = BetaGeoFitter(penalizer_coef=0.1).fit(s["frequency"] - 1, s["tenure"] - s["recency"], s["tenure"])
            pa = lambda d: 1 - np.asarray(bg.conditional_probability_alive(d["frequency"] - 1, d["tenure"] - d["recency"], d["tenure"]))
            sv, stt = pa(va), pa(te)
        if not (np.isfinite(sv).all() and np.isfinite(stt).all()): raise ValueError("non-finite p_alive")
        if np.std(stt) < 1e-9: raise ValueError("BG/NBD score is constant (fit degenerate) - check data / penalizer")
        add("B5 BG/NBD", sv, stt)
    except Exception as e:                       # lifetimes missing or failed
        info["B5_skipped"] = str(e)[:80]

    # ---- B6 Cox time-varying (optional) ----
    try:
        from lifelines import CoxTimeVaryingFitter
        d = tr.sample(min(len(tr), cfg.max_cox_rows), random_state=cfg.seed) if len(tr) > cfg.max_cox_rows else tr
        cx = pd.DataFrame(np.log1p(d[RFM].clip(lower=0)).values, columns=RFM)
        cx = (cx - cx.mean()) / cx.std().replace(0, 1)
        cx["id"] = d["customer_id"].values; cx["start"] = d["period"].values; cx["stop"] = d["period"].values + 1; cx["event"] = d["y"].values
        ctv = CoxTimeVaryingFitter(penalizer=0.1).fit(cx, id_col="id", event_col="event", start_col="start", stop_col="stop")
        mu, sd = np.log1p(d[RFM].clip(lower=0)).mean(), np.log1p(d[RFM].clip(lower=0)).std().replace(0, 1)
        sc6 = lambda dd: ((np.log1p(dd[RFM].clip(lower=0)) - mu) / sd)
        add("B6 Cox time-varying", ctv.predict_partial_hazard(sc6(va)).values, ctv.predict_partial_hazard(sc6(te)).values)
    except Exception as e:
        info["B6_skipped"] = str(e)[:80]

    # ---- hazard models ----
    f_b8 = RFM + ["period"]
    f_b9 = RFM + ["period"] + TRAJ                                   # trajectory features, NO segments (control)
    f_a1 = RFM + ["period", "seg", "prev_seg"]
    f_a2 = RFM + ["period"] + ["seg", "prev_seg"] + MIGF
    f_smh = RFM + ["period"] + SEGF + MIGF
    f_smhp = f_smh + TRAJ + MIG2                                     # SMH-Churn+: segments + migration history + trajectory
    params = prm("SMH", tr, va, f_smh)
    models = {}
    for name, fs, key in (("B8 Hazard GBDT (no segments)", f_b8, "B8"), (B9, f_b9, "B9"), ("A1 No migration features", f_a1, "SMH"),
                          ("A2 No seg_prior", f_a2, "SMH"), ("SMH-Churn", f_smh, "SMH"), (SMHP, f_smhp, "SMHP")):
        pr = params if key == "SMH" else prm(key, tr, va, fs)
        m = fit_lgb(tr, va, fs, cfg, pr); models[name] = (m, fs)
        add(name, m.predict_proba(va[fs])[:, 1], m.predict_proba(te[fs])[:, 1])
    m, fs = models["SMH-Churn"]
    add("A3 SMH without calibration", None, m.predict_proba(te[fs])[:, 1], cal=False)
    try:    # A5: recalibrate on the most recent validation origin only (reduces prior shift between validation and test)
        last = (va.period == va.period.max()).values
        cal5 = calibrate(m.predict_proba(va[fs])[:, 1][last], va["y"].values[last])
        out["A5 Recent-window calibration"] = pd.Series(cal5.predict(m.predict_proba(te[fs])[:, 1]), index=te_idx)
    except Exception as e:
        info["A5_skipped"] = str(e)[:80]
    info["model"], info["feats"] = m, fs

    if cfg.gmm_variant:
        try:
            trg, vag, teg, _, _, _ = with_seg("gmm")
            mg = fit_lgb(trg, vag, f_smh, cfg, params)
            iso = calibrate(mg.predict_proba(vag[f_smh])[:, 1], vag["y"].values)
            out["A4 GMM segmenter"] = pd.Series(iso.predict(mg.predict_proba(teg[f_smh])[:, 1]), index=teg.index).reindex(te_idx)
        except Exception as e:
            info["A4_skipped"] = str(e)[:80]
    return out, info

# ----------------------------------------------------------------------------- metrics
def ece(y, p, bins=10):
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    d = pd.DataFrame({"y": y, "p": p, "q": q.values})
    g = d.groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] / g["n"].sum() * (g["y"] - g["p"]).abs()).sum())

def decile_lift(y, p):
    n = max(1, int(0.1 * len(y))); idx = np.argsort(-p)[:n]
    base = y.mean(); return float(y[idx].mean() / base) if base > 0 else np.nan

def targeting(te, p, cfg):
    """Per test origin: rank by NB = p*gamma*V - cost, contact the top fraction f. Returns dict of averaged metrics."""
    res = {f"VaR@{int(b*100)}": [] for b in cfg.budgets}
    res.update({f"RealCap@{int(b*100)}": [] for b in cfg.budgets}); res["NB@10"] = []
    for k, d in te.assign(p=p).groupby("period"):
        if len(d) < 20 or d["y"].sum() == 0: continue
        nb = d["p"].values * cfg.gamma * d["V"].values - cfg.cost
        order = np.argsort(-nb); pv = d["p"].values * d["V"].values; yv = d["y"].values * d["V"].values
        for b in cfg.budgets:
            n = max(1, int(b * len(d))); sel = [i for i in order[:n] if nb[i] > 0]
            res[f"VaR@{int(b*100)}"].append(pv[sel].sum() / max(pv.sum(), 1e-9))
            res[f"RealCap@{int(b*100)}"].append(yv[sel].sum() / max(yv.sum(), 1e-9))
            if b == 0.10: res["NB@10"].append(float((nb[sel]).sum()))
    return {k: float(np.mean(v)) if v else np.nan for k, v in res.items()}

def evaluate(te, p, cfg):
    y = te["y"].values; p = np.asarray(p, float)
    ok = ~np.isnan(p); y, p, tt = y[ok], p[ok], te[ok]
    if len(np.unique(y)) < 2: return None
    r = dict(AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p), Brier=brier_score_loss(y, np.clip(p, 0, 1)),
             ECE=ece(y, p), Lift10=decile_lift(y, p), base_rate=y.mean(), n=len(y))
    r.update(targeting(tt, p, cfg)); return r

# ----------------------------------------------------------------------------- explanations
def explain(info, cfg, outdir, name):
    import shap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m, fs, te = info["model"], info["feats"], info["te"]
    sv = shap.TreeExplainer(m).shap_values(te[fs]); sv = sv[1] if isinstance(sv, list) else sv
    A = pd.DataFrame(np.abs(sv), columns=fs, index=te.index)
    by_seg = A.groupby(te["seg"]).mean()
    glob = A.mean().sort_values(ascending=False)
    # stability: Spearman of segment-wise importance vectors between consecutive test origins
    rows = []
    periods = sorted(te["period"].unique())
    for s in sorted(te["seg"].unique()):
        vecs = [A[(te["period"] == p) & (te["seg"] == s)].mean() for p in periods]
        for a, b in zip(vecs[:-1], vecs[1:]):
            if a.notna().all() and b.notna().all() and a.std() > 0 and b.std() > 0:
                rows.append(dict(seg=s, rho=stats.spearmanr(a.values, b.values)[0]))
    stab = pd.DataFrame(rows)
    top3 = {int(s): ", ".join(by_seg.loc[s].sort_values(ascending=False).index[:3]) for s in by_seg.index}
    by_seg.to_csv(os.path.join(outdir, f"shap_by_segment_{name}.csv")); glob.to_csv(os.path.join(outdir, f"shap_global_{name}.csv"))
    fig, ax = plt.subplots(figsize=(7, 4)); glob.head(12)[::-1].plot.barh(ax=ax, color="#1F3A5F"); ax.set_xlabel("mean |SHAP|"); ax.set_title("Global importance")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"fig_shap_global_{name}.png"), dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4)); z = by_seg[glob.index[:12]]; im = ax.imshow(z.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(z.shape[1])); ax.set_xticklabels(z.columns, rotation=60, ha="right"); ax.set_yticks(range(len(z))); ax.set_yticklabels([f"S{int(i)+1}" for i in z.index])
    fig.colorbar(im, label="mean |SHAP|"); ax.set_title("Segment-wise importance"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"fig_shap_segment_heatmap_{name}.png"), dpi=300); plt.close(fig)
    return dict(top3=top3, stability_mean=float(stab.rho.mean()) if len(stab) else np.nan, stability_min=float(stab.rho.min()) if len(stab) else np.nan)

def plots(preds, panel_info, outdir, name):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # reliability diagram for best baseline and SMH
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--", lw=1)
    for mname, c in (("SMH-Churn", "#2A7F8E"), ("B3 LightGBM (static)", "#B7791F"), ("B8 Hazard GBDT (no segments)", "#6B3FA0")):
        d = preds[preds.model == mname].dropna(subset=["p"])
        if d.empty: continue
        q = pd.qcut(d["p"].rank(method="first"), 10, labels=False); g = d.groupby(q).agg(p=("p", "mean"), y=("y", "mean"))
        ax.plot(g["p"], g["y"], "o-", color=c, label=mname)
    ax.set_xlabel("predicted churn probability"); ax.set_ylabel("observed churn rate"); ax.legend(fontsize=8); ax.set_title("Reliability")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"fig_reliability_{name}.png"), dpi=300); plt.close(fig)

# ----------------------------------------------------------------------------- statistics
def holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0
    for r, i in enumerate(idx): run = max(run, (m - r) * pvals[i]); adj[i] = min(1, run)
    return adj

def stat_tests(res, preds, cfg, ref="SMH-Churn", metric="PR_AUC", n_boot=500):
    keep = [m for m in res.model.unique() if m.startswith("B") or m in (ref, SMHP)]
    pv = res[res.model.isin(keep)].pivot_table(index="fold", columns="model", values=metric)
    pv = pv.dropna(axis=1, thresh=max(2, len(pv))).dropna()
    out = {}
    if pv.shape[0] >= 3 and pv.shape[1] >= 3:
        chi, p = stats.friedmanchisquare(*[pv[c].values for c in pv.columns])
        ranks = pv.rank(axis=1, ascending=False).mean().sort_values()
        out["friedman"] = dict(chi2=float(chi), p=float(p), df=int(pv.shape[1] - 1), mean_ranks=ranks.round(2).to_dict())
    rows = []
    if ref in pv.columns:
        for c in pv.columns:
            if c == ref: continue
            try: w = stats.wilcoxon(pv[ref], pv[c]); rows.append(dict(vs=c, p=float(w.pvalue), mean_diff=float((pv[ref] - pv[c]).mean())))
            except ValueError: pass
        if rows:
            adj = holm([r["p"] for r in rows])
            for r, a in zip(rows, adj): r["p_holm"] = float(a)
    out["wilcoxon"] = rows
    # paired customer-level bootstrap of the PR-AUC difference (works with any number of folds, incl. one)
    def boot_pair(an, bn):
        a_ = preds[preds.model == an].set_index(["fold", "customer_id", "period"]); b_ = preds[preds.model == bn].set_index(["fold", "customer_id", "period"])
        j = a_[["y", "p"]].join(b_[["p"]], rsuffix="_o", how="inner").dropna(subset=["p", "p_o"]).reset_index()
        cust = j.customer_id.unique(); rng = np.random.default_rng(cfg.seed); diffs = []
        grp = {c: d for c, d in j.groupby("customer_id")}
        for _ in range(n_boot):
            s_ = pd.concat([grp[c] for c in rng.choice(cust, len(cust))])
            if s_.y.nunique() < 2: continue
            diffs.append(average_precision_score(s_.y, s_.p) - average_precision_score(s_.y, s_.p_o))
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        return dict(mean=float(np.mean(diffs)), ci95=[lo, hi], excludes_zero=bool(lo > 0 or hi < 0), n_boot=len(diffs))
    try:
        base = res[res.model.str.startswith("B") & (res.model != ref)].groupby("model")["PR_AUC"].mean().sort_values(ascending=False)
        targets = [m for m in res.model.unique() if m.startswith(("B8", "B3p", "B9"))]
        if len(base): targets.append(base.index[0])
        out["bootstrap"] = {o: boot_pair(ref, o) for o in dict.fromkeys(targets)}
    except Exception as e:
        out["bootstrap"] = dict(error=str(e)[:200])
    # key pre-specified pairs: does trajectory help (B9 vs B8)? do segments add beyond trajectory (SMH+ vs B9)? is SMH+ better than v3 SMH?
    pairs = []
    try:
        pp = res.pivot_table(index="fold", columns="model", values=metric)
        for an, bn in ((B9, "B8 Hazard GBDT (no segments)"), (SMHP, B9), (SMHP, "SMH-Churn"), (SMHP, "B3p LightGBM (pooled snapshots)"), (SMHP, "B3 LightGBM (static)")):
            if an in pp.columns and bn in pp.columns:
                dff = (pp[an] - pp[bn]).dropna(); d = dict(a=an, b=bn, mean_diff=float(dff.mean()), wins=int((dff > 0).sum()), n_folds=int(len(dff)))
                try: d["p_wilcoxon"] = float(stats.wilcoxon(pp[an].loc[dff.index], pp[bn].loc[dff.index]).pvalue)
                except Exception: d["p_wilcoxon"] = None
                try: d["bootstrap"] = boot_pair(an, bn)
                except Exception as e: d["bootstrap"] = dict(error=str(e)[:100])
                pairs.append(d)
    except Exception as e:
        pairs = [dict(error=str(e)[:200])]
    out["pairs"] = pairs
    return out

# ----------------------------------------------------------------------------- driver
def run(orders_raw, cfg, outdir, name="data"):
    os.makedirs(outdir, exist_ok=True); CAL['kind'] = cfg.calibration
    orders = to_order_level(orders_raw)
    origins = make_origins(orders, cfg)
    print(f"[{name}] customers={orders.customer_id.nunique():,} orders={len(orders):,} origins={len(origins)}  H={cfg.H} delta={cfg.delta}")
    panel = build_base_panel(orders, origins, cfg)
    folds = make_folds(len(origins), cfg)
    if not folds: raise SystemExit("Not enough origins for the requested folds: lower H/delta/n_valid/n_test or use more data.")
    print(f"[{name}] panel rows={len(panel):,}  churn rate (rows)={panel.y.mean():.3f}  folds={len(folds)}")
    # dataset statistics for Table 3
    stats_tbl = dict(customers=int(orders.customer_id.nunique()), transactions=int(len(orders_raw)), orders=int(len(orders)),
                     first=str(orders.order_date.min().date()), last=str(orders.order_date.max().date()), origins=len(origins),
                     rows=int(len(panel)), churn_rate_rows=float(panel.y.mean()))
    # inter-purchase gap distribution (justification of H)
    gaps = orders["gap"].dropna(); stats_tbl["gap_p50"], stats_tbl["gap_p90"], stats_tbl["gap_p95"] = [float(gaps.quantile(q)) for q in (.5, .9, .95)]
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5)); ax.hist(gaps.clip(upper=gaps.quantile(.99)), bins=50, color="#2A7F8E")
        ax.axvline(cfg.H, color="#C0392B", ls="--", label=f"H = {cfg.H}"); ax.set_xlabel("days between consecutive orders"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"fig_gap_distribution_{name}.png"), dpi=300); plt.close(fig)
    except Exception: pass

    rows, pred_rows, last_info = [], [], None
    for fi, fold in enumerate(folds):
        out, info = run_fold(panel, fold, cfg); te = info["te"]; last_info = info
        for mname, p in out.items():
            r = evaluate(te, p.values, cfg)
            if r is None: continue
            rows.append(dict(dataset=name, fold=fi, model=mname, **r))
            pred_rows.append(pd.DataFrame(dict(fold=fi, model=mname, customer_id=te["customer_id"].values, period=te["period"].values,
                                               y=te["y"].values, p=p.values)))
        print(f"  fold {fi + 1}/{len(folds)} done: test origins {fold['test']}  rows={len(te):,}  skipped={ {k: v for k, v in info.items() if k.endswith('skipped')} }")
    res = pd.DataFrame(rows); preds = pd.concat(pred_rows, ignore_index=True)
    res.to_csv(os.path.join(outdir, f"results_long_{name}.csv"), index=False)

    # ---- Table 6: overall performance
    ms = ["AUC", "PR_AUC", "Brier", "ECE", "Lift10"]
    t6 = res.groupby("model")[ms].agg(["mean", "std"]).round(4); t6.to_csv(os.path.join(outdir, f"table6_overall_{name}.csv"))
    # ---- Table 7: ablation deltas vs full model (paired by fold)
    full = res[res.model == "SMH-Churn"].set_index("fold")
    ab = []
    for mname in [m for m in res.model.unique() if m.startswith(("A", "B8", "B9", "SMH-Churn+"))]:
        o = res[res.model == mname].set_index("fold"); j = full.join(o, rsuffix="_o", how="inner")
        d = dict(variant=mname, dPR_AUC=(j["PR_AUC_o"] - j["PR_AUC"]).mean(), dBrier=(j["Brier_o"] - j["Brier"]).mean(),
                 dVaR10=(j["VaR@10_o"] - j["VaR@10"]).mean(), dRealCap10=(j["RealCap@10_o"] - j["RealCap@10"]).mean())
        try: d["p_wilcoxon"] = stats.wilcoxon(j["PR_AUC_o"], j["PR_AUC"]).pvalue
        except Exception: d["p_wilcoxon"] = np.nan
        ab.append(d)
    pd.DataFrame(ab).round(4).to_csv(os.path.join(outdir, f"table7_ablation_{name}.csv"), index=False)
    # ---- Table 8: targeting
    cols8 = [c for c in res.columns if c.startswith(("VaR@", "RealCap@", "NB@"))]
    res.groupby("model")[cols8].mean().round(4).to_csv(os.path.join(outdir, f"table8_targeting_{name}.csv"))
    # ---- Table 9: transition matrix + segment shares (last fold, train window)
    P, tr = last_info["P"], last_info["tr"]
    P.round(4).to_csv(os.path.join(outdir, f"table9_transition_matrix_{name}.csv"))
    tr["seg"].value_counts(normalize=True).sort_index().round(4).to_csv(os.path.join(outdir, f"table9_segment_shares_{name}.csv"))
    seg_profile = tr.groupby("seg")[["recency", "frequency", "monetary", "avg_gap", "y"]].mean().round(2); seg_profile.to_csv(os.path.join(outdir, f"segment_profile_{name}.csv"))
    # ---- Table 10 + stability
    ex = explain(last_info, cfg, outdir, name)
    # ---- stats + plots
    st = stat_tests(res, preds, cfg); plots(preds, last_info, outdir, name)
    # predicted vs observed retention by segment (last fold, test)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        te = last_info["te"]; fig, ax = plt.subplots(figsize=(6, 4))
        for s, d in te.groupby("seg"):
            g = d.groupby("period")["y"].mean(); ax.plot(range(1, len(g) + 1), 1 - g.values, "o-", label=f"S{int(s)+1} observed retention")
        ax.set_xlabel("test origin"); ax.set_ylabel("share retained over next H days"); ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fig_retention_by_segment_{name}.png"), dpi=300); plt.close(fig)
    except Exception: pass
    summary = dict(dataset=name, cfg=asdict(cfg), stats=stats_tbl, explain=ex, tests=st, silhouette=last_info["sil"])
    with open(os.path.join(outdir, f"summary_{name}.json"), "w") as f: json.dump(summary, f, indent=2, default=str)
    print("\n=== Table 6 (mean over folds) ===\n", res.groupby("model")[ms].mean().round(4).sort_values("PR_AUC", ascending=False))
    print("\n=== Friedman / mean ranks ===\n", st.get("friedman"))
    print("\n=== Bootstrap ===\n", st.get("bootstrap"))
    for pr_ in st.get("pairs", []): print("PAIR", pr_)
    print("\n=== Top-3 SHAP drivers by segment ===\n", ex["top3"], f"\nStability mean={ex['stability_mean']:.3f} min={ex['stability_min']:.3f}")
    return res, preds, summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "online_retail_ii", "dunnhumby", "hm", "csv"])
    ap.add_argument("--path"); ap.add_argument("--mapping", help="JSON mapping for --dataset csv")
    ap.add_argument("--outdir", default="out"); ap.add_argument("--name")
    for k, v in asdict(Cfg()).items():
        if isinstance(v, bool): continue
        if isinstance(v, (int, float, str)): ap.add_argument(f"--{k}", type=type(v), default=v)
    a = ap.parse_args()
    cfg = Cfg(**{k: getattr(a, k) for k in asdict(Cfg()) if hasattr(a, k)})
    CAL['kind'] = cfg.calibration
    if a.dataset == "synthetic": df = make_synthetic()
    elif a.dataset == "online_retail_ii": df = load_online_retail_ii(a.path)
    elif a.dataset == "dunnhumby": df = load_dunnhumby(a.path)
    elif a.dataset == "hm": df = load_hm(a.path)
    else: df = load_generic(a.path, json.loads(a.mapping))
    run(df, cfg, a.outdir, a.name or a.dataset)

if __name__ == "__main__":
    main()
