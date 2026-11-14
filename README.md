# SMH-Churn

Code and results for the paper *Leakage-Safe Discrete-Time Hazard Learning for Explainable and Value-Driven Customer Churn Prediction in Non-Contractual Retail: Evidence on Segment-Migration Features from Three Public Datasets* (M. Dhivya, V. Aruna).

## Contents
| Path | What it is |
|---|---|
| `smh_churn_v4.py` | Full pipeline (rolling-origin protocol, hazard models, baselines, ablations, tests, SHAP). Version used in the paper. |
| `SMH_Churn_Colab_v4.ipynb` | Google Colab notebook that runs the pipeline on D1, D2 and D3. |
| `scripts/make_hm_sample.py` | Draws the 50,000-customer sample of the H&M data (seed 42). |
| `results/out_d1_v4`, `out_d2_v4`, `out_d3_v4` | Tables, figures and run summaries for each dataset. |
| `requirements.txt` | Python packages. |

## Datasets (not included; download from the sources)
- **D1** UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+II
- **D2** Dunnhumby The Complete Journey: https://www.dunnhumby.com/source-files/ (`transaction_data.csv`)
- **D3** H&M Personalized Fashion Recommendations: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations (`transactions_train.csv`; sample with `scripts/make_hm_sample.py`)

## Running
Open the notebook in Colab, set `QUICK=True` for a fast check, then `QUICK=False` for the full run. Settings used in the paper: D1 H=90, step 15; D2 H=60, step 14; D3 H=90, step 15; K=4, L=3, seed 42. D1 and D3 used 15 Optuna trials per model and fold; D2 was run untuned.

Command line:
```bash
python smh_churn_v4.py --dataset online_retail_ii --path online_retail_II.xlsx --outdir out_d1_v4
```
(see `python smh_churn_v4.py --help` for the other options).

## Results files
Per dataset: `table6_overall` (Table 5 in the paper), `table7_ablation`, `table8_targeting`, `table9_*` (segments and transitions), `results_long` (fold-level metrics), `shap_*`, `summary_*.json` (tests, settings, data statistics) and figures.

## Citation
Please cite the paper once published.
