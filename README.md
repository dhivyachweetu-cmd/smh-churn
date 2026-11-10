SMH-Churn
Code and configuration for the manuscript "Segment-Migration-Aware Discrete-Time Hazard Learning for Explainable and Value-Driven Customer Churn Prediction in Non-Contractual Retail" (M. Dhivyaa, V. Aruna, Panimalar Engineering College, Chennai).
SMH-Churn is a discrete-time hazard model for churn in non-contractual retail. It adds segment-migration features to a LightGBM hazard learner, trains on a leakage-safe rolling-origin person-period panel, ranks customers by value-at-risk under a contact budget, and explains predictions with TreeSHAP.
What is in this repository
File	Description
`SMH_Churn_Colab.ipynb`	Colab notebook containing the full pipeline: data loaders, rolling-origin folds, segmentation, migration features, hazard model, baselines (B1–B8), ablations (A1–A4), targeting and SHAP
`requirements.txt`	Pinned package versions
`LICENSE`	MIT License
`table6_overall_*.csv`, `table7_ablation_*.csv`, `table8_targeting_*.csv`, `table9_*.csv`	Result tables for D1 and D2 (Tables 6–9 of the paper)
`results_long_*.csv`, `summary_*.json`	Per-model metrics and run summaries (configuration and dataset statistics)
`shap_*.csv`, `segment_profile_*.csv`	SHAP importances and segment profiles
`fig_*.png`	Figures used in the paper (gap distribution, reliability, retention by segment, SHAP)
File names ending in `D1_online_retail_ii` belong to Online Retail II, and those ending in `D2_dunnhumby` to Dunnhumby.
Datasets (not redistributed here)
Download them yourself and upload them to Colab when the notebook asks for them.
D1 – Online Retail II, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/502/online+retail+ii
D2 – The Complete Journey, dunnhumby: https://www.dunnhumby.com/source-files/
Preprocessing is done inside the loaders in the notebook. D1: cancelled invoices (starting with `C`), non-positive quantity or price, and rows without a customer ID are dropped. D2: rows with non-positive sales value are dropped and the day index is mapped to calendar dates. Line items are collapsed to one row per order.
Environment
Experiments in the paper were run in Python 3.13 on Google Colab (CPU only, 2 vCPUs, 13.6 GB RAM) with:
```
scikit-learn==1.6.1
lightgbm==4.6.0
shap==0.54.1
lifelines==0.30.3
lifetimes==0.11.3
```
Install with:
```bash
pip install -r requirements.txt
```
How to run
Download the two datasets (links above).
Open `SMH_Churn_Colab.ipynb` in Google Colab (Open in Colab, or upload the file at colab.research.google.com).
Upload the dataset files when the notebook asks for them, or place them in the folder it points to.
Choose the dataset (D1 or D2) in the first configuration cell and select Runtime → Run all.
The notebook writes the tables, figures and JSON summaries listed below.
Configuration used in the paper
Setting	D1 Online Retail II	D2 Dunnhumby
Churn horizon H (days)	90	60
Origin spacing Δ (days)	30	28
Origins (at-risk rows)	18 (18,664)	19 (34,363)
Fold design	validation 3 origins, test 3 origins, gaps of 3 origins, first training block 6 origins	same
Folds that fit	1	1
Test rows (churn rate)	1,985 (15.1%)	4,724 (1.5%)
Segments K / look-back L	4 (K-means) / 3	4 (K-means) / 3
Margin / save rate / contact cost	0.2 / 0.3 / 1.0	0.2 / 0.3 / 1.0
Learner	LightGBM, 400 trees (early stopping, 50 rounds), learning rate 0.05, 31 leaves, subsample 0.8, colsample 0.8	same
Calibration	Platt scaling on the validation block	same
Tuning	none (0 Optuna trials)	none
Random seed	42	42
All reported numbers come from a single run with seed 42. Each dataset yields one rolling-origin fold, so the tables contain single values (no standard deviations), and the Friedman, Wilcoxon and bootstrap tests are not reported. To get several folds, reduce `n_valid`, `n_test`, `min_train`, H or Δ in the config.
Outputs
Each run writes:
`table6_overall_*.csv` – overall performance (AUC, PR-AUC, Brier, ECE, Lift@10%) for all models (Table 6 in the paper)
`table7_ablation_*.csv` – ablations A1–A4 and B8 (Table 7)
`table8_targeting_*.csv` – value-at-risk capture and net benefit at 5/10/20% contact budgets (Table 8)
`table9_segment_shares_*.csv`, `table9_transition_matrix_*.csv` – segment shares and migration matrix (Table 9)
`shap_global_*.csv`, `shap_by_segment_*.csv`, `segment_profile_*.csv`
`results_long_*.csv` – per-fold, per-model metrics
`summary_*.json` – configuration and dataset statistics
`fig_*.png` – reliability diagrams, gap distribution, retention by segment, SHAP summaries
Citation
If you use this code, please cite the manuscript:
```
M. Dhivyaa and V. Aruna, "Segment-Migration-Aware Discrete-Time Hazard Learning for
Explainable and Value-Driven Customer Churn Prediction in Non-Contractual Retail",
manuscript under review, 2026.
```
License
Code released under the MIT License (see `LICENSE`). The datasets are subject to their providers' terms of use.
Contact
M. Dhivyaa – Department of Computer Science and Engineering, Panimalar Engineering College, Chennai, India.
