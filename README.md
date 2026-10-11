# Federated household load forecasting

Question: can a federated forecaster with light per-household personalization beat a plain FedAvg neural forecaster on next-hour load?

- Data: UCI Electricity Load Diagrams 2011-2014, 60 clients, hourly; last 4 weeks are the test set.
- Baseline: FedAvg of a scikit-learn MLPRegressor (64 hidden units).
- Improved: same FedAvg global model plus a per-household RidgeCV correction (alpha tuned by time-series CV on local training data).
- Metric: mean per-household NMAE (lower is better), averaged over seeds 42-44.

Run `python analysis.py`; it writes `results.json` and `figures/nmae.png`.

## Results (latest run)

On 60 UCI Electricity 2011-2014 clients, a standard FedAvg MLP forecaster reached 0.120 mean per-household NMAE for next-hour load over the last 4 weeks, versus 0.115 with a local ridge correction: a small gain that improved 68% of households.

| Metric | Value |
|---|---|
| baseline_score | 0.1199 |
| improved_score | 0.1154 |
| improvement | 0.004528 |
| baseline_score_std | 0.001377 |
| improved_score_std | 0.0008328 |
| baseline_worst10_nmae | 0.4491 |
| improved_worst10_nmae | 0.4379 |
| centralized_nmae | 0.1 |
| centralized_nmae_std | 0.003164 |
| frac_households_improved | 0.6778 |

Run on 2026-10-11. Full output: [results.json](results.json) · [run.log](run.log)

Notes: n_rows is the number of raw 15-min rows in the file; n_train/n_test are the hourly samples of the 60 selected clients (training uses each client's last 52 weeks before the test period). Clients were chosen at random among those with almost no zero readings in the training window (training data only), so results do not cover late-starting clients. The ridge is fitted on in-sample global predictions of the local training data. One-step-ahead test features use actual past values. Single client subset and single time split; seeds vary only MLP init/shuffling. Centralized MLP uses the same number of epochs as FedAvg rounds x local epochs.

## Reference

This project is motivated by the following paper:

> CFD Lam, H Wu, Z Pourmirza (2024). *Exploring Federated Learning for Energy Consumption Forecasting in Smart Homes*. 2024 IEEE 8th.

Paper author: Prof. Aad van Moorsel, University of Birmingham (a.vanmoorsel@bham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
