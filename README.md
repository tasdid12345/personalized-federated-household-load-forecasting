# Federated household load forecasting

Question: can a federated forecaster with light per-household personalization beat a plain FedAvg neural forecaster on next-hour load?

- Data: UCI Electricity Load Diagrams 2011-2014, 60 clients, hourly; last 4 weeks are the test set.
- Baseline: FedAvg of a scikit-learn MLPRegressor (64 hidden units).
- Improved: same FedAvg global model plus a per-household RidgeCV correction (alpha tuned by time-series CV on local training data).
- Metric: mean per-household NMAE (lower is better), averaged over seeds 42-44.

Run `python analysis.py`; it writes `results.json` and `figures/nmae.png`.

## Reference

This project is motivated by the following paper:

> CFD Lam, H Wu, Z Pourmirza (2024). *Exploring Federated Learning for Energy Consumption Forecasting in Smart Homes*. 2024 IEEE 8th.

Paper author: Prof. Aad van Moorsel, University of Birmingham (a.vanmoorsel@bham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
