# Federated load forecasting with light per-household personalization

Motivation: the paper "Exploring Federated Learning for Energy Consumption Forecasting in Smart Homes" (2024) studies federated learning for household energy forecasting. Households differ a lot in consumption patterns, so a single global model may fit some of them poorly. This experiment tests whether a cheap local correction helps.

## Question
Can a federated forecaster with light per-household personalization beat a plain FedAvg neural forecaster on next-hour load across households with very different consumption patterns?

## Dataset
UCI Electricity Load Diagrams 2011-2014 (370 clients, 15-minute readings), downloaded from the UCI archive. 60 clients are sampled, resampled to hourly and scaled by each client's own training mean. The last 4 weeks of each household are the test set. This is **not** the paper's own dataset.

## Baseline vs improved
- **Baseline:** standard FedAvg with a scikit-learn MLPRegressor (each household is a client, fixed number of rounds). It is a stand-in for the paper's unstated model.
- **Improved:** the same FedAvg global model and rounds, plus a per-household ridge correction fitted only on that household's local training data (global prediction plus seasonal lags). No raw data is shared.
- Main metric: mean per-household normalized MAE (lower is better). Secondary: worst-10% household NMAE and a centralized MLP reference. Models are run with seeds 42, 43, 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> CFD Lam, H Wu, Z Pourmirza (2024). *Exploring Federated Learning for Energy Consumption Forecasting in Smart Homes*. 2024 IEEE 8th.

Paper author: Prof. Aad van Moorsel, University of Birmingham (a.vanmoorsel@bham.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
