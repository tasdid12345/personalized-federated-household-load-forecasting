"""Federated next-hour load forecasting: FedAvg MLP vs FedAvg MLP + per-household ridge correction."""
import io, json, os, sys, time, zipfile
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
SEEDS, N_CLIENTS, TEST_H, TRAIN_H = (42, 43, 44), 60, 28 * 24, 52 * 168
ROUNDS, LOCAL_EPOCHS = 10, 2  # identical compute budget for FedAvg and centralized reference
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


def download(url, tries=5):
    for i in range(tries):  # retry with backoff on 429/5xx
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            log("download retry", i, e)
            time.sleep(2 ** i * 2)
    sys.exit("download failed")


# ---------- data ----------
zf = zipfile.ZipFile(io.BytesIO(download(URL)))
name = [n for n in zf.namelist() if n.endswith(".txt")][0]
raw = pd.read_csv(zf.open(name), sep=";", decimal=",", index_col=0, parse_dates=True, dtype=np.float32)
n_raw_rows = len(raw)
hourly = raw.resample("1h").mean()
T = len(hourly)
cut = T - TEST_H  # first test hour; everything before is training period
win = hourly.iloc[cut - TRAIN_H:cut]  # training window used for selection, scaling and fitting
ok = [c for c in hourly.columns if (win[c] == 0).mean() < 0.01 and win[c].notna().all()]
log(f"{len(raw)} raw rows, {T} hourly steps, {len(ok)} eligible clients (selected on training period only)")
if len(ok) < N_CLIENTS:
    sys.exit("not enough eligible clients")
chosen = sorted(np.random.RandomState(42).choice(ok, N_CLIENTS, replace=False))
hr, dw = hourly.index.hour.values, hourly.index.dayofweek.values
time_feats = np.c_[np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                   np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)]


def build(c):
    """Lag features for one client, scaled by its own training mean. Cols: lag1..lag24, lag168, 4 time feats."""
    s = hourly[c].values.astype(np.float64)
    s = np.nan_to_num(s / win[c].mean())
    t = np.arange(168, T)
    X = np.c_[np.stack([s[t - k] for k in range(1, 25)], 1), s[t - 168], time_feats[t]]
    y = s[t]
    tr = (t >= cut - TRAIN_H) & (t < cut)
    te = t >= cut
    return X[tr], y[tr], X[te], y[te]


clients = [build(c) for c in chosen]
n_train, n_test = sum(len(c[1]) for c in clients), sum(len(c[3]) for c in clients)
log(f"{N_CLIENTS} clients, train samples {n_train}, test samples {n_test}")


# ---------- models ----------
def new_mlp(seed):
    return MLPRegressor((64,), batch_size=256, learning_rate_init=1e-3, random_state=seed)


def local_model(g, X, y, seed):
    """Fresh MLP initialised with global weights; partial_fit on 2 rows only builds the structure."""
    m = new_mlp(seed)
    m.partial_fit(X[:2], y[:2])
    m.coefs_, m.intercepts_ = [w.copy() for w in g.coefs_], [b.copy() for b in g.intercepts_]
    for _ in range(LOCAL_EPOCHS):
        m.partial_fit(X, y)
    return m


def fedavg(seed):
    g = new_mlp(seed)
    g.partial_fit(clients[0][0][:2], clients[0][1][:2])
    w = np.array([len(c[1]) for c in clients], float)
    w /= w.sum()
    for r in range(ROUNDS):
        ms = [local_model(g, X, y, seed * 1000 + r * 100 + i) for i, (X, y, _, _) in enumerate(clients)]
        g.coefs_ = [sum(wi * m.coefs_[l] for wi, m in zip(w, ms)) for l in range(len(g.coefs_))]
        g.intercepts_ = [sum(wi * m.intercepts_[l] for wi, m in zip(w, ms)) for l in range(len(g.intercepts_))]
    return g


def centralized(seed):
    X, y = np.vstack([c[0] for c in clients]), np.concatenate([c[1] for c in clients])
    m = new_mlp(seed)
    for _ in range(ROUNDS * LOCAL_EPOCHS):
        m.partial_fit(X, y)
    return m


def personalize(g, X, y):
    """Local ridge on [global pred, lag24, lag168]; alpha chosen by time-series CV on local train data."""
    f = lambda A: np.c_[g.predict(A), A[:, 23], A[:, 24]]
    return f, RidgeCV(alphas=[0.1, 1, 10, 100], cv=TimeSeriesSplit(3)).fit(f(X), y)


def nmae(y, p):
    return np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9)


def summarize(v):
    v = np.sort(v)
    return v.mean(), v[-int(np.ceil(0.1 * len(v))):].mean()


# ---------- run ----------
res = {k: [] for k in ["b", "i", "bw", "iw", "c", "cw", "frac"]}
for seed in SEEDS:
    g = fedavg(seed)
    log(f"seed {seed}: FedAvg done")
    b, i = [], []
    for X, y, Xt, yt in clients:
        f, rg = personalize(g, X, y)
        b.append(nmae(yt, g.predict(Xt)))
        i.append(nmae(yt, rg.predict(f(Xt))))
    cm = centralized(seed)
    c = [nmae(yt, cm.predict(Xt)) for _, _, Xt, yt in clients]
    log(f"seed {seed}: centralized done")
    for k, v in zip("b i c".split(), (b, i, c)):
        res[k].append(summarize(v)[0])
        if k + "w" in res:
            res[k + "w"].append(summarize(v)[1])
    res["frac"].append(float(np.mean(np.array(i) < np.array(b))))
    log(f"seed {seed}: baseline {res['b'][-1]:.4f} improved {res['i'][-1]:.4f} central {res['c'][-1]:.4f}")

m = lambda k: float(np.mean(res[k]))
sd = lambda k: float(np.std(res[k]))
metrics = {
    "baseline_score": m("b"), "improved_score": m("i"), "improvement": m("b") - m("i"),
    "baseline_score_std": sd("b"), "improved_score_std": sd("i"),
    "baseline_worst10_nmae": m("bw"), "improved_worst10_nmae": m("iw"),
    "centralized_nmae": m("c"), "centralized_nmae_std": sd("c"),
    "frac_households_improved": m("frac"),
}
if not all(np.isfinite(v) for v in metrics.values()):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
plt.bar(["FedAvg", "FedAvg+ridge", "Centralized"], [m("b"), m("i"), m("c")],
        yerr=[sd("b"), sd("i"), sd("c")])
plt.ylabel("mean per-household NMAE (test)")
plt.tight_layout()
plt.savefig("figures/nmae.png", dpi=120)

json.dump({
    "question": "Can a federated forecaster with light per-household personalization beat a plain FedAvg neural forecaster on next-hour load across households with very different consumption patterns?",
    "method": "60 UCI electricity clients, hourly, scaled by own training mean; lag features (24 h, same hour last week, hour/day-of-week); last 4 weeks test; FedAvg MLPRegressor vs same global model + local ridge correction; 3 seeds.",
    "dataset": {"name": "UCI Electricity Load Diagrams 2011-2014 (60 clients, hourly)", "source_url": URL,
                "n_rows": int(n_raw_rows), "n_train": int(n_train), "n_test": int(n_test),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "mean per-household NMAE (sum|err|/sum|y|) on last 4 weeks, next-hour",
    "higher_is_better": False,
    "baseline": "Standard FedAvg of scikit-learn MLPRegressor (64 hidden units), fixed rounds, sample-weighted averaging; stand-in for the paper's unstated model.",
    "improved": "Same FedAvg global model and rounds, plus per-household RidgeCV on [global prediction, lag-24h, lag-168h] fitted on local training data only.",
    "metrics": metrics,
    "notes": "n_rows is the number of raw 15-min rows in the file; n_train/n_test are the hourly samples of the 60 selected clients (training uses each client's last 52 weeks before the test period). Clients were chosen at random among those with almost no zero readings in the training window (training data only), so results do not cover late-starting clients. The ridge is fitted on in-sample global predictions of the local training data. One-step-ahead test features use actual past values. Single client subset and single time split; seeds vary only MLP init/shuffling. Centralized MLP uses the same number of epochs as FedAvg rounds x local epochs.",
}, open("results.json", "w", encoding="utf-8"), indent=2)
log("wrote results.json")
