# Decisions

A short log of decisions that are not obvious from the code: what we chose and why.
Newest first. Add an entry whenever you settle something the other person (or the jury) might ask about.

## 2026-11-10 · Models (Dennis)

- **Two tasks, one pipeline.** `direction` (will the next 15m candle close higher? horizon 1) and `volatility`
  (std of the next four 15m log returns, horizon 4). Both use `train_model.py` / `predict_model.py`; a task is one
  entry in `TASKS`.
- **Split by time, never shuffled:** 70 % train, 15 % validation, 15 % test, the same cut-off dates for all symbols.
  The last `horizon` candles before each cut are left out, so no training label uses prices from the next part.
- **Model choice by validation score only** (direction: AUC, volatility: RMSE). The chosen model is refit on
  train + validation and scored once on test. The test score never influences a choice.
- **Baselines stay simple and are always reported:** direction = share of up candles, volatility = the last
  4 hours (`rv_16`). A model is only interesting if it beats them.
- **Small, regularised models** (logistic regression / ridge, small gradient-boosted trees). 15-minute returns are
  mostly noise, and Nicolas grades the pipeline, not the score.
- **Volatility is learned on the log scale** and turned back with `exp()`: it is positive and right-skewed.
- **One model per symbol is active;** a shared model can be trained (`--shared`) for comparison and is used as a
  fallback for symbols without their own model.
- **`scikit-learn` is pinned** in `requirements.txt`, and `predict_model.py` refuses a model file saved with a
  different version: a saved model only loads safely with the version that trained it.
- **Model files are not in git** (`*.joblib` is ignored). `model_registry` records which file belongs to which
  model; everyone trains against their own database.
- **The API never loads a model.** `predict_model.py --all` writes predictions into `predictions`; the API reads the
  newest row of the active model (decided with PR #7).
