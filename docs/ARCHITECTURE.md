# Architecture and interface contract

`frontend/app.py` is presentation-only. It calls FastAPI over localhost HTTP.
`backend/schemas.py` validates finite nonnegative amounts/balances, supported
types, integer hours and nonblank identifiers. Labels and post-balances are
ignored extras. Request parsing failures return HTTP 422 with a null estimate
and a cannot-assess action. Missing balance is accepted but causes a hold and
withheld probability. Model loading failure gives 503 readiness.

`fraudsense/features.py` computes the same named float32 predictors for training,
single requests and CSV batches. Aggregation happens by entity and complete
hour, with shifted cumulative statistics. Twenty-four-hour history includes
`step-24` and excludes `step`. Previous amount is last-hour mean, never an
invented within-hour last transaction. Sender/recipient identifiers are never
numeric model inputs. Full evaluation uses unlabeled earlier test transactions
as causal history, which corresponds to sequential deployment.

`fraudsense/service.py` loads one trusted versioned bundle with model,
calibration, feature order, thresholds, fitting coverage, Isolation Forest and
SHAP background. It does not persist uploaded data. Each batch computes history
from its valid earlier rows; independent requests are stateless.

Decision priority: invalid input -> cannot assess; missing balance / balance
shortfall / unsupported coverage -> validation hold; supported high model alert
-> priority review; lower model alert -> verification; otherwise standard checks.
Anomalies are informational because development evidence did not justify added
operational alerts. Batch errors remain row-level and are excluded from history.

## API

| Route | Contract |
|---|---|
| GET `/health` | Readiness, version, request-row count, uptime |
| GET `/model/info` | Feature names, thresholds, coverage, calibration |
| GET `/model/metrics` | Actual experiment and test results |
| GET `/model/plots/{name}` | Allowlisted PNG report plots |
| POST `/predict` | JSON `transaction` plus optional `history` list |
| POST `/explain` | Same input; calibrated-probability permutation SHAP |
| POST `/predict/batch` | Multipart `file`, CSV <=5 MB and <=5,000 rows |

Single response fields are typed by `Assessment`: schema/model versions,
nullable fraud probability, probability status, model alert, independent
validation/anomaly objects, final action, reasons, next step, limitations and
nullable explanation. Batch responses preserve original row numbers and contain
either this assessment or a row error. See generated `/docs` for full schemas.

No account credentials, banking integration, hidden historical lookup, external
telemetry, model uploads or arbitrary file path routes are provided. Local
joblib artifacts must be trusted because pickle-based loading can execute code.

## Memory and reproducibility

Arrow-backed identifier strings, int16 steps, int8 labels and float32 engineered
features reduce memory. Repeated-entity aggregation avoids millions of Python
account dictionaries. Logistic regression scales on fitting rows only. Random
Forest uses 64 trees with 25% bootstrap draws per tree from the entire fitting
population; this is an algorithmic training parameter, not a discarded dataset
subset. XGBoost uses histograms and four threads. Isolation Forest is deliberately
trained on a fixed 100,000-row sample of the fitting block and evaluated on the
full selection/test blocks. No SMOTE or label resampling is used.

The 80% development partition includes calibration and selection; supervised
base estimators fit the earlier 3,283,561 rows. Refitting on calibration rows
would invalidate the saved calibrator, so the base estimator is not refit after
selection. Exact partitions and hyperparameters are in the new reports.
