# FraudSense

FraudSense is a Machine Learning course project that screens synthetic mobile-money transactions. A FastAPI service handles individual transactions and CSV batches. A Streamlit dashboard shows the model score, separate validation checks, and the suggested screening action. The explanation workflow uses SHAP to show which features influenced a prediction.

## Experiment

The team compared Logistic Regression, Random Forest, and XGBoost on PaySim. The dataset contains 6,362,620 simulated transactions and 8,213 fraud labels. History features use strictly earlier hours. The chronological split separates training, model selection, probability calibration, threshold selection, and the final holdout.

| Final holdout result | Value |
| --- | ---: |
| Transactions | 1,293,523 |
| Average precision | 0.998591 |
| Precision | 0.999295 |
| Recall | 0.998356 |
| True positives / false positives / false negatives | 4,251 / 3 / 7 |
| Average precision without balance-derived features | 0.384915 |

These are archived results for the isotonic-calibrated Random Forest and its matched ablation. In the holdout, 4,140 of 4,258 fraud cases nearly depleted the sender's opening balance, while no legitimate cases matched that pattern. The ablation shows how much the benchmark depended on that simulator signal. The separate validation policy also held 726,426 transactions, which matters when evaluating review workload. The benchmark is evidence about synthetic PaySim data.

## Run locally

Use Python 3.13. Create and activate a virtual environment, then install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Place your trusted project model at `artifacts/champion.joblib`, or obtain the full PaySim CSV separately and train your own model:

```sh
python scripts/train_full.py --data data/PS_20174392719_1491204439457_log.csv
```

See [dataset attribution](data/README.md). The public repository includes source code and selected aggregate experiment summaries. The dataset and serialized model files are separate inputs.

From the repository root, run each command in its own terminal:

```sh
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
python -m streamlit run frontend/app.py --server.port 8501
```

Open http://127.0.0.1:8501 for the dashboard or http://127.0.0.1:8000/docs for the API. Use synthetic examples such as [the sample batch](docs/batch_demo.csv). To run the tests, install `requirements-dev.txt` and run `python -m pytest`. Some tests require the trained artifacts.

## Team

Ricky Maishari, Kashish Jain, and Subhasish Behera.
