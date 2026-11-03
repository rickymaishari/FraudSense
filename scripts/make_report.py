"""Render a technical report directly from measured experiment JSON files."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'reports/new_experiments'
def read(name):return json.loads((P/name).read_text())

def main():
    r=read('results.json');q=read('dataset.json');m=r['test'];splits=r['split']['partitions']
    comparison=read('model_comparison.json');common=read('common_test_comparison.json')
    ablation=read('matched_rf_ablation.json') if (P/'matched_rf_ablation.json').exists() else None
    cold=read('cold_start_test.json') if (P/'cold_start_test.json').exists() else None
    old=read('old_models_common_test.json') if (P/'old_models_common_test.json').exists() else []
    operational=read('operational_test.json') if (P/'operational_test.json').exists() else None
    text=['# FraudSense 2 — measured technical report','',
      'Run date: 25 October 2026. All values below are read from executed experiment outputs. No target accuracy was imposed.',
      '', '## Dataset and provenance','',
      f"Verified {q['rows']:,} rows and {q['frauds']:,} fraud labels ({q['prevalence']:.6%}), simulation hours {q['first_hour']}–{q['last_hour']}. No nulls or exact duplicate rows were found.",
      f"CSV SHA-256: `{q['csv_sha256']}`.",
      'Local archive: `Downloads/PS_20174392719_1491204439457_log.csv.zip`; complete ZIP CRC passed. Publisher reference: https://www.kaggle.com/datasets/ealaxi/paysim1. The local download chain is not independently authenticated.',
      '', '## Chronological development and independent test','',
      '| Partition | Rows | Frauds | Prevalence | Hours |','|---|---:|---:|---:|---|']
    for name,v in splits.items():text.append(f"| {name} | {v['rows']:,} | {v['frauds']:,} | {v['prevalence']:.6%} | {v['first_hour']}–{v['last_hour']} |")
    text += ['',f"Development contains {r['split']['development_fraction']:.6%} of all rows. The nearest whole-hour boundary to 80% is after hour 354. No hour is divided. Test fraud prevalence is {splits['test']['prevalence']/splits['development']['prevalence']:.2f} times development prevalence.",
      'The supervised base estimators fit the first 3,283,561 records. The rest of the approximately 80% development partition is reserved for fitting calibration, checking calibration, and model/threshold selection. Calibration rows are not folded back into model fitting. All random seeds are 42.',
      '', '## Model comparisons on the same selection block','',
      '| Candidate | AP | Precision | Recall | F1 | Training seconds |','|---|---:|---:|---:|---:|---:|']
    for v in comparison:
        z=v['validation'];text.append(f"| {v['name']} | {z['average_precision']:.6f} | {z['precision']:.6f} | {z['recall']:.6f} | {z['f1']:.6f} | {v['train_seconds']:.2f} |")
    text += ['',f"Champion: **{r['champion']}**. Selection used highest development AP, with F1 as tie-break. Ablations were diagnostic, not eligible champions. No test result selected the model.",
      'Random Forest: 64 trees, max depth 14, minimum leaf size 5, class_weight=balanced_subsample, max_samples=0.25, four threads. Each tree draws a bootstrap sample from the entire fitting population. Logistic regression uses fitting-only StandardScaler, C=0.1 and balanced classes. XGBoost compares depths 4/6 with 180 histogram trees, learning rate 0.07 and square-root class-imbalance weighting. Exact parameters are saved in `model_comparison.json` and model artifacts.',
      '', '## Final untouched test','',
      '| Metric | Value |','|---|---:|']
    for key in ['precision','recall','f1','average_precision','roc_auc','brier','log_loss','false_alarms_per_1000','inference_ms_per_row']:
        text.append(f'| {key} | {m[key]:.10g} |')
    text += [f"| True positives | {m['tp']:,} |",f"| False positives | {m['fp']:,} |",f"| False negatives | {m['fn']:,} |",f"| True negatives | {m['tn']:,} |",'',
      '![Final test curves and confusion matrix](new_experiments/test_performance.png)',
      'Inference timing is model scoring per row in a large batch; it excludes feature generation, HTTP, anomaly scoring and SHAP. UI latency is therefore higher.',
      'Every test label, score, source row ID and support flag is saved in `test_predictions.parquet`. Error rows are in `false_positives_and_negatives.csv`. Subgroup reports are `by_type.json` and `by_time.json`; review-budget metrics are in `review_budgets.json`.',
      '', '## Calibration and thresholds','',
      f"Selected calibrator: **{r['calibration']['chosen']}**. Sigmoid and isotonic were fitted only on hours 252–280. Raw and calibrated scores were compared on disjoint later hours 281–307. Candidates must not worsen either Brier score or log loss; lowest log loss selects among eligible options.",
      '| Independent calibration check | Brier | Log loss |','|---|---:|---:|']
    for name,v in r['calibration']['independent_check'].items():text.append(f"| {name} | {v['brier']:.10g} | {v['log_loss']:.10g} |")
    text += ['',f"Frozen high threshold: {r['policy']['high']}; verification threshold: {r['policy']['low']}. High maximizes selection F1 subject to 0.5% review budget; low respects whole tied-score groups. Actual selection alert rate: {r['policy']['selection_review_rate']:.6%}. This is a validation target, not a future workload guarantee.",
      '![Independent development calibration](new_experiments/calibration_development.png)',
      '![Untouched test calibration](new_experiments/calibration_test.png)',
      'Reliability tables include bin counts because rare events and probability ties can make calibration plots deceptively reassuring. Isotonic can return endpoint values 0/1; these are model estimates, never certainty. Final test proper scores are reported separately from the independent development check.',
      '', '## Coverage and operational behavior','',
      f"Serving withholds {r['test_withheld']} of the final test estimates, including {r['test_withheld_frauds']} labelled frauds; supported coverage is {r['test_coverage']:.6%}. Supported-only precision is {r['test_supported']['precision']:.6%}, recall {r['test_supported']['recall']:.6%}. The all-row benchmark includes raw model estimates for diagnostic completeness; the UI never displays those withheld estimates.",
      'Coverage checks use fitting-only per-type amount, opening balance and amount/balance ratio ranges. Missing pre-balance and simulation hours outside the evaluated PaySim period also withhold estimates. These checks are necessary but cannot detect every multivariate or real-world shift.',
      'The required CASH_OUT of 10,000,000,000 with balance 1,000 at step 17 exceeds the fitting CASH_OUT maximum of 10,000,000 and recorded funds. API and browser show **Payment on hold — validation required.** and **Fraud probability: N/A — Outside evaluated model range.**',
      'Operational balance shortfalls can occur for legitimate-labelled PaySim rows. No rule turns such a shortfall into proof of fraud. Anomaly scores are never probabilities.',
      '', '## Ablation and simulator shortcut','',
      f"Among fitting frauds, {r['shortcut']['fit']['fraud_equal']:,}/{r['shortcut']['fit']['frauds']:,} exactly deplete opening balance within one cent; none of {r['shortcut']['fit']['legitimate']:,} legitimate fitting rows do. In the final test, {r['shortcut']['test']['fraud_equal']:,}/{r['shortcut']['test']['frauds']:,} frauds deplete balance and zero legitimate rows do. This is a severe synthetic shortcut.",
      '| Fixed candidate on common test | AP | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    for v in common:text.append(f"| {v['name']} (raw) | {v['average_precision']:.6f} | {v['precision']:.6f} | {v['recall']:.6f} | {v['f1']:.6f} |")
    if ablation:
        z=ablation['test'];text += ['',f"Matched Random Forest without balance: test AP **{z['average_precision']:.6f}**, precision {z['precision']:.6%}, recall {z['recall']:.6%}. Same fixed forest hyperparameters and partitions, with balance-derived predictors removed. This supplemental diagnostic ran after the main test was opened; it did not reselect or modify the champion. Its threshold still uses development selection only."]
    text += ['','Removing balance features sharply degrades performance. These results must not be advertised as production financial-fraud accuracy. Balance-depleting legitimate counterfactual inputs can receive fraud alerts even though their operational balance check passes.','', '## Anomaly and cold-start checks','',
       f"Isolation Forest validation AP: {r['anomaly_validation']['average_precision']:.6f}; it added {r['anomaly_validation']['additional_alerts']:,} reviews but **{r['anomaly_validation']['additional_frauds']}** additional fraud detections beyond supervised alerts. Therefore it is informational only. Final test anomaly AP: {r['anomaly_test_average_precision']:.6f}. The detector fits a seeded 100,000-row sample of the fitting block, not the full population."]
    if cold:text += ['',f"Standalone no-history predictions were evaluated on every final test row using the shared feature builder. Precision {cold['precision']:.6%}, recall {cold['recall']:.6%}, AP {cold['average_precision']:.6f}; TP {cold['tp']}, FP {cold['fp']}, FN {cold['fn']}. These are a separate inference condition, not the main chronological-history benchmark. Scores are saved alongside chronological scores."]
    text += ['','## Old models on an identical new test','',
      'Historical subset-test metrics are not compared directly with full-data metrics. The frozen old artifacts were instead scored on the exact new final rows, retaining their old calibration and thresholds. Training sizes differ, so this is an external evaluation of fixed systems, not a controlled architecture-only experiment.',
      '| Old artifact | AP | Precision | Recall |','|---|---:|---:|---:|']
    for v in old:text.append(f"| {v['artifact']} | {v['average_precision']:.6f} | {v['precision']:.6f} | {v['recall']:.6f} |")
    if operational:
        text += ['','## Operational workload is separate from fraud alerts','',
          f"The final decision rules hold **{operational['held_rows']:,}** test transactions for validation; **{operational['held_legitimate_labels']:,}** have legitimate labels. Balance shortfalls account for {operational['balance_shortfalls']:,} rows. This large operational workload reflects PaySim balance inconsistencies and the requested conservative validation rule, not model fraud false positives. A real ledger integration would be required to resolve these holds.",
          '| Final action | Test rows |','|---|---:|']
        for key,value in operational['action_counts'].items():text.append(f'| {key} | {value:,} |')
    text += ['','## Explainability','',
      'Permutation SHAP explains the actual calibrated probability with 24 training reference rows. Additivity is tested; unsupported estimates have no SHAP probability explanation. The global illustration uses six fraud and six legitimate examples, deliberately not a prevalence-representative population sample.',
      '![SHAP diagnostic sample](new_experiments/shap_global.png)',
      '', '## Reproducibility and practical limits','',
      f"Main full-data run: {r['elapsed_seconds']:.2f} seconds on the available Windows machine, Python {r['python']}. This is an observed run time, not a hardware-independent promise. Saved artifacts load without retraining. Exact environment is in `requirements-lock.txt`; source, input and artifact hashes are recorded.",
      'The full raw CSV and feature cache are excluded from the ZIP. Arrow strings, compact numeric types, four training threads and bounded tree/sample settings make execution practical on the available approximately 16 GB machine. Peak process memory was not instrumented, so no measured peak-RAM claim is made.',
      'No real financial data, customer identity lookup, persistent ledger, public hosting, authentication, drift retraining, bank integration or external validity has been demonstrated. The app must be kept on localhost. Range checks and calibration do not guarantee individual predictions. Final results should be replicated on independent populations before any consequential use.',
      'See `QA_REPORT.md` for actual tests, browser evidence, resolved failures and packaging verification.']
    (ROOT/'reports/TECHNICAL_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    print('Technical report generated from measured outputs')

if __name__=='__main__':main()
