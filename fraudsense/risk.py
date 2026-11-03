"""Operational validity, model support, and fraud alerts are independent."""
import numpy as np

OUTGOING={'CASH_OUT','TRANSFER','PAYMENT','DEBIT'}


def support_reasons(row,features,bundle):
    reasons=[]
    coverage=bundle['coverage'].get(row['type'])
    if coverage is None:
        return ['Unsupported transaction type']
    if row.get('oldbalanceOrg') is None or not np.isfinite(float(row.get('oldbalanceOrg',np.nan))):
        reasons.append('Opening balance unavailable; this model requires an observed pre-transaction balance')
    for col,(low,high) in coverage.items():
        value=float(features[col])
        # float32 range tolerance reflects actual trained representation.
        if value < low-1e-5*max(1,abs(low)) or value > high+1e-5*max(1,abs(high)):
            reasons.append(f'{col} outside fitting-data range [{low:g}, {high:g}] for {row["type"]}')
    if not bundle['step_range'][0] <= row['step'] <= bundle['step_range'][1]:
        reasons.append('Simulation hour outside evaluated PaySim period')
    return reasons


def supported_rows(d,x,bundle):
    supported=np.ones(len(d),dtype=bool)
    for t,coverage in bundle['coverage'].items():
        m=(d.type==t).to_numpy()
        for col,(low,high) in coverage.items():
            a=x[col].to_numpy()
            supported[m] &= (a[m]>=low-1e-5*max(1,abs(low))) & (a[m]<=high+1e-5*max(1,abs(high)))
    supported &= d.type.isin(bundle['coverage']).to_numpy()
    supported &= x.balance_missing.to_numpy()==0
    supported &= d.step.between(*bundle['step_range']).to_numpy()
    return supported


def decision(row,features,p,anomaly_score,bundle):
    outside=support_reasons(row,features,bundle)
    balance=row.get('oldbalanceOrg');warnings=[];codes=[]
    if balance is None:
        warnings.append('Recorded opening balance is missing');codes.append('MISSING_BALANCE')
    elif row['type'] in OUTGOING and row['amount']>balance:
        warnings.append('Amount exceeds recorded available balance; verify the ledger and funding');codes.append('BALANCE_SHORTFALL')
    if outside:
        warnings.extend(outside);codes.append('OUTSIDE_EVALUATED_RANGE')
    probability=None if outside else float(p)
    tier='unavailable' if outside else ('priority' if p>=bundle['policy']['high'] else 'verification' if p>=bundle['policy']['low'] else 'no_model_alert')
    if codes:
        action='Payment on hold — validation required.';next_step='Validate the input and available funds before any processing.'
    elif tier=='priority':
        action='Priority fraud review';next_step='Request human fraud review before processing.'
    elif tier=='verification':
        action='Additional verification';next_step='Verify the sender and payment details.'
    else:
        action='No model alert / standard checks';next_step='Continue standard transaction checks; this is not authorization or guaranteed legitimacy.'
    anomalous=anomaly_score>=bundle['anomaly_threshold']
    reasons=list(warnings)
    if tier in ['priority','verification']:reasons.append('Model score exceeds the frozen validation-selected alert threshold')
    if anomalous:reasons.append('Isolation Forest detects an unusual pattern; this is not a fraud probability')
    if not features['history_available']:reasons.append('No earlier sender activity in the supplied history; cold-start features are used')
    return {'schema_version':'2.0','model_version':bundle['version'],'fraud_probability':probability,
      'probability_status':'N/A — Outside evaluated model range.' if outside else 'Model estimate within evaluated input bounds; uncertainty remains',
      'model_alert':tier,'validation':{'status':'hold' if codes else 'checks_passed','codes':codes,'warnings':warnings},
      'anomaly':{'status':'unusual' if anomalous else 'not_flagged','score':float(anomaly_score),
                 'threshold':bundle['anomaly_threshold'],'affects_action':False},
      'final_action':action,'reasons':reasons,'next_step':next_step,
      'limitations':['Synthetic PaySim demonstration, not validated for real banking.',
        'Balance features can exploit simulator-specific patterns.',
        'Range checks do not detect every distribution shift; no score guarantees legitimacy.',
        'History is supplied explicitly; no live bank ledger or persistent account history is connected.'],
      'explanation':None}
