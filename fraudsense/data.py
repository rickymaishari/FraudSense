import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest',
            'oldbalanceDest','newbalanceDest','isFraud','isFlaggedFraud']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_dataset(path):
    d = pd.read_csv(path, dtype={'nameOrig':'string[pyarrow]', 'nameDest':'string[pyarrow]',
        'type':'category', 'step':'int16', 'isFraud':'int8', 'isFlaggedFraud':'int8'})
    if set(REQUIRED) != set(d.columns):
        raise ValueError('Unexpected PaySim columns')
    if not d.isFraud.isin([0,1]).all() or d[REQUIRED].isna().any().any():
        raise ValueError('Missing values or invalid labels')
    if not d.isFlaggedFraud.isin([0,1]).all() or not d.type.isin(['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']).all():
        raise ValueError('Invalid flag labels or transaction categories')
    numeric = d.select_dtypes(include='number')
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
        raise ValueError('Negative or nonfinite dataset fields')
    return d.sort_values('step', kind='stable').reset_index(drop=True)


def boundary(d, fraction):
    counts = d.groupby('step', observed=True).size()
    cumulative = counts.cumsum().iloc[:-1]
    return int((cumulative - len(d) * fraction).abs().idxmin())


def partitions(d):
    """Closest complete-hour split to 80% of RECORDS, not 80% of hours."""
    test_cut = boundary(d, .8)
    dev = d.loc[d.step <= test_cut]
    # 65/10/10/15% of development = fit/cal-fit/cal-check/selection.
    cuts = [boundary(dev, f) for f in [.65,.75,.85]] + [test_cut]
    bins = np.searchsorted(cuts, d.step.to_numpy(), side='left')
    names = ['fit','calibration_fit','calibration_check','selection','test']
    masks = {name: bins == i for i,name in enumerate(names)}
    info = {}
    for name,m in masks.items():
        part = d.loc[m]
        if part.isFraud.nunique() != 2:
            raise ValueError(f'Partition {name} needs both classes')
        info[name] = {'rows':len(part),'frauds':int(part.isFraud.sum()),'prevalence':float(part.isFraud.mean()),
                      'first_hour':int(part.step.min()),'last_hour':int(part.step.max())}
    info['development'] = {'rows':len(dev),'frauds':int(dev.isFraud.sum()),'prevalence':float(dev.isFraud.mean()),
                            'first_hour':int(dev.step.min()),'last_hour':int(dev.step.max())}
    return masks, {'cut_hours':cuts, 'partitions':info, 'development_fraction':len(dev)/len(d),
                   'logic':'Nearest cumulative row count at complete hour; four chronological development blocks', 'seed':42}


def inspect_dataset(d, path):
    stats = {'source':'Local archive PS_20174392719_1491204439457_log.csv.zip; publisher reference https://www.kaggle.com/datasets/ealaxi/paysim1',
             'provenance_limit':'Local filename and content verified; original download chain cannot be independently authenticated.',
             'csv_sha256':sha256(path),'rows':len(d),'frauds':int(d.isFraud.sum()),'prevalence':float(d.isFraud.mean()),
             'columns':list(d.columns),'dtypes':{c:str(t) for c,t in d.dtypes.items()},
             'missing':d.isna().sum().astype(int).to_dict(),'exact_duplicate_rows':int(d.duplicated().sum()),
             'types':d.groupby('type',observed=True).isFraud.agg(['size','sum']).to_dict('index'),
             'first_hour':int(d.step.min()),'last_hour':int(d.step.max()),'hours':int(d.step.nunique()),
             'amount_min':float(d.amount.min()),'amount_max':float(d.amount.max()),
             'sender_repeated_rows':int(d.duplicated('nameOrig',keep=False).sum())}
    return stats
