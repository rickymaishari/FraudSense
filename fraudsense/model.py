import numpy as np


def calibrate(raw, kind, calibrator):
    p=np.clip(np.asarray(raw,dtype=float),1e-7,1-1e-7)
    if kind=='sigmoid':
        p=calibrator.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
    elif kind=='isotonic':
        p=calibrator.predict(p)
    else:
        p=np.asarray(raw,dtype=float)
    return np.round(p,12)


def probabilities(bundle,x):
    return calibrate(bundle['pipeline'].predict_proba(x[bundle['columns']])[:,1],
                     bundle['calibration_kind'],bundle['calibrator'])
