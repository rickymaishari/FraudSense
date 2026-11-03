from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fraudsense.features import build_features
from fraudsense.model import probabilities
from fraudsense.risk import decision

ROOT=Path(__file__).resolve().parents[1]


class ScreeningService:
    def __init__(self,path=None):
        # Only load bundled, trusted local artifacts. Never accept uploaded pickles.
        self.bundle=joblib.load(path or ROOT/'artifacts/champion.joblib')
        required={'pipeline','columns','calibrator','calibration_kind','policy','coverage','version','anomaly'}
        if not required.issubset(self.bundle):raise ValueError('Incomplete model bundle')

    def predict(self,rows,history=None,explain=False):
        d=pd.DataFrame(rows);h=pd.DataFrame(history) if history else None
        x=build_features(d,h);b=self.bundle;p=probabilities(b,x)
        anomaly=-b['anomaly'].score_samples(x[b['feature_columns']])
        output=[decision(row,features,prob,score,b) for row,features,prob,score in
                zip(rows,x.to_dict('records'),p,anomaly)]
        if explain:
            import shap
            # Permutation SHAP explains the ACTUAL calibrated probability, not a proxy.
            fn=lambda z:probabilities(b,pd.DataFrame(z,columns=b['columns']))
            explainer=shap.Explainer(fn,shap.maskers.Independent(b['background'],max_samples=24),
                                     algorithm='permutation',seed=42)
            for i,result in enumerate(output):
                if result['fraud_probability'] is None:
                    result['explanation']={'status':'withheld','reason':'No trustworthy probability to explain outside evaluated bounds'}
                    continue
                sv=explainer(x.loc[[i],b['columns']],max_evals=2*len(b['columns'])+1)
                vals=sv.values[0];base=float(np.ravel(sv.base_values)[0])
                result['explanation']={'status':'available','method':'Permutation SHAP of calibrated probability',
                    'base_probability':base,'additivity_error':float(abs(base+vals.sum()-p[i])),
                    'features':[{'feature':c,'value':float(x.loc[i,c]),'contribution':float(v)} for c,v in zip(b['columns'],vals)],
                    'note':'Model associations relative to 24 training reference rows, not causal explanations.'}
        return output
