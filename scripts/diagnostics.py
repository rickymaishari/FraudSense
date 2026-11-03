"""Pre-specified diagnostic ablation and cold-start analysis; never reselects champion."""
import sys,json,argparse,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from fraudsense.data import load_dataset,partitions
from fraudsense.features import build_features,BALANCE
from fraudsense.model import probabilities,calibrate
from fraudsense.evaluation import metrics,choose_thresholds
from fraudsense.service import ScreeningService

def main(data,old_dir):
    report=ROOT/'reports/new_experiments';d=load_dataset(data);masks,_=partitions(d)
    x=pd.read_parquet(ROOT/'data/features.parquet');b=joblib.load(ROOT/'artifacts/champion.joblib')
    fit,val,test=masks['fit'],masks['selection'],masks['test'];y=d.isFraud.to_numpy()
    print('Matched Random Forest ablation, fixed hyperparameters; no champion selection',flush=True)
    cols=[c for c in x.columns if c not in BALANCE]
    model=RandomForestClassifier(n_estimators=64,max_depth=14,min_samples_leaf=5,max_samples=.25,
                                 class_weight='balanced_subsample',n_jobs=4,random_state=42)
    tick=time.perf_counter();model.fit(x.loc[fit,cols],y[fit]);elapsed=time.perf_counter()-tick
    vp=model.predict_proba(x.loc[val,cols])[:,1];policy=choose_thresholds(y[val],vp)
    tp=model.predict_proba(x.loc[test,cols])[:,1]
    result={'purpose':'Fixed-hyperparameter diagnostic; trained after main evaluation; no model or threshold changes to champion',
      'model':'random_forest_without_balance','train_seconds':elapsed,'policy':policy,
      'validation':metrics(y[val],vp,policy['high']),'test':metrics(y[test],tp,policy['high'])}
    (report/'matched_rf_ablation.json').write_text(json.dumps(result,indent=2))
    joblib.dump({'pipeline':model,'columns':cols},ROOT/'artifacts/ablation_rf_no_balance.joblib',compress=3)
    print('Cold-start standalone inference evaluated on every final test row',flush=True)
    tx=d.loc[test].reset_index(drop=True);cold=[]
    for start in range(0,len(tx),100000):
        chunk=tx.iloc[start:start+100000].copy()
        # Replace join keys with unique tokens to reproduce standalone no-history
        # features for every row using the identical shared builder.
        ids=np.arange(len(chunk)).astype(str);chunk['nameOrig']=ids;chunk['nameDest']=ids
        cx=build_features(chunk);cold.append(probabilities(b,cx))
    cp=np.concatenate(cold);coldmetrics=metrics(y[test],cp,b['policy']['high'])
    (report/'cold_start_test.json').write_text(json.dumps(coldmetrics,indent=2))
    saved=pd.read_parquet(report/'test_predictions.parquet');saved['cold_start_probability']=cp
    saved.to_parquet(report/'test_predictions.parquet',index=False)
    if old_dir:
        print('Frozen old artifacts on identical new final test rows',flush=True)
        ox=pd.DataFrame({'amount':tx.amount,'hour':tx.step%24,'type':tx.type,'oldbalanceOrg':tx.oldbalanceOrg})
        balance=tx.oldbalanceOrg.to_numpy();amount=tx.amount.to_numpy()
        ox['spent_fraction']=np.where(balance>0,amount/np.maximum(balance,1),0).clip(0,1000)
        ox['log_amount']=np.log1p(amount);ox['log_prebalance']=np.log1p(balance)
        comparisons=[]
        for filename in ['fraud.joblib','fraud_balance_research.joblib']:
            old=joblib.load(Path(old_dir)/filename)
            raw=old['pipeline'].predict_proba(ox[old['columns']])[:,1]
            p=calibrate(raw,'sigmoid' if old['calibrator'] is not None else 'raw',old['calibrator'])
            comparisons.append({'artifact':filename,'trained_on':'Original limited development subset; frozen old calibration and thresholds',
                'test_protocol':'Same 1,293,523 final rows as new model; no old artifact retraining or tuning',
                **metrics(y[test],p,old['policy']['high'])})
        (report/'old_models_common_test.json').write_text(json.dumps(comparisons,indent=2))
    print('Genuine SHAP explanation sample',flush=True)
    service=ScreeningService();examples=pd.read_csv(report/'labelled_test_examples.csv')
    rows=examples.head(6).to_dict('records')+examples.tail(6).to_dict('records')
    explanations=service.predict(rows,explain=True)
    (report/'shap_examples.json').write_text(json.dumps(explanations,indent=2))
    available=[r['explanation'] for r in explanations if r['explanation'] and r['explanation']['status']=='available']
    importance=pd.DataFrame([dict((f['feature'],abs(f['contribution'])) for f in e['features']) for e in available]).mean().sort_values()
    importance.to_csv(report/'shap_importance.csv',header=['mean_absolute_probability_contribution'])
    fig,ax=plt.subplots(figsize=(9,8));importance.tail(16).plot.barh(ax=ax,color='#365fd5')
    ax.set_title('Calibrated SHAP | 6 fraud + 6 legitimate examples\nDescriptive sample, not population importance')
    fig.tight_layout();fig.savefig(report/'shap_global.png',dpi=150);plt.close(fig)
    (report/'shap_validation.json').write_text(json.dumps({'explained_rows':len(available),
        'max_additivity_error':max(e['additivity_error'] for e in available),'background_rows':24},indent=2))
    print('DIAGNOSTICS COMPLETE',result['test'],coldmetrics,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--old-model-dir')
    a=p.parse_args();main(a.data,a.old_model_dir)
