"""Full-data chronological experiment; final test labels consulted only after freeze."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,time,argparse,platform,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import PrecisionRecallDisplay,RocCurveDisplay,ConfusionMatrixDisplay,average_precision_score
from sklearn.calibration import CalibrationDisplay
from xgboost import XGBClassifier
from fraudsense.data import load_dataset,inspect_dataset,partitions,sha256
from fraudsense.features import build_features,BALANCE
from fraudsense.evaluation import metrics,choose_thresholds,budgets
from fraudsense.model import calibrate,probabilities

REPORT=ROOT/'reports/new_experiments';ART=ROOT/'artifacts'
def save(name,obj):
    encoded=json.dumps(obj,default=lambda v:v.item() if hasattr(v,'item') else str(v))
    # XGBoost's missing-value hyperparameter is NaN; encode it as JSON null.
    clean=json.loads(encoded,parse_constant=lambda _:None)
    (REPORT/name).write_text(json.dumps(clean,indent=2,allow_nan=False))

def main(path):
    start=time.perf_counter();print('Loading and verifying entire dataset',flush=True)
    d=load_dataset(path);quality=inspect_dataset(d,path);save('dataset.json',quality)
    if len(d)<6_000_000:
        raise ValueError('Full-data run requires complete PaySim, not a development subset')
    masks,split=partitions(d);save('split.json',split)
    print(json.dumps(split),flush=True)
    print('Building shared strictly earlier-hour features',flush=True)
    cache=ROOT/'data/features.parquet'
    if cache.exists() and (ROOT/'data/features.sha256').exists() and (ROOT/'data/features.sha256').read_text()==quality['csv_sha256']+sha256(ROOT/'fraudsense/features.py'):
        x=pd.read_parquet(cache)
    else:
        x=build_features(d);x.to_parquet(cache,index=False)
        (ROOT/'data/features.sha256').write_text(quality['csv_sha256']+sha256(ROOT/'fraudsense/features.py'))
    y=d.isFraud.to_numpy();fit=masks['fit'];val=masks['selection'];test=masks['test']
    allcols=list(x.columns);no_balance=[c for c in allcols if c not in BALANCE]
    no_history=[c for c in allcols if c in BALANCE or c in ['amount','log_amount','hour'] or c.startswith('type_')]
    ratio=float((y[fit]==0).sum()/y[fit].sum())
    def xgb(depth,weight):
        return XGBClassifier(n_estimators=180,max_depth=depth,learning_rate=.07,subsample=.85,
              colsample_bytree=.9,scale_pos_weight=weight,tree_method='hist',n_jobs=4,random_state=42)
    candidates=[('majority',allcols,DummyClassifier(strategy='most_frequent')),
        ('logistic',allcols,make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=250,C=.1,random_state=42))),
        ('random_forest',allcols,RandomForestClassifier(n_estimators=64,max_depth=14,min_samples_leaf=5,
              max_samples=.25,class_weight='balanced_subsample',n_jobs=4,random_state=42)),
        ('xgboost_depth4',allcols,xgb(4,np.sqrt(ratio))),
        ('xgboost_depth6',allcols,xgb(6,np.sqrt(ratio))),
        ('ablation_no_balance',no_balance,xgb(4,np.sqrt(ratio))),
        ('ablation_no_history',no_history,xgb(4,np.sqrt(ratio)))]
    records=[];policies={}
    for name,cols,model in candidates:
        dest=ART/(name+'.joblib');print('TRAIN',name,'rows',int(fit.sum()),flush=True)
        tick=time.perf_counter();model.fit(x.loc[fit,cols],y[fit]);elapsed=time.perf_counter()-tick
        tick=time.perf_counter();p=model.predict_proba(x.loc[val,cols])[:,1];latency=(time.perf_counter()-tick)*1000/val.sum()
        policy=choose_thresholds(y[val],p);policies[name]=policy
        rec={'name':name,'columns':cols,'train_seconds':elapsed,'inference_ms_per_row':latency,
             'params':model.get_params(),'validation':metrics(y[val],p,policy['high'])}
        records.append(rec);joblib.dump({'pipeline':model,'columns':cols},dest,compress=3)
        save('model_comparison.json',records);print(name,rec['validation'],flush=True)
        del model,p;gc.collect()
    contenders=[r for r in records if not r['name'].startswith('ablation')]
    best=max(contenders,key=lambda r:(r['validation']['average_precision'],r['validation']['f1']))
    bundle=joblib.load(ART/(best['name']+'.joblib'))
    cfit=masks['calibration_fit'];ccheck=masks['calibration_check']
    rawfit=bundle['pipeline'].predict_proba(x.loc[cfit,bundle['columns']])[:,1]
    rawcheck=bundle['pipeline'].predict_proba(x.loc[ccheck,bundle['columns']])[:,1]
    clipped=np.clip(rawfit,1e-7,1-1e-7)
    sigmoid=LogisticRegression(C=1000,random_state=42).fit(np.log(clipped/(1-clipped)).reshape(-1,1),y[cfit])
    isotonic=IsotonicRegression(out_of_bounds='clip').fit(rawfit,y[cfit])
    calibrators={'raw':None,'sigmoid':sigmoid,'isotonic':isotonic}
    calresults={k:metrics(y[ccheck],calibrate(rawcheck,k,c)) for k,c in calibrators.items()}
    eligible=[k for k,m in calresults.items() if m['brier']<=calresults['raw']['brier'] and m['log_loss']<=calresults['raw']['log_loss']]
    chosen=min(eligible,key=lambda k:calresults[k]['log_loss'])
    bundle.update(calibration_kind=chosen,calibrator=calibrators[chosen])
    pval=probabilities(bundle,x.loc[val]);policy=choose_thresholds(y[val],pval)
    # Coverage from fitting data only; per-type min/max AND balance-ratio bounds.
    coverage={}
    for t in d.type.cat.categories:
        m=fit & (d.type==t).to_numpy()
        coverage[str(t)]={c:[float(x.loc[m,c].min()),float(x.loc[m,c].max())] for c in ['amount','opening_balance','amount_balance_ratio']}
    print('Fit and evaluate independent anomaly component',flush=True)
    rng=np.random.default_rng(42);sample=rng.choice(np.flatnonzero(fit),min(100000,int(fit.sum())),replace=False)
    anomaly=IsolationForest(n_estimators=100,max_samples=1024,contamination='auto',n_jobs=4,random_state=42).fit(x.iloc[sample])
    aval=-anomaly.score_samples(x.loc[val]);anomaly_threshold=float(np.quantile(aval,.995))
    additional=(aval>=anomaly_threshold)&(pval<policy['low'])
    anomaly_validation={'average_precision':float(average_precision_score(y[val],aval)),
       'additional_alerts':int(additional.sum()),'additional_frauds':int(y[val][additional].sum()),
       'additional_precision':float(y[val][additional].mean()) if additional.any() else 0.,
       'routing_enabled':False,'reason':'Exploratory anomaly indicator only; no operational routing without external validation'}
    bundle.update(policy=policy,coverage=coverage,anomaly=anomaly,anomaly_threshold=anomaly_threshold,
                  feature_columns=allcols,version='FraudSense-2.0-'+quality['csv_sha256'][:10],
                  background=x.loc[fit,bundle['columns']].sample(24,random_state=42),
                  step_range=[int(d.step.min()),int(d.step.max())])
    # Frozen before final 20% scoring. Never refit after calibration.
    freeze={'champion':best['name'],'calibration':{'chosen':chosen,'independent_check':calresults},
       'policy':policy,'coverage':coverage,'anomaly_validation':anomaly_validation,'split':split,
       'selection_rule':'Highest selection AP among supervised candidates; F1 tie-break; ablations excluded',
       'python':platform.python_version(),'seed':42,'dataset_sha256':quality['csv_sha256']}
    save('frozen_development_decisions.json',freeze)
    joblib.dump(bundle,ART/'champion.joblib',compress=3)
    print('Development frozen. Evaluating untouched final test',flush=True)
    tick=time.perf_counter();p=probabilities(bundle,x.loc[test]);infer=(time.perf_counter()-tick)*1000/test.sum()
    final=metrics(y[test],p,policy['high']);final['inference_ms_per_row']=infer
    atest=-anomaly.score_samples(x.loc[test])
    # Range masks use the exact independent coverage function used in serving.
    from fraudsense.risk import supported_rows
    supported=supported_rows(d.loc[test].reset_index(drop=True),x.loc[test].reset_index(drop=True),bundle)
    testframe=pd.DataFrame({'row_id':np.flatnonzero(test),'step':d.loc[test,'step'].to_numpy(),
        'type':d.loc[test,'type'].astype(str).to_numpy(),'label':y[test],'probability':p,
        'supported':supported,'anomaly_score':atest,'alert':p>=policy['high']})
    testframe.to_parquet(REPORT/'test_predictions.parquet',index=False)
    testframe.head(100).to_csv(REPORT/'test_predictions_preview.csv',index=False)
    comparison=[]
    for record in records:
        saved=joblib.load(ART/(record['name']+'.joblib'))
        pred=saved['pipeline'].predict_proba(x.loc[test,saved['columns']])[:,1]
        comparison.append({'name':record['name'],**metrics(y[test],pred,policies[record['name']]['high'])})
    save('common_test_comparison.json',comparison)
    outgoing=d.loc[test,'type'].isin(['CASH_OUT','TRANSFER','PAYMENT','DEBIT']).to_numpy()
    shortfall=outgoing & (d.loc[test,'amount'].to_numpy()>d.loc[test,'oldbalanceOrg'].to_numpy())
    held=shortfall | ~supported
    action=np.where(held,'validation_hold',np.where(p>=policy['high'],'priority_review',
                    np.where(p>=policy['low'],'additional_verification','standard_checks')))
    save('operational_test.json',{'balance_shortfalls':int(shortfall.sum()),'held_rows':int(held.sum()),
        'held_legitimate_labels':int((held&(y[test]==0)).sum()),'held_fraud_labels':int((held&(y[test]==1)).sum()),
        'action_counts':pd.Series(action).value_counts().to_dict(),
        'note':'Operational holds are not model false positives or fraud verdicts; many PaySim balances are inconsistent'})
    bytype={t:metrics(g.label,g.probability,policy['high']) for t,g in testframe.groupby('type')}
    timegroups=testframe.assign(period=(testframe.step//24)*24).groupby('period')
    bytime={int(t):metrics(g.label,g.probability,policy['high']) for t,g in timegroups}
    save('by_type.json',bytype);save('by_time.json',bytime);save('review_budgets.json',budgets(y[test],p))
    # Saved error examples are traceable to source rows and never used for tuning.
    errors=testframe[testframe.alert != testframe.label.astype(bool)]
    errors.to_csv(REPORT/'false_positives_and_negatives.csv',index=False)
    labelled_examples=pd.concat([d.loc[test & (y==1)].head(12),d.loc[test & (y==0)].head(12)])
    labelled_examples.to_csv(REPORT/'labelled_test_examples.csv',index=False)
    balance_equal=np.isclose(d.amount,d.oldbalanceOrg,rtol=0,atol=.01)&(d.amount>0)
    shortcut={name:{'fraud_equal':int((balance_equal&m&(y==1)).sum()),'frauds':int(y[m].sum()),
                    'legitimate_equal':int((balance_equal&m&(y==0)).sum()),'legitimate':int((m&(y==0)).sum())}
              for name,m in [('fit',fit),('test',test)]}
    results={**freeze,'test':final,'test_supported':metrics(y[test][supported],p[supported],policy['high']) if supported.any() else None,
        'test_coverage':float(supported.mean()),'test_withheld':int((~supported).sum()),
        'test_withheld_frauds':int(y[test][~supported].sum()),'shortcut':shortcut,
        'anomaly_test_average_precision':float(average_precision_score(y[test],atest)),
        'elapsed_seconds':time.perf_counter()-start,'complete_dataset_rows':len(d)}
    save('results.json',results)
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    PrecisionRecallDisplay.from_predictions(y[test],p,ax=axes[0]);RocCurveDisplay.from_predictions(y[test],p,ax=axes[1])
    ConfusionMatrixDisplay.from_predictions(y[test],p>=policy['high'],ax=axes[2],colorbar=False,values_format='d')
    axes[0].legend([f'Champion AP = {final["average_precision"]:.6f}'])
    axes[1].legend([f'Champion ROC-AUC = {final["roc_auc"]:.6f}'])
    fig.tight_layout();fig.savefig(REPORT/'test_performance.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    CalibrationDisplay.from_predictions(y[test],p,n_bins=10,strategy='uniform',ax=ax,name='Final test')
    fig.tight_layout();fig.savefig(REPORT/'calibration_test.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for k,c in calibrators.items():
        CalibrationDisplay.from_predictions(y[ccheck],calibrate(rawcheck,k,c),n_bins=10,ax=ax,name=k)
    fig.tight_layout();fig.savefig(REPORT/'calibration_development.png',dpi=150);plt.close(fig)
    # Quantile reliability bins expose sparse-tail calibration uncertainty.
    table=testframe.copy();table['bin']=pd.qcut(table.probability,10,duplicates='drop')
    table.groupby('bin',observed=True).agg(rows=('label','size'),frauds=('label','sum'),mean_probability=('probability','mean'),observed_rate=('label','mean')).to_csv(REPORT/'reliability_bins.csv')
    print('COMPLETE',json.dumps(results),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',required=True)
    main(parser.parse_args().data)
