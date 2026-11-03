import io,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.schemas import Transaction
from fraudsense.service import ScreeningService,ROOT
from fraudsense.evaluation import metrics,choose_thresholds
from fraudsense.model import calibrate
from fraudsense.risk import decision

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:yield c

@pytest.fixture(scope='module')
def service():return ScreeningService()

def row(**changes):
    d={'step':17,'type':'CASH_OUT','amount':1000.,'oldbalanceOrg':10000.,'nameOrig':'C_DEMO_001','nameDest':'M_DEMO_002'}
    d.update(changes);return d

def test_extreme_regression(client):
    response=client.post('/predict',json={'transaction':row(amount=1e10,oldbalanceOrg=1000)})
    assert response.status_code==200
    r=response.json()
    assert r['fraud_probability'] is None and r['model_alert']=='unavailable'
    assert r['final_action']=='Payment on hold — validation required.'
    assert 'BALANCE_SHORTFALL' in r['validation']['codes']
    assert 'OUTSIDE_EVALUATED_RANGE' in r['validation']['codes']

@pytest.mark.parametrize('change',[{'amount':-1},{'amount':'nan'},{'amount':'inf'},
    {'type':'UNKNOWN'},{'oldbalanceOrg':-1},{'nameOrig':' '},{'step':1.5},{'step':0}])
def test_invalid_inputs(client,change):
    r=client.post('/predict',json={'transaction':row(**change)})
    assert r.status_code==422 and r.json()['fraud_probability'] is None

def test_missing_balance_withheld(client):
    r=client.post('/predict',json={'transaction':row(oldbalanceOrg=None)}).json()
    assert r['fraud_probability'] is None and r['validation']['status']=='hold'

def test_normal_and_zero_boundary(client):
    for amount in [0.,1000.]:
        r=client.post('/predict',json={'transaction':row(amount=amount)})
        assert r.status_code==200
        if r.json()['fraud_probability'] is not None:assert 0<=r.json()['fraud_probability']<=1

def test_hold_is_not_fraud_override(service):
    b=service.bundle
    features={'amount':100.,'opening_balance':10.,'amount_balance_ratio':10.,'history_available':0}
    r=decision(row(amount=100,oldbalanceOrg=10),features,0.,0.,b)
    assert r['validation']['status']=='hold'
    assert r['fraud_probability'] in [None,0.]
    assert r['final_action'].startswith('Payment on hold')

def test_legitimate_balance_depletion_not_validation_fraud(client):
    r=client.post('/predict',json={'transaction':row(amount=1000,oldbalanceOrg=1000)}).json()
    assert 'BALANCE_SHORTFALL' not in r['validation']['codes']
    assert 'confirmed fraud' not in r['final_action'].lower()

def test_batch_single_history_consistency(client):
    rows=[row(step=16,amount=200),row(step=17,amount=1000),row(step=17,amount=700)]
    csv=pd.DataFrame(rows).to_csv(index=False)
    batch=client.post('/predict/batch',files={'file':('test.csv',csv,'text/csv')})
    assert batch.status_code==200
    for i in [1,2]:
        single=client.post('/predict',json={'transaction':rows[i],'history':rows[:1]}).json()
        actual=batch.json()['rows'][i]
        assert single['fraud_probability']==actual['fraud_probability']
        assert single['final_action']==actual['final_action']

def test_batch_row_errors_and_extra_labels(client):
    rows=[row(),row(amount=-5),{**row(),'isFraud':1,'newbalanceOrig':999}]
    r=client.post('/predict/batch',files={'file':('test.csv',pd.DataFrame(rows).to_csv(index=False),'text/csv')}).json()
    assert r['valid_rows']==2 and r['invalid_rows']==1
    assert r['rows'][0]['fraud_probability']==r['rows'][2]['fraud_probability']

@pytest.mark.parametrize('contents',['','not,csv\nx,y\n','step,type,amount,nameOrig,nameDest\n'])
def test_malformed_csv(client,contents):
    assert client.post('/predict/batch',files={'file':('bad.csv',contents,'text/csv')}).status_code==422

def test_same_hour_history_rejected(client):
    assert client.post('/predict',json={'transaction':row(),'history':[row()]}).status_code==422

def test_health_info_metrics_and_plots(client):
    for endpoint in ['/health','/model/info','/model/metrics','/model/plots/test_performance.png']:
        assert client.get(endpoint).status_code==200
    assert client.get('/model/plots/nope').status_code==404

def test_calibration_and_thresholds():
    p=np.array([.01,.01,.9,.8]);y=np.array([0,0,1,1])
    assert np.array_equal(calibrate(p,'raw',None),p)
    policy=choose_thresholds(np.zeros(2000),np.zeros(2000))
    assert policy['high']>1 and policy['low']>1

def test_saved_test_predictions_match_report():
    p=ROOT/'reports/new_experiments'
    r=json.loads((p/'results.json').read_text());d=pd.read_parquet(p/'test_predictions.parquet')
    actual=metrics(d.label,d.probability,r['policy']['high'])
    for k in ['precision','recall','f1','average_precision','brier','log_loss','tp','fp','tn','fn']:
        assert actual[k]==pytest.approx(r['test'][k],abs=1e-12)
    assert len(d)==r['split']['partitions']['test']['rows']
    assert int(d.label.sum())==r['split']['partitions']['test']['frauds']

def test_artifact_load_reproducibility(service):
    a=service.predict([row()])[0];b=ScreeningService().predict([row()])[0]
    assert a==b

def test_shap_genuine_additivity_and_ood_withholding(client):
    r=client.post('/explain',json={'transaction':row()})
    assert r.status_code==200
    e=r.json()['explanation']
    assert e['status']=='available' and e['additivity_error']<1e-7
    r=client.post('/explain',json={'transaction':row(amount=1e10)}).json()
    assert r['explanation']['status']=='withheld'

def test_labelled_frauds_are_assessable_without_labels(client):
    examples=pd.read_csv(ROOT/'reports/new_experiments/labelled_test_examples.csv')
    for record in examples[examples.isFraud==1].head(3).to_dict('records'):
        r=client.post('/predict',json={'transaction':record})
        assert r.status_code==200 and r.json()['model_version']
        # Fraud recall is measured across the test set, not asserted perfect per row.

def test_invalid_model_artifact_rejected(tmp_path):
    import joblib
    path=tmp_path/'invalid.joblib';joblib.dump({},path)
    with pytest.raises(ValueError):ScreeningService(path)
