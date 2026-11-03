"""Streamlit widgets exercised against a real HTTP backend, not mock predictions."""
import os,subprocess,sys,time
import httpx
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from fraudsense.service import ROOT

@pytest.fixture(scope='module',autouse=True)
def live_backend():
    env=os.environ.get('FRAUDSENSE_API_URL')
    os.environ['FRAUDSENSE_API_URL']='http://127.0.0.1:8017'
    process=subprocess.Popen([sys.executable,'-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8017'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if httpx.get('http://127.0.0.1:8017/health').status_code==200:break
            except httpx.HTTPError:pass
            time.sleep(.5)
        else:raise RuntimeError('Test backend failed to start')
        yield
    finally:
        process.terminate();process.wait(timeout=15)
        if env is None:os.environ.pop('FRAUDSENSE_API_URL',None)
        else:os.environ['FRAUDSENSE_API_URL']=env

def test_six_pages_render_with_live_metrics():
    at=AppTest.from_file(str(ROOT/'frontend/app.py'),default_timeout=60).run()
    assert not at.exception
    for page in ['Analyze Transaction','Batch Analysis','Model Performance','Explainable AI','How It Works','Overview']:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, page
    assert 'Prediction service connected' in at.success[0].value

def test_extreme_form_calls_backend_and_withholds_probability():
    at=AppTest.from_file(str(ROOT/'frontend/app.py'),default_timeout=60).run()
    at.sidebar.radio[0].set_value('Analyze Transaction').run()
    at.selectbox[0].set_value('Extreme-amount regression case').run()
    at.button[0].click().run()
    assert not at.exception
    assert any('Payment on hold' in v.value for v in at.error)
    assert any('N/A' in v.value for v in at.warning)
    assert at.session_state['last_result']['fraud_probability'] is None

def test_normal_form_and_genuine_shap_page():
    at=AppTest.from_file(str(ROOT/'frontend/app.py'),default_timeout=120).run()
    at.sidebar.radio[0].set_value('Analyze Transaction').run()
    at.button[0].click().run()
    assert not at.exception and at.session_state['last_result']['fraud_probability'] is not None
    at.sidebar.radio[0].set_value('Explainable AI').run()
    at.button[0].click().run()
    assert not at.exception and len(at.dataframe)>0

def test_batch_live_response_renders_downloads():
    # AppTest cannot drive its file uploader; upload through the real API then
    # exercise the same result-rendering state. Browser upload tested separately.
    csv='step,type,amount,nameOrig,nameDest,oldbalanceOrg\n17,CASH_OUT,10000000000,A,B,1000\n17,PAYMENT,-1,C,D,100\n'
    data=httpx.post('http://127.0.0.1:8017/predict/batch',files={'file':('batch.csv',csv,'text/csv')}).json()
    at=AppTest.from_file(str(ROOT/'frontend/app.py'),default_timeout=60)
    at.session_state['batch_result']=data
    at.run();at.sidebar.radio[0].set_value('Batch Analysis').run()
    assert not at.exception and len(at.dataframe)>0
    assert data['valid_rows']==1 and data['invalid_rows']==1
