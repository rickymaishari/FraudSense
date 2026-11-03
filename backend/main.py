import io,json,time,logging
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI,UploadFile,File,HTTPException,Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse,FileResponse
from pydantic import ValidationError
from backend.schemas import Transaction,PredictRequest,Assessment
from fraudsense.service import ScreeningService,ROOT

state={'service':None,'error':None,'predictions':0,'started':time.time()}

@asynccontextmanager
async def lifespan(app):
    try:state['service']=ScreeningService()
    except Exception as exc:
        state['error']=str(exc);logging.exception('Model loading failed')
    yield

app=FastAPI(title='FraudSense 2 API',version='2.0',lifespan=lifespan)

@app.exception_handler(RequestValidationError)
async def validation_error(request:Request,exc):
    return JSONResponse(status_code=422,content={'final_action':'Cannot assess / invalid input',
       'fraud_probability':None,'model_alert':'unavailable','errors':[
          {'location':list(e['loc']),'message':e['msg']} for e in exc.errors()]})

def service():
    if state['service'] is None:raise HTTPException(503,'Model unavailable; inspect local logs and artifacts')
    return state['service']

@app.get('/health')
def health():
    ready=state['service'] is not None
    return JSONResponse(status_code=200 if ready else 503,content={'status':'ready' if ready else 'unavailable',
       'model_version':state['service'].bundle['version'] if ready else None,
       'predictions':state['predictions'],'uptime_seconds':time.time()-state['started']})

@app.get('/model/info')
def info():
    b=service().bundle
    return {'model_version':b['version'],'columns':b['columns'],'policy':b['policy'],'coverage':b['coverage'],
            'calibration':b['calibration_kind'],'history':'Explicit earlier-hour history; batch advances by complete hours',
            'scope':'Synthetic PaySim research; not a production payment authorization system'}

@app.get('/model/metrics')
def model_metrics():
    service();p=ROOT/'reports/new_experiments'
    result={name:json.loads((p/file).read_text()) for name,file in
       [('results','results.json'),('comparison','model_comparison.json'),('common_test','common_test_comparison.json'),
        ('dataset','dataset.json'),('by_type','by_type.json'),('by_time','by_time.json'),('budgets','review_budgets.json')]}
    result['diagnostics']={name:json.loads((p/file).read_text()) for name,file in
        [('cold_start','cold_start_test.json'),('matched_rf_ablation','matched_rf_ablation.json'),
         ('old_models_common_test','old_models_common_test.json'),('operational_actions','operational_test.json')] if (p/file).exists()}
    return result

@app.get('/model/plots/{name}')
def plot(name:str):
    if name not in ['test_performance.png','calibration_test.png','calibration_development.png','shap_global.png']:
        raise HTTPException(404,'Unknown plot')
    path=ROOT/'reports/new_experiments'/name
    if not path.exists():raise HTTPException(404,'Plot unavailable')
    return FileResponse(path)

def assess(payload,explain=False):
    try:
        result=service().predict([payload.transaction.model_dump()],[t.model_dump() for t in payload.history],explain)[0]
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    state['predictions']+=1
    return result

@app.post('/predict',response_model=Assessment)
def predict(payload:PredictRequest):return assess(payload)

@app.post('/explain',response_model=Assessment)
def explain(payload:PredictRequest):return assess(payload,True)

@app.post('/predict/batch')
def batch(file:UploadFile=File(...)):
    raw=file.file.read(5_000_001)
    if len(raw)>5_000_000:raise HTTPException(413,'CSV maximum size is 5 MB')
    try:d=pd.read_csv(io.BytesIO(raw),dtype={'nameOrig':str,'nameDest':str})
    except Exception as exc:raise HTTPException(422,'Malformed or empty CSV') from exc
    if d.empty or len(d)>5000:raise HTTPException(422,'Upload 1 to 5,000 rows per batch')
    required={'step','type','amount','nameOrig','nameDest'}
    if not required.issubset(d.columns):raise HTTPException(422,'Missing columns: '+', '.join(sorted(required-set(d.columns))))
    rows=[];indices=[];output=[]
    for i,row in enumerate(d.to_dict('records')):
        if pd.isna(row.get('oldbalanceOrg')):row['oldbalanceOrg']=None
        try:
            item=Transaction.model_validate(row);rows.append(item.model_dump());indices.append(i)
        except ValidationError as exc:
            output.append({'row':i+1,'status':'error','final_action':'Cannot assess / invalid input',
                'fraud_probability':None,'errors':[e['msg'] for e in exc.errors()]})
    if rows:
        predictions=service().predict(rows)
        output.extend({'row':i+1,'status':'ok',**p} for i,p in zip(indices,predictions))
        state['predictions']+=len(rows)
    return {'rows':sorted(output,key=lambda r:r['row']),'valid_rows':len(rows),'invalid_rows':len(d)-len(rows),
            'history_policy':'Valid earlier hours within this CSV supply history; invalid rows are excluded'}
