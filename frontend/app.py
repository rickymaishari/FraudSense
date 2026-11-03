"""Presentation only. FastAPI is the sole authority for all assessments."""
import os,json,io
import pandas as pd
import httpx
import streamlit as st

API=os.environ.get('FRAUDSENSE_API_URL','http://127.0.0.1:8000')
st.set_page_config(page_title='FraudSense 2 | Risk intelligence',page_icon='🛡️',layout='wide')
st.markdown('''<style>
[data-testid="stAppViewContainer"]{background:#f5f7fc}
[data-testid="stSidebar"]{background:#e9eef9}
.block-container{max-width:1320px;padding-top:2rem}
h1,h2,h3{color:#182c51;letter-spacing:-.035em}
.hero{padding:2rem 2.3rem;border-radius:22px;background:linear-gradient(120deg,#142747,#304cac 70%,#277c9b);margin-bottom:24px;color:white}
.hero h1{color:white;font-size:2.6rem;margin:.35rem 0}.hero p{color:#dce7ff;max-width:800px}
.eyebrow{letter-spacing:.16em;font-size:.75rem;font-weight:700;color:#a9d9ff}
[data-testid="stMetric"]{padding:18px;background:white;border:1px solid #e0e7f2;border-radius:15px}
.note{background:#fff;border-left:4px solid #5378ed;border-radius:8px;padding:18px;margin:12px 0;color:#3a4c68}
.stButton button{border-radius:10px;font-weight:650}
@media(max-width:700px){.hero{padding:1.2rem}.hero h1{font-size:2rem}}
</style>''',unsafe_allow_html=True)


def request(method,path,**kwargs):
    try:
        response=httpx.request(method,API+path,timeout=120,**kwargs)
        if response.status_code>=400:
            try:detail=response.json()
            except Exception:detail={'detail':response.text[:200]}
            st.error(f'Assessment unavailable ({response.status_code}). {detail}')
            return None
        return response.json()
    except httpx.HTTPError:
        st.error('The prediction service is unavailable. Start the application with scripts\\run_windows.bat.')
        return None


def render_assessment(result):
    st.subheader('Assessment')
    if result['validation']['status']=='hold':st.error(result['final_action'])
    elif result['model_alert']=='priority':st.error(result['final_action'])
    elif result['model_alert']=='verification':st.warning(result['final_action'])
    else:st.info(result['final_action'])
    probability=result['fraud_probability']
    if probability is None:st.warning('Fraud probability: N/A — Outside evaluated model range.')
    else:st.metric('ML fraud probability',f'{probability:.4%}')
    a,b,c=st.columns(3)
    a.metric('Model alert',result['model_alert'].replace('_',' ').title())
    b.metric('Transaction validation',result['validation']['status'].replace('_',' ').title())
    c.metric('Anomaly status',result['anomaly']['status'].replace('_',' ').title())
    st.caption('Anomaly scores measure unusualness, not the chance of fraud. Operational holds are not fraud verdicts.')
    for reason in result['reasons']:st.write('• '+reason)
    st.info('Next step: '+result['next_step'])
    st.caption('Model version: '+result['model_version'])
    with st.expander('Limitations and complete API response'):
        for limitation in result['limitations']:st.write('• '+limitation)
        st.json(result)


st.sidebar.title('🛡️ FraudSense 2')
st.sidebar.caption('TRANSACTION RISK STUDIO')
page=st.sidebar.radio('Workspace',['Overview','Analyze Transaction','Batch Analysis','Model Performance','Explainable AI','How It Works'])
st.sidebar.divider()
st.sidebar.caption('Synthetic data • Human review\n\nA research demonstration, not payment authorization.')

if page=='Overview':
    st.markdown('<div class="hero"><div class="eyebrow">FRAUDSENSE 2.0 / RISK INTELLIGENCE</div><h1>Clarity before every decision.</h1><p>Screen transactions with three independent signals: a supervised fraud estimate, operational validation, and anomaly detection. Understand the evidence before taking action.</p></div>',unsafe_allow_html=True)
    health=request('GET','/health')
    if health:st.success('Prediction service connected • '+health['model_version'])
    a,b,c=st.columns(3)
    a.metric('Assessment layers','3');b.metric('Evaluation protocol','80 / 20');c.metric('History policy','Earlier hours')
    st.subheader('A clear path from transaction to action')
    for title,body in [('01 · Check the transaction','Validate amounts, balances and model coverage. Unsupported estimates are withheld.'),
                       ('02 · Read the signals','See model alerts and unusual patterns separately. A warning never becomes an invented probability.'),
                       ('03 · Choose the next step','Hold for validation, request review, or continue standard checks with the limitations visible.')]:
        st.markdown(f'<div class="note"><strong>{title}</strong><br>{body}</div>',unsafe_allow_html=True)
    st.warning('PaySim is synthetic. Strong benchmark results may depend on simulation patterns, especially opening balances.')

elif page=='Analyze Transaction':
    st.title('Analyze a transaction')
    st.caption('Enter pre-transaction facts. Post-transaction balances and labels never enter the model.')
    preset=st.selectbox('Input preset',['Custom transaction','Extreme-amount regression case'])
    extreme=preset.startswith('Extreme')
    with st.form('transaction'):
        a,b=st.columns(2)
        kind=a.selectbox('Transaction type',['CASH_OUT','TRANSFER','PAYMENT','CASH_IN','DEBIT'])
        amount=a.number_input('Amount',min_value=0.,max_value=1e15,value=1e10 if extreme else 1000.,key='ext_amount' if extreme else 'amount')
        step=a.number_input('Simulation hour',min_value=1,value=17,step=1)
        balance=b.number_input('Sender opening balance',min_value=0.,max_value=1e15,value=1000. if extreme else 10000.,key='ext_balance' if extreme else 'balance')
        missing=b.checkbox('Opening balance unavailable')
        sender=b.text_input('Sender identifier','C_DEMO_001')
        receiver=b.text_input('Receiver identifier','M_DEMO_002')
        history=st.text_area('Optional earlier history (JSON array)',value='[]',help='All supplied history must be from hours earlier than this transaction. No labels are used.')
        submit=st.form_submit_button('Assess transaction',type='primary')
    if submit:
        try:past=json.loads(history)
        except json.JSONDecodeError:st.error('History must be valid JSON.');past=None
        if past is not None:
            payload={'transaction':{'step':step,'type':kind,'amount':amount,'oldbalanceOrg':None if missing else balance,
                                   'nameOrig':sender,'nameDest':receiver},'history':past}
            with st.spinner('Checking transaction and model signals…'):result=request('POST','/predict',json=payload)
            if result:
                st.session_state['last_payload']=payload;st.session_state['last_result']=result
    if 'last_result' in st.session_state:
        st.caption('Last submitted assessment. Changing form fields does not rescore until you submit.')
        render_assessment(st.session_state['last_result'])

elif page=='Batch Analysis':
    st.title('Batch analysis')
    st.write('Upload up to 5,000 transactions in a CSV (5 MB maximum). Valid earlier hours within the batch provide history; records in the same hour cannot see one another.')
    st.code('step,type,amount,nameOrig,nameDest,oldbalanceOrg\n17,CASH_OUT,10000000000,C_DEMO_001,M_DEMO_002,1000',language='csv')
    upload=st.file_uploader('Transactions CSV',type=['csv'])
    if upload and st.button('Analyze batch',type='primary'):
        with st.spinner('Screening valid rows and collecting row-level errors…'):
            result=request('POST','/predict/batch',files={'file':(upload.name,upload.getvalue(),'text/csv')})
        if result:st.session_state['batch_result']=result
    if 'batch_result' in st.session_state:
        result=st.session_state['batch_result'];a,b=st.columns(2)
        a.metric('Valid rows',result['valid_rows']);b.metric('Invalid rows',result['invalid_rows'])
        flat=pd.json_normalize(result['rows']);st.dataframe(flat,use_container_width=True)
        st.bar_chart(flat.final_action.value_counts())
        st.download_button('Download assessment CSV',flat.to_csv(index=False).encode(),'fraudsense_assessments.csv','text/csv')
        st.download_button('Download full JSON',json.dumps(result,indent=2).encode(),'fraudsense_assessments.json','application/json')

elif page=='Model Performance':
    st.title('Measured performance')
    data=request('GET','/model/metrics')
    if data:
        r=data['results'];m=r['test'];st.caption('Frozen champion: '+r['champion']+' · final chronological test, not training performance')
        a,b,c,d=st.columns(4)
        a.metric('Precision',f'{m["precision"]:.2%}');b.metric('Recall',f'{m["recall"]:.2%}');c.metric('Average precision',f'{m["average_precision"]:.4f}');d.metric('False alarms / 1,000',f'{m["false_alarms_per_1000"]:.3f}')
        st.warning('Benchmark metrics include all test model scores. The application withholds estimates outside fitting-data coverage; see supported-only metrics below. Balance patterns are simulator-specific.')
        st.dataframe(pd.DataFrame(r['split']['partitions']).T,use_container_width=True)
        st.subheader('Development-only model selection')
        st.dataframe(pd.DataFrame([{'model':v['name'],'AP':v['validation']['average_precision'],'F1':v['validation']['f1'],'train_seconds':v['train_seconds']} for v in data['comparison']]),use_container_width=True)
        st.subheader('Final test diagnostics')
        for name in ['test_performance.png','calibration_test.png','calibration_development.png']:
            try:
                response=httpx.get(API+'/model/plots/'+name,timeout=30)
                if response.status_code==200:st.image(response.content,use_container_width=True)
            except httpx.HTTPError:st.warning('Plot service unavailable')
        with st.expander('Coverage, ablations, review budgets and subgroup metrics'):
            st.write('Test coverage',r['test_coverage']);st.json(r['test_supported'])
            st.dataframe(pd.DataFrame(data['common_test']),use_container_width=True)
            st.json({'by_type':data['by_type'],'by_time':data['by_time'],'budgets':data['budgets']})
            st.json(data.get('diagnostics',{}))

elif page=='Explainable AI':
    st.title('Understand the model estimate')
    st.write('SHAP attributes the actual calibrated model estimate to its input features relative to training reference transactions. These are associations, not causes.')
    if 'last_payload' not in st.session_state:st.info('Assess a transaction first, then return here to explain it.')
    elif st.button('Explain last assessment',type='primary'):
        with st.spinner('Computing genuine SHAP contributions…'):
            result=request('POST','/explain',json=st.session_state['last_payload'])
        if result:
            render_assessment(result);e=result['explanation']
            if e and e['status']=='available':
                frame=pd.DataFrame(e['features']).sort_values('contribution',key=abs,ascending=False).head(12)
                st.bar_chart(frame.set_index('feature').contribution)
                st.dataframe(frame,use_container_width=True);st.caption(f'Additivity error: {e["additivity_error"]:.3g}')
            else:st.warning(e['reason'] if e else 'Explanation unavailable')

else:
    st.title('How FraudSense works')
    for heading,body in [
        ('The data','PaySim simulates mobile money transactions. It is a research benchmark, not your bank account data. No live banking service is connected.'),
        ('An honest test','The closest whole-hour boundary to 80% of all records separates development from the final 20% test. Earlier development blocks fit models and calibration; a later block selects models and alert thresholds.'),
        ('History without looking ahead','Only strictly earlier hours contribute to account history. The previous amount feature is the mean of the last observed hour because ordering within an hour is unknown. A new account has explicit cold-start features.'),
        ('What probability means','A calibrated estimate is a model prediction, not a guarantee. Calibration is checked on a later development block and the final test. Unsupported amounts, balances, ratios and simulation periods cause the estimate to be withheld.'),
        ('Why a payment can be held','An amount exceeding recorded funds requires a ledger check even when the model gives a low fraud estimate. PaySim can label inconsistent balances as legitimate. A hold is operational validation, not an accusation.'),
        ('Anomaly detection','Isolation Forest identifies unusual combinations of inputs. Its score is not a fraud percentage. It is shown separately and does not automatically change the recommendation.'),
        ('Where the model can fail','Synthetic balance-depletion patterns can inflate performance. Later periods have different fraud prevalence. Bounds cannot catch every new pattern. Real deployments need independent real-world validation, monitoring, security and reliable ledger integration.')]:
        with st.expander(heading,expanded=True):st.write(body)
