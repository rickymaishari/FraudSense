import numpy as np
import pandas as pd
import pytest
from fraudsense.features import build_features
from fraudsense.data import boundary,partitions,load_dataset

def frame():
    return pd.DataFrame({'step':[1,1,2,25,26],'amount':[10.,30.,50.,70.,90.],
       'type':['CASH_OUT']*5,'nameOrig':['A']*5,'nameDest':['B']*5,'oldbalanceOrg':[100.]*5})

def test_strict_hour_history_and_24h_window():
    x=build_features(frame())
    assert list(x.sender_count)==[0,0,2,3,4]
    assert list(x.sender_count_24h)==[0,0,2,3,2]
    assert list(x.sender_volume_24h)==[0,0,40,90,120]
    assert x.loc[2,'previous_hour_mean']==20
    assert x.loc[2,'sender_max']==30
    assert list(x.recipient_count)==[0,0,2,3,4]
    assert list(x.pair_count)==[0,0,2,3,4]

def test_same_hour_permutation_and_future_invariance():
    d=frame();x=build_features(d)
    shuffled=d.iloc[[1,0,2,3,4]].reset_index(drop=True)
    np.testing.assert_allclose(build_features(shuffled).iloc[[1,0,2,3,4]],x)
    d.loc[4,'amount']=1e9
    np.testing.assert_allclose(build_features(d).iloc[:4],x.iloc[:4])

def test_forbidden_columns_never_used():
    d=frame();x=build_features(d)
    for c in ['isFraud','isFlaggedFraud','newbalanceOrig','newbalanceDest','oldbalanceDest']:d[c]=999999
    pd.testing.assert_frame_equal(build_features(d),x)

def test_history_single_batch_parity():
    d=frame();whole=build_features(d)
    single=build_features(d.iloc[[4]],d.iloc[:4])
    np.testing.assert_allclose(single.iloc[0],whole.iloc[4])
    with pytest.raises(ValueError):build_features(d.iloc[[1]],d.iloc[[0]])

def test_missing_history_balance_and_ids_not_predictors():
    d=frame().iloc[[0]].drop(columns='oldbalanceOrg');x=build_features(d)
    assert x.balance_missing.iloc[0]==1 and x.history_available.iloc[0]==0
    assert 'nameOrig' not in x and 'nameDest' not in x

def test_record_boundary_keeps_complete_hours():
    d=pd.DataFrame({'step':np.repeat(np.arange(1,41),10),'isFraud':np.tile([0]*9+[1],40)})
    masks,info=partitions(d)
    assert info['development_fraction']==.8
    assert sum(m.sum() for m in masks.values())==len(d)
    assert d.loc[masks['selection'],'step'].max()<d.loc[masks['test'],'step'].min()

def test_malformed_dataset_rejected(tmp_path):
    path=tmp_path/'bad.csv';path.write_text('step,amount\n1,10\n')
    with pytest.raises(ValueError):load_dataset(path)
