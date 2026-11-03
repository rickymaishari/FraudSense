from typing import Literal
from pydantic import BaseModel,Field,ConfigDict,field_validator


class Transaction(BaseModel):
    model_config=ConfigDict(extra='ignore',allow_inf_nan=False)
    step:int=Field(ge=1,le=1000000)
    type:Literal['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']
    amount:float=Field(ge=0,le=1e15)
    nameOrig:str=Field(min_length=1,max_length=100)
    nameDest:str=Field(min_length=1,max_length=100)
    oldbalanceOrg:float|None=Field(default=None,ge=0,le=1e15)

    @field_validator('nameOrig','nameDest')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():raise ValueError('Identifier cannot be blank')
        return value.strip()


class PredictRequest(BaseModel):
    transaction:Transaction
    history:list[Transaction]=Field(default_factory=list,max_length=100000)


class ValidationResult(BaseModel):
    status:str
    codes:list[str]
    warnings:list[str]


class AnomalyResult(BaseModel):
    status:str
    score:float
    threshold:float
    affects_action:bool


class Assessment(BaseModel):
    schema_version:str
    model_version:str
    fraud_probability:float|None
    probability_status:str
    model_alert:str
    validation:ValidationResult
    anomaly:AnomalyResult
    final_action:str
    reasons:list[str]
    next_step:str
    limitations:list[str]
    explanation:dict|None=None
