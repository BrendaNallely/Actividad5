"""Limpieza determinista de Bank Marketing; no aprende estadísticas globales."""
import numpy as np
import pandas as pd

def limpiar_datos(datos):
    d=datos.copy()
    d.columns=d.columns.str.strip()
    for c in d.select_dtypes(include=['object','string']).columns:
        d[c]=d[c].str.strip().str.lower().replace('',np.nan)
    numericas=['age','duration','campaign','pdays','previous','emp.var.rate',
               'cons.price.idx','cons.conf.idx','euribor3m','nr.employed']
    for c in numericas:
        d[c]=pd.to_numeric(d[c],errors='raise')
    if d.y.isna().any() or not d.y.isin(['yes','no']).all():
        raise ValueError('Objetivo ausente o inválido')
    if not d.age.dropna().between(1,120).all():
        raise ValueError('Revisar edad fuera del intervalo de control')
    if (d.campaign.dropna()<1).any():
        raise ValueError('campaign debe ser al menos 1')
    for c in ['duration','pdays','previous']:
        if (d[c].dropna()<0).any(): raise ValueError('Valor negativo en '+c)
    d['y']=d.y.map({'no':0,'yes':1}).astype(int)
    d['sin_contacto_previo']=d.pdays.eq(999).astype(int)
    d['pdays']=d.pdays.mask(d.pdays.eq(999),0)
    return d.drop(columns='duration')
