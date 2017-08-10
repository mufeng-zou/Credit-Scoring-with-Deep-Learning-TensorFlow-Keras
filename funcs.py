# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:13:10 2017

sampling codes

@author: mxz20
"""

import numpy as np
import pandas as pd
import pyodbc
import re
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils



def get_hd(query, channel = 'impala'):
    if channel.lower() == 'impala':
        dsn = 'DSN=Impala AU Prod'
    elif channel.lower() == 'hive':
        dsn = 'DSN=HIVE AU Prod'
    else:
        dsn = channel;
        
    conn = pyodbc.connect(dsn, autocommit = True, ansi = True)
    cur = conn.cursor()
    cur.execute(query)
    
    colnames = []
    colinfo = cur.description
    for col in colinfo:
        name = re.sub('^.+[\.]','',col[0])
        colnames.append(name)

    rows = cur.fetchall()
        
    cur.close()
    conn.close()
    
    out = pd.DataFrame.from_records(rows)
    out.columns = colnames
    
    return out
    
def exec_hd(query, channel = 'hive'):
    if channel.lower() == 'impala':
        dsn = 'DSN=Impala AU Prod'
    elif channel.lower() == 'hive':
        dsn = 'DSN=HIVE AU Prod'
    else:
        dsn = channel;
    conn = pyodbc.connect(dsn, autocommit = True, ansi = True)
    cur = conn.cursor()
    cur.execute(query)
    cur.close()
    conn.close()
    return
    
def create_dummy(col, prefix):
    x = np.array(col, dtype = 'str')
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(x)
    encoded = encoder.transform(x)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy = np_utils.to_categorical(encoded)
    
    out = pd.DataFrame.from_records(dummy)
    colnames = []
    for name in encoder.classes_:
        colnames.append(prefix+'_'+name)
    out.columns = colnames
    return out
    
if __name__ == "__main__":
    
    query = 'select * from cda487_db.rq_tbl_ccronly;'
    df = get_hd(query)   
    df
    y = create_dummy(df['industrycode'], prefix = 'industrycode')
    
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(ind)
    encoded = encoder.transform(ind)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy = np_utils.to_categorical(encoded)

    np.savez('../data/sample.npz', npa=npa, npa2=npa2)
    
    npas = np.load('../data/testsave.npz')
    
    run_hive('''create table cda487a_db.testmz as
SELECT businessid FROM trainingproject_db.CGR_business_current WHERE archive = 20170301 LIMIT 10;''')

    run_hive('''drop table cda487a_db.testmz''')