# -*- coding: utf-8 -*-
"""
Created on Wed May 24 14:01:50 2017

sample codes for validating data

@author: mxz20
"""
    
import funcs

if __name__ == "__main__":
    
    query = '''select a.primary_fn, 
            b.dbacctno, 
            cast(substr(a.ce_date,3,2) as int)*100+cast(substr(a.ce_date,6,2) as int)-b.month as monthsince,
            b.accttype, 
            cast(b.maxcreditavail as int) as maxcreditavail, 
            cast(case when b.paymentstatus = 'X' then '7' else b.paymentstatus end as int) as paymentstatus
            from cda487_db.sample_ccronly a
            join cda487_db.ccr_tbl_full b on a.primary_fn=b.primary_fn;'''
    df = get_hd(query)
    
    #create dummy coding from categorical fields, e.g. industrycode
    dummy_accttype = pd.get_dummies(df['accttype'], prefix = 'accttype')
    
    ccr = pd.concat([df[df.columns[0:3]], dummy_accttype, df[df.columns[4:]]], axis=1)   