import pandas as pd
from itertools import chain
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import abc
from functools import partial

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt






start_date="2014-01-01"
end_date="2021-12-30"
# stocks = get_index_stocks('000300.XSHG')
t=pd.read_csv('data/date.csv').set_index('date')
def get_date(start_date,end_date):
    t1=t.loc[start_date:end_date]
    return t1
t1=get_date(start_date,end_date)
d1=t1.date2.tolist()


factors=['002', '011','013','014','015','016','018','019','020','022',
                         '023','024','026','029','031','034','037','038','040','042','043','045','046','047',
                         '049','052','053','056','058','059','060','065','066','070','071','074','076',
                         '078','080','081','084','085','086','089','093','094','095','097','098',
                         '100','102','103','104','106','107','112','117','118','120','126','129',
                         '132','133','135','139','142','144','145','150','151','152','153','154','155','158',
                         '160','161','163','167','168','169','171','172','173','174','175','177','178','180','185','186',
                         '187','188','189','191']

def ScaleNormalize(df):
    df_test=df.copy()
    for i in range(1,(df.shape[1])) :
        cmean=df.mean(axis=0)#0是列，1是行 ##标准化
        cvar=df.var(axis=0)
        df_test.iloc[:,i]=(df.iloc[:,i]-cmean[i])/cvar[i]
    return df_test

def filter_extreme_MAD(series,n): #MAD:中位数去极值
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n*new_median
    min_range = median - n*new_median
    return np.clip(series,min_range,max_range)

def MAD(df):
    df_final=df.copy()
    for i in range(1,(df.shape[1])):
        temp=df.iloc[:,i]
        df_final.iloc[:,i] = filter_extreme_MAD(temp,3)
    return df_final


is_st=pd.read_csv('data/stInfo.csv').set_index('date')
def isst(start_date,end_date):
    t1=is_st.loc[start_date:end_date]
    return t1

close=pd.read_csv('data/close.csv').set_index('date')
def ret(start_date,end_date):
    t1=close.pct_change().loc[start_date:end_date].dropna(how='all',axis=0)
    return t1
p=ret(start_date,end_date)


import alphaCalculator as Alpha
symbol=pd.read_csv('data/symbol.csv').set_index('symbol')
new_index = pd.Series(symbol['symbol_name'].tolist(), name="symbol2")
def run_alpha_calculator(alpha_index):
    alpha_method_name = f'alpha_{alpha_index:03}'
    if hasattr(alpha_calculator, alpha_method_name):
        alpha_method = getattr(alpha_calculator, alpha_method_name)
        result = alpha_method()  # 调用相应的计算方法
        #print(f"因子%s的值为：{result}" % alpha_method_name)
    # else:
    #     print("请输入有效的 alpha 因子编号！")
    return result

alpha_calculator = Alpha.GTJA_191()
def get_factor(start_date,end_date,factor):
    alpha_calculator.set_date_and_stock(start_date='2013-03-04', end_date=end_date,stock_list=stocks)
    df=run_alpha_calculator(factor)
    df1=pd.DataFrame(df)
    df1=df1.reindex(stocks)
    df1.columns = [end_date]
    df1=df1.rename_axis(factor).T
    return df1

def assign_value(x):
    if x > 0.005:
        return 1
    else:
        return 0
    

dfr=pd.DataFrame()
dfr1=pd.DataFrame()
roc_auc=pd.read_csv('data/date.csv').set_index('date')
for j,dd in enumerate(d1[:-1]):
    if j>=21:
        dd=d1[j]
        sd=d1[j-21]
        ed=dd
        print(sd,ed)
        df2=isst(start_date=sd, end_date=ed)
        # print(df2)
        stocks =list(df2[df2==False].dropna(axis=1).columns)
        # print(stocks)
        p=ret(d1[j-20],dd)[stocks]
        #print(p)
        b=list(chain.from_iterable(p.head(20).values.tolist()))#前20天作为train data
        b1=list(chain.from_iterable(p.tail(1).values.tolist()))#最后一天作为test
        dftrain=pd.DataFrame({'ret':b},index=p.columns.tolist()*20)
        dftest=pd.DataFrame({'ret':b1},index=p.columns)
        #print(b)
        for factor in factors:
            dftemp=pd.DataFrame()
            for date_ in d1[j-20:j]:
                factor_data0 = get_factor(start_date,date_,factor)
                #print(factor_data0)
                dftemp=pd.concat([dftemp,factor_data0])#因子值从9，10，11，12
                a=list(chain.from_iterable(dftemp.head(20).values.tolist()))#取前三天，也就是train的所有因子值
                print(factor,date_)
            #print(dftemp)
            a1=list(chain.from_iterable(dftemp.tail(1).values.tolist()))#test 因子
            dftrain.loc[:,factor]=a
            dftest.loc[:,factor]=a1

        df_train=MAD(dftrain)
        df_train=ScaleNormalize(dftrain)
        df_test=MAD(dftest)
        df_test=ScaleNormalize(dftest)
        df_train.dropna(how='all',axis=1,inplace=True)
        df_test.dropna(how='all',axis=1,inplace=True)
        df_train.dropna(how='any',axis=0,inplace=True)
        df_test.dropna(how='any',axis=0,inplace=True)
        common_columns = np.intersect1d(df_train.columns, df_test.columns)
        #print(common_columns)
        df_train=df_train[common_columns]
        df_test=df_test[common_columns]
        # print(df_train)
        # print(df_test)
        x_train=df_train.iloc[:,:-1]
        y_train=df_train.iloc[:,-1]
        x_test=df_test.iloc[:,:-1]
        y_test=df_test.iloc[:,-1]
        y_train_label=y_train.apply(assign_value)
        y_test_label=y_test.apply(assign_value)
        forest= xgb.XGBClassifier(max_depth=2,learning_rate=0.1,n_estimators=500,
                                    min_child_weight=5,colsample_bytree=0.7,reg_lambda=0.4,
                                    scale_pos_weight=0.8,subsample=0.8)
        forest.fit(x_train,y_train_label)
        print(forest.score(x_train, y_train_label))
        print(forest.score(x_test, y_test_label))

        factor_weight = pd.DataFrame({'features':list(x_train.columns),'importance':forest.feature_importances_}).sort_values(by='importance', ascending = False)
        #这里根据重要程度降序排列，一遍遍找到重要性最高的特征
        factor_weight['sum']=factor_weight['importance'].cumsum()
        indice_all=np.where(factor_weight['sum']>0.9)
        indice=np.min(indice_all)
        factor_important=factor_weight['features'].iloc[:indice]
        factors_n = [col for col in factor_important if col in x_train.columns]
        dfr1=pd.concat([dfr1,pd.DataFrame({"trade_date":[dd]*len(factors_n),
                                        "factor":factors_n,
                                        "importance":factor_weight['importance'][:len(factors_n)]})])
        x_train_xgb=df_train.loc[:,factors_n]
        y_train_xgb=df_train.iloc[:,-1]
        x_test_xgb=df_test.loc[:,factors_n]
        y_test_xgb=df_test.iloc[:,-1]
        # y_train_label=pd.cut(y_train,[-np.inf,-0.10,0,0.05,0.10,np.inf],labels=list("abcde"))
        # y_test_label=pd.cut(y_test,[-np.inf,-0.10,0,0.05,0.10,np.inf],labels=list("abcde"))
        y_train_label_xgb=y_train.apply(assign_value)
        y_test_label_xgb=y_test.apply(assign_value)
        model_XGB=xgb.XGBClassifier(max_depth=2,learning_rate=0.1,n_estimators=500,
                                    min_child_weight=5,colsample_bytree=0.7,reg_lambda=0.4,
                                    scale_pos_weight=0.8,subsample=0.8)
        model_XGB.fit(x_train_xgb,y_train_label_xgb)
        train_accuracy = model_XGB.score(x_train_xgb, y_train_label_xgb)
        print('Training accuracy:', train_accuracy)
        test_accuracy = model_XGB.score(x_test_xgb, y_test_label_xgb)
        print('Testing accuracy:', test_accuracy)
        y_prob = model_XGB.predict_proba(x_test_xgb)
        #y_prob
        # This will give you positive class prediction probabilities
        # y pred = np.where(y_prob > 0.5, 1, @) 
        # This will threshold the probabilities to give class predictions# model XGB.score(x test, y_ pred)

        #roc=metrics.roc auc score(y test,y pred)# auc roc
        fpr, tpr, thersholds = roc_curve(y_test_label, y_prob[:,1])
        roc_auc.loc[ed] = auc(fpr, tpr)
        print('auc:',auc(fpr, tpr) )
        pool=pd.Series(y_prob[:,-1],index=df_test.index).sort_values().tail(20)
        weight=pool/pool.sum()
        dfr=pd.concat([dfr ,pd.DataFrame({"trade date":[ed]*len(pool),
                                        "ts code":pool.index.tolist(),
                                        "prob":pool.values,
                                        'weight':weight})])

# dfr.to_csv('final2_dfr9.csv')
# dfr1.to_csv('final2_imp9.csv')


# roc_auc.to_csv('final2_roc9.csv')