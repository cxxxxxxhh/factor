import pandas as pd
import numpy as np
weight1=pd.read_csv('final2_dfr1.csv')
weight2=pd.read_csv('final2_dfr2.csv')
weight3=pd.read_csv('final2_dfr3.csv')
weight4=pd.read_csv('final2_dfr4.csv')
weight5=pd.read_csv('final2_dfr5.csv')
weight6=pd.read_csv('final2_dfr6.csv')
weight7=pd.read_csv('final2_dfr7.csv')
weight8=pd.read_csv('final2_dfr8.csv')
weight=pd.concat([weight1, weight2,weight3,weight4,weight5,weight6,weight7,weight8])
weight=weight.set_index(['trade date','code'])

close=pd.DataFrame()
close1=pd.read_csv("data/close.csv").set_index('date')
open=pd.read_csv("data/open.csv").set_index('date')
ret=(close1-open)/open
ret = ret.transpose()
ret.index.name = 'code'

# 使用 pivot() 方法将 close DataFrame 转换为以 Stock 和 Date 为索引的形式
ret = ret.reset_index().melt(id_vars=['code'], var_name='trade date', value_name='ret')
ret=ret.set_index(['trade date', 'code'])
#ret=ret[:'2021-12-31']


df=pd.merge(ret,weight,on=['trade date','code'])
df['ret_prob']=df['ret']*df['weight']

grouped = df['ret_prob'].groupby('trade date')

group=pd.DataFrame()
group['model']=grouped.sum()

benchmark=pd.read_csv("data/benchmark.csv").set_index('date')
benchmark=benchmark['close'].pct_change()

group['benchmark']=benchmark

group['model_rate']=1+group['model']-0.001
group['bench_rate']=1+group['benchmark']
group['nav_model']=group['model_rate'].cumprod()
group['nav_bench']=group['bench_rate'].cumprod()

annual_returns = (group.iloc[(group.shape[0]-1),4])**(244/group.shape[0])-1

# 计算组合标准差
portfolio_std = np.std(group['model_rate']-1)

# 计算Sharpe比率
sharpe_ratio = (annual_returns) / portfolio_std

# 计算最大回撤
cum_returns = (group['model_rate']).cumprod()
max_drawdown = np.max(np.maximum.accumulate(cum_returns) - cum_returns) / np.max(cum_returns)

# 打印结果
print('年化收益率:', annual_returns)
print('Sharpe比率:', sharpe_ratio)
print('最大回撤:', max_drawdown)

###加入择时策略


import pandas as pd
import os
import datetime
import numpy as np 
import statsmodels.formula.api as sml
import matplotlib.pyplot as plt
import tushare as ts
import scipy.stats as scs
import matplotlib.mlab as mlab
date=pd.read_csv("data/date.csv").set_index('date')
def getdata(HS300,N,M):
    HS300['beta'] = 0
    HS300['R2'] = 0
    for  i in range(18,len(HS300)-2):
        date2=HS300.index[i]
        end_index=date.index.get_loc(date2)
        #end_index
        df_ne=HS300.iloc[end_index -18:end_index ,:]
        #print(df_ne)
        model = sml.ols(formula='high~low',data = df_ne)
        result=model.fit()
        HS300.iloc[i+1,4] = result.params[1]#斜率
        HS300.iloc[i+1,5] = result.rsquared
    HS300['ret'] = HS300.close.pct_change(1)
    HS300.iloc[0,6]=0
    HS300['beta_norm'] = (HS300['beta'] - HS300.beta.rolling(M).mean().shift(1))/HS300.beta.rolling(M).std().shift(1)
    for i in range(18,M):
        HS300.iloc[i,7] = (HS300.iloc[i,4] - HS300.iloc[:i-1,4].mean())/HS300.iloc[:i-1,4].std() 
    HS300['RSRS_R2'] = HS300.beta_norm*HS300.R2
    HS300 = HS300.fillna(0)
    HS300['beta_right'] = HS300.RSRS_R2*HS300.beta
    return HS300

def RSRS1(HS300,S1 = 1.0,S2 = 0.8):

    data  = HS300.copy()
    data['flag'] = 0 # 买卖标记
    data['position'] = 0 # 持仓标记
    position = 0 # 是否持仓，持仓：1，不持仓：0
    for i in range(1,data.shape[0]-1):
        
            # 开仓
        if data.iloc[i,4]>S1 and position ==0:
            data.iloc[i,10] = 1
            data.iloc[i+1,11] = 1
            position = 1
        # 平仓
        elif data.iloc[i,4]<S2 and position ==1: 
            data.iloc[i,10] = -1
            data.iloc[i+1,11] = 0     
            position = 0

        # 保持
        else:
            data.iloc[i+1,11] = data.iloc[i,11]     
        
    data['nav'] = (1+data.close.pct_change(1).fillna(0)*data.position).cumprod() 

    return(data)

def RSRS2(HS300,S = 0.7):
    data = HS300.copy()
    data['flag'] = 0 # 买卖标记
    data['position'] = 0 # 持仓标记
    position = 0 # 是否持仓，持仓：1，不持仓：0
    for i in range(1,data.shape[0]-1):
        
        # 开仓
        if data.iloc[i,7]>S and position ==0:
            data.iloc[i,10] = 1
            data.iloc[i+1,11] = 1
            position = 1
        # 平仓
        elif data.iloc[i,7]<-S and position ==1: 
            data.iloc[i,10] = -1
            data.iloc[i+1,11] = 0     
            position = 0
        
        # 保持
        else:
            data.iloc[i+1,11] = data.iloc[i,11]     
        
    data['nav'] = (1+data.close.pct_change(1).fillna(0)*data.position).cumprod() 
           
    return(data)

def RSRS3(HS300,S = 0.7):
    data = HS300.copy()
    data['flag'] = 0 # 买卖标记
    data['position'] = 0 # 持仓标记
    position = 0 # 是否持仓，持仓：1，不持仓：0
    for i in range(1,data.shape[0]-1):
        
        # 开仓
        if data.iloc[i,8]>S and position ==0:
            data.iloc[i,10] = 1
            data.iloc[i+1,11] = 1
            position = 1
        # 平仓
        elif data.iloc[i,8]<-S and position ==1: 
            data.iloc[i,10] = -1
            data.iloc[i+1,11] = 0     
            position = 0
        
        # 保持
        else:
            data.iloc[i+1,11] = data.iloc[i,11]     
        
    data['nav'] = (1+data.close.pct_change(1).fillna(0)*data.position).cumprod() 
           
    return(data)



def RSRS4(HS300,S = 0.7):
    data = HS300.copy()
    data['flag'] = 0 # 买卖标记
    data['position'] = 0 # 持仓标记
    position = 0 # 是否持仓，持仓：1，不持仓：0
    for i in range(1,data.shape[0]-1):
        
        # 开仓
        if data.iloc[i,9]>S and position ==0:
            data.iloc[i,10] = 1
            data.iloc[i+1,11] = 1
            position = 1
        # 平仓
        elif data.iloc[i,8]<-S and position ==1: 
            data.iloc[i,10] = -1
            data.iloc[i+1,11] = 0     
            position = 0
        
        # 保持
        else:
            data.iloc[i+1,11] = data.iloc[i,11]     
        
    data['nav'] = (1+data.close.pct_change(1).fillna(0)*data.position).cumprod() 
           
    return(data)

HS300=pd.read_csv("data/benchmark.csv").set_index('date')
HS300=getdata(HS300,18,600)
result = RSRS4(HS300)

result = result.rename_axis('trade date')
group1=group.copy()

weight_RS=pd.merge(group1,result['position'],on=['trade date'])
weight_RS.loc[weight_RS['position'] == 0, 'RSRS'] = 1
weight_RS.loc[weight_RS['position'] != 0, 'RSRS'] = weight_RS['model_rate']
weight_RS['nav_model_RSRS']=weight_RS['RSRS'].cumprod()

import matplotlib.pyplot as plt


plt.figure(figsize=(15, 10))
# 绘制model曲线
plt.plot(weight_RS['nav_model'],label='model')
plt.plot(weight_RS['nav_model_RSRS'], label='model_RSRS')

# 绘制bench曲线
plt.plot(weight_RS['nav_bench'], label='benchmark')

plt.xticks(range(0, len(weight_RS), 244), weight_RS.index[::244])

#plt.xticks(np.arange(min(x), max(x)+1, 1.0))

# 添加图例
plt.legend()

# 显示图形
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

plt.plot(weight_RS['model_rate']-1, label='model')

# 绘制model曲线
plt.plot(weight_RS['RSRS']-1, label='model_RS')

# 绘制bench曲线
plt.plot(weight_RS['benchmark'], label='benchmark')

plt.xticks(range(0, len(weight_RS), 244), weight_RS.index[::244])

# 添加图例
plt.legend()

# 显示图形
plt.show()

annual_returns_RS = (weight_RS.iloc[(weight_RS.shape[0]-1),8])**(244/weight_RS.shape[0])-1

# 计算组合标准差
portfolio_std_RS = np.std(weight_RS['RSRS']-1)

# 计算Sharpe比率
sharpe_ratio_RS = (annual_returns_RS) / portfolio_std_RS

# 计算最大回撤
cum_returns_RS = (weight_RS['RSRS']).cumprod()
max_drawdown_RS = np.max(np.maximum.accumulate(cum_returns_RS) - cum_returns_RS) / np.max(cum_returns_RS)

# 打印结果
print('年化收益率:', annual_returns_RS)
print('Sharpe比率:', sharpe_ratio_RS)
print('最大回撤:', max_drawdown_RS)

annual_returns_bench = (weight_RS.iloc[(weight_RS.shape[0]-1),5])**(244/weight_RS.shape[0])-1

# 计算组合标准差
portfolio_std_bench = np.std(weight_RS['bench_rate']-1)

# 计算Sharpe比率
sharpe_ratio_bench = (annual_returns_bench) / portfolio_std_bench

# 计算最大回撤
cum_returns_bench = (weight_RS['bench_rate']).cumprod()
max_drawdown_bench = np.max(np.maximum.accumulate(cum_returns_bench) - cum_returns_bench) / np.max(cum_returns_bench)

# 打印结果
print('年化收益率:', annual_returns_bench)
print('Sharpe比率:', sharpe_ratio_bench)
print('最大回撤:', max_drawdown_bench)

import matplotlib.pyplot as plt

roc_final=pd.read_csv("final2_roc1.csv")

# 绘制model曲线
plt.scatter(roc_final.index,roc_final['date2'], label='auc')



# 绘制bench曲线
# plt.plot(weight_RS['benchmark'], label='benchmark')

# 添加图例
plt.legend()

# 显示图形
plt.show()

weight_RS['cum_returns'] = weight_RS['model'].cumsum()

# 计算每一天之前的最大累计收益率
weight_RS['max_cum_returns'] = weight_RS['cum_returns'].cummax()

# 计算每一天的回撤和最大回撤，并将结果存储到新的列中
weight_RS['drawdowns'] = weight_RS.apply(lambda x: x['cum_returns'] - x['max_cum_returns'], axis=1)
weight_RS['max_drawdowns'] = weight_RS['drawdowns'].cummin()



weight_RS['cum_returns_RS'] =(weight_RS['RSRS']-1).cumsum()

# 计算每一天之前的最大累计收益率
weight_RS['max_cum_returns_RS'] = weight_RS['cum_returns_RS'].cummax()

# 计算每一天的回撤和最大回撤，并将结果存储到新的列中
weight_RS['drawdowns_RS'] = weight_RS.apply(lambda x: x['cum_returns_RS'] - x['max_cum_returns_RS'], axis=1)
weight_RS['max_drawdowns_RS'] = weight_RS['drawdowns_RS'].cummin()


weight_RS['cum_returns_bench'] = weight_RS['benchmark'].cumsum()

# 计算每一天之前的最大累计收益率
weight_RS['max_cum_returns_bench'] = weight_RS['cum_returns_bench'].cummax()

# 计算每一天的回撤和最大回撤，并将结果存储到新的列中
weight_RS['drawdowns_bench'] = weight_RS.apply(lambda x: x['cum_returns_bench'] - x['max_cum_returns_bench'], axis=1)
weight_RS['max_drawdowns_bench'] = weight_RS['drawdowns_bench'].cummin()

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

plt.plot(weight_RS['max_drawdowns'], label='model')

# 绘制model曲线
plt.plot(weight_RS['max_drawdowns_RS']-1, label='model_RS')

# 绘制bench曲线
plt.plot(weight_RS['max_drawdowns_bench'], label='benchmark')

plt.xticks(range(0, len(weight_RS), 244), weight_RS.index[::244])



# 添加图例
plt.legend()

# 显示图形
plt.show()

import pandas as pd
import numpy as np
factor_sample=pd.read_csv('unsuccess/df_test.csv').set_index('code').iloc[0:2044].drop(['close1','close2','signal2','signal3'], axis=1)
factor_sample=factor_sample.dropna(axis=1)
future_return=factor_sample['return']
factor=factor_sample.drop(['return'], axis=1)
#mask = np.isinf(factor_sample)
cols=['002', '011','013','014','015','018','019','020','022',
                         '023','024','029','031','034','037','038','040','042','043','046','047',
                         '049','052','053','058','059','060','065','066','070','071','076',
                         '078','080','081','084','085','089','093','094','095','097','098',
                         '100','102','103','104','106','107','112','117','118','120','126','129',
                         '132','133','139','142','144','145','150','151','153','155','158',
                         '160','161','163','167','168','169','171','172','173','174','175','177','178','180','185','186',
                         '187','188','189','191']

factor_pair=factor_sample[cols]

from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# 计算每个因子与未来收益率之间的相关系数
ic_values = []
for col in factor.columns:
    corr, _ = pearsonr(factor[col], future_return)
    ic_values.append(corr)

#ic_values

# 将IC值乘以每个因子的平均值
# mean_values = factor_sample.mean()
# ic_values = np.array(ic_values) * mean_values.values

# 绘制IC值图
# sns.barplot(x=factor.columns, y=ic_values)

xticks_pos = np.arange(0, factor.shape[1], 20)
#xticks_labels = ['Factor 1', 'Factor 3', 'Factor 5', 'Factor 7', 'Factor 9']
sns.barplot(x=factor.columns, y=ic_values)

plt.xticks(xticks_pos)

# 显示图像
plt.show()

sns.pairplot(factor_pair,hue ='return')



corr_matrix = factor_sample.corr()

# # 绘制热力图
# sns.heatmap(corr_matrix, cmap='coolwarm')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.figure(figsize=(25, 20))

sns.heatmap(corr_matrix,
            cmap='coolwarm',
            # annot=True,
            # fmt='d'
            ).get_figure().savefig("temp.png",dpi=1000,bbox_inches = 'tight') # fmt显示完全，dpi显示清晰，bbox_inches保存完全

plt.show()

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(factor)
# pca_X_train = pca.fit_transform(factor)
# pca_X_trains=pd.DataFrame(pca_X_train)
# pca_X_trains['symbol']=x_train.index
# pca_X_trains=pca_X_trains.set_index('symbol')
variance_ratio = pca.explained_variance_ratio_

# 计算累计方差
cumulative_variance_ratio = np.cumsum(variance_ratio)

import matplotlib.pyplot as plt
import numpy as np

# 生成一个包含10个元素的随机数据
#data = np.random.randint(0, 10, size=(10,))

# 绘制数据的柱状图
#pca_plot=pd.DataFrame(cumulative_variance_ratio)
plt.bar(np.arange(len(cumulative_variance_ratio)),cumulative_variance_ratio)

# 显示图像
plt.show()