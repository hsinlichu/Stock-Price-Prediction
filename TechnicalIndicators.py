
# coding: utf-8

# In[1]:


#----------資料讀入----------
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
TW50= pd.read_csv('TW50.csv', index_col=0 ) #讀檔
TW50.dropna(how='any',inplace=True)


# In[2]:


#----------技術指標計算-----------
import talib #技術指標套件
from talib.abstract import *
gap = 34
df = TW50
TW50_C = TW50.copy()
TW50_H = TW50.copy()
TW50_L = TW50.copy()
del TW50_C['open'],TW50_C['high'],TW50_C['low']
del TW50_H['open'],TW50_H['low'],TW50_H['close']
del TW50_L['open'],TW50_L['high'],TW50_L['close']
#close = [float(x) for x in df['close']]

#----------計算DIF-----------
def myDIF(EMA12, EMA26):
    dif = EMA12 - EMA26
    return dif
#----------計算n-m Bias-----------
#n-m Bias = n Bias – m Bias
def myBIAS3_6(BIAS3,BIAS6):
    bias = BIAS3 - BIAS6
    return bias

def TechnicalIndicators(value, df, dff):
    category = [float(x) for x in df[value]]
    
    K,D = talib.STOCH(high = np.array(df.high), 
                    low = np.array(df.low), 
                    close = np.array(df.close),
                    fastk_period=9,
                    slowk_period=3,
                    slowk_matype=0,
                    slowd_period=3,
                    slowd_matype=0)
    dff['K'] = pd.DataFrame(K, index = df.index, columns = ['K'])
    dff['D'] = pd.DataFrame(D, index = df.index, columns = ['D'])
    
    dff['MA6'] = talib.MA(np.array(category), 6, matype=0)   #簡單平均線(SMA)：6天收盤價的平均。
    dff['MA12'] = talib.MA(np.array(category), 12, matype=0)   #簡單平均線(SMA)：12天收盤價的平均。
    dff['RSI']=talib.RSI(np.array(category), timeperiod=6)     #RSI指標
    dff['WILLR']=talib.WILLR(df.high, df.low, df.close, timeperiod=12) #威廉指標:12日
    dff['MOM']=talib.MOM(np.array(category), timeperiod=6)     #MOM運動量指標:6日
    dff['MOM_avg6']=talib.MOM(np.array(dff.MA6), timeperiod=6)     #MOM運動量指標
    dff['EMA12'] = talib.EMA(np.array(category), timeperiod=12)  
    dff['EMA26'] = talib.EMA(np.array(category), timeperiod=26)  
    dff['MACD'],dff['MACDsignal'],dff['MACDhist'] = talib.MACD(np.array(category),
                                fastperiod=12, slowperiod=26, signalperiod=9) #聚散指標
    dff['DIF'] = myDIF(dff['EMA12'].values, dff['EMA26'].values)    #差離值
    dff['BIAS3']=100*(df[value]-df[value].rolling(3).mean())/df[value].rolling(3).mean()   #乖離率:3日
    dff['BIAS6']=100*(df[value]-df[value].rolling(6).mean())/df[value].rolling(6).mean()   #乖離率:6日
    dff['BIAS3_6']=myBIAS3_6(dff['BIAS3'].values, dff['BIAS6'].values)   #乖離率:3-6日
    del dff['EMA12'],dff['EMA26'],dff['MACDsignal'],dff['MACDhist'],dff['BIAS3']
    
    return dff
TW50_C = TechnicalIndicators('close', TW50, TW50_C)
TW50_H = TechnicalIndicators('high', TW50, TW50_H)
TW50_L = TechnicalIndicators('low', TW50, TW50_L)
#print(TW50_C.head(10)) #列出前20筆數據
#print(TW50_H.head(10)) #列出前20筆數據
#print(TW50_L.head(10)) #列出前20筆數據

join = pd.concat([TW50, TW50_C, TW50_H, TW50_L], axis=1)
join.dropna(how='any',inplace=True)
#print(join.head(10))
#df.tail(20) #列出後10筆數據

export_csv = join.to_csv("TW50_processed.csv",  header=True)
