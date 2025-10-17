import pandas as pd 
import numpy as np 
from ta import momentum, trend, volatility, volume 

def get_labels_peak_trough(df, period=13):
    
    assert isinstance(df, pd.DataFrame)
    df.columns = [x.lower() for x in df.columns]
    
    lag = (period) / 2 if period % 2==0 else (period+1)/2 
    lag = int(lag)
    
    tmp = df[["close"]]
    tmp['min_11'] = tmp.close.rolling(lag).min()
    tmp['fmin_11'] = tmp.close.shift(-lag).rolling(lag).min()
    tmp['max_11'] = tmp.close.rolling(lag).max()
    tmp['fmax_11'] = tmp.close.shift(-lag).rolling(lag).max()
    tmp = tmp.dropna()

    idx1= (tmp.close==tmp.min_11) & (tmp.min_11<=tmp.fmin_11)  # buy=1
    idx2 = (tmp.close==tmp.max_11) & (tmp.max_11>=tmp.fmax_11) # sell=0  hold=2
    idx3 = (~idx1)&(~idx2)
#    print(np.sum(idx1)/len(tmp), np.sum(idx2)/len(tmp))
    idx = (idx1 * 1 + idx2*0 + idx3*2).to_frame()
    idx.columns =['position']
    tmp = pd.merge(df, idx, right_index=True, left_index=True)
    
    return tmp     

def get_ta_momentum(df, period=13):
    # assert error
    assert isinstance(df, pd.DataFrame)
    df.columns = [x.lower() for x in df.columns]
    
    if type(period)==int:
        df['rsi_'+str(period)] = momentum.rsi(df.close, period)
        df['roc_'+str(period)] = momentum.roc(df.close, period)
        df['cmo_'+str(period)] = get_cmo(df, period=period)
        df['wr_'+str(period)]  = momentum.williams_r(df.high, df.low, df.close, period)

    else:
        periods = period

    # if spatial nearness is in among ta rather than among tas, then
    # combine all procedure with one-loop 
    
        for period in periods:
            df['rsi_'+str(period)] = momentum.rsi(df.close, period)
        for period in periods:
            df['roc_'+str(period)] = momentum.roc(df.close, period)
        for period in periods:
            df['cmo_'+str(period)] = get_cmo(df.close, period=period)
        for period in periods:
            df['wr_'+str(period)]  = momentum.williams_r(df.high, df.low, df.close, period)
            
    return df 

def get_ta_volume(df, period=13):
    # assert error
    assert isinstance(df, pd.DataFrame)
    df.columns = [x.lower() for x in df.columns]
    
    if type(period)==int:
        df['cmf_'+str(period)] = volume.chaikin_money_flow(df.high, df.low, df.close, df.volume, period)
        df['mfi_'+str(period)] = volume.money_flow_index(df.high, df.low, df.close, df.volume, period)
        df['fi_'+ str(period)] = volume.force_index(df.close, df.volume, period)
        df['eom_'+str(period)] = volume.ease_of_movement(df.high, df.low, df.volume, period)
        
    else:
        periods = period

    # if spatial nearness is in among ta rather than among tas, then
    # combine all procedure with one-loop 
    
        for period in periods:
            df['cmf_'+str(period)] = volume.chaikin_money_flow(df.high, df.low, df.close, df.volume, period)
        for period in periods:
            df['mfi_'+str(period)] = volume.money_flow_index(df.high, df.low, df.close, df.volume, period)
        for period in periods:
            df['fi_'+ str(period)] = volume.force_index(df.close, df.volume, period)
        for period in periods:
            df['eom_'+str(period)] = volume.ease_of_movement(df.high, df.low, df.volume, period)
            
    return df 


def get_ta_volatility(df, period=13):
    # assert error
    assert isinstance(df, pd.DataFrame)
    df.columns = [x.lower() for x in df.columns]
    
    if type(period)==int:
        df['bbh_'+str(period)] = volatility.bollinger_hband(df.close, period) 
        df['bbl_'+str(period)] = volatility.bollinger_lband(df.close, period) 
        df['bbp_'+str(period)] = volatility.bollinger_pband(df.close, period)
        df['atr_'+str(period)] = volatility.average_true_range(df.high, df.low, df.close, period)

    else:
        periods = period

    # if spatial nearness is in among ta rather than among tas, then
    # combine all procedure with one-loop 
    
        for period in periods:
            df['bbh_'+str(period)] = volatility.bollinger_hband(df.close, period) 
        for period in periods:
            df['bbl_'+str(period)] = volatility.bollinger_lband(df.close, period) 
        for period in periods:
            df['bbp_'+str(period)] = volatility.bollinger_pband(df.close, period)
        for period in periods:
            df['atr_'+str(period)] = volatility.average_true_range(df.high, df.low, df.close, period)
            
    return df 

def get_ta_trend(df, period=13):
    # assert error
    assert isinstance(df, pd.DataFrame)
    df.columns = [x.lower() for x in df.columns]
    
    if type(period)==int:
        df['sma_'+str(period)]  = trend.sma_indicator(df.close, period)
        df['smaO_'+str(period)] = trend.sma_indicator(df.open, period)
        df['ema_'+str(period)]  = trend.ema_indicator(df.close, period)
        df['wma_'+str(period)] = trend.wma_indicator(df.close, period)
        ### below: relative number matter
        df['trix_'+str(period)] = trend.trix(df.close, period)
        df['cci_'+str(period)] = trend.cci(df.high, df.low, df.close, period)
        df['dpo_'+str(period)] = trend.dpo(df.close, period)  # trend occilator
        df['kst_'+str(period)] = trend.kst(df.close, period)  # know sure thing
        df['dmiPos_'+str(period)] = trend.adx_pos(df.high, df.low, df.close, period)  # dmi: directional movement index
        df['dmiNeg_'+str(period)] = trend.adx_neg(df.high, df.low, df.close, period)  # dmi: directional movement index
        df['adx_'+str(period)] = trend.adx(df.high, df.low, df.close, period)  # adx:strength of trend

        # df['icmBase_'+str(period)] = trend.ichimoku_base_line(df.high, df.low, period, int(period*2.89))
        # df['icmLag_'+str(int(period*2.89))] = df.close.shift(int(period*2.89))

        df['icmBase_'+str(period)] = trend.ichimoku_base_line(df.high, df.low, period, int(period))
        df['icmLag_'+str(int(period))] = df.close.shift(int(period))

    else:
        periods = period

    # if spatial nearness is in among ta rather than among tas, then
    # combine all procedure with one-loop 
    
        for period in periods:
            df['sma_'+str(period)]  = trend.sma_indicator(df.close, period)
        for period in periods:
            df['smaO_'+str(period)] = trend.sma_indicator(df.open, period)
        for period in periods:
            df['ema_'+str(period)]  = trend.ema_indicator(df.close, period)
        for period in periods:
            df['wma_'+str(period)] = trend.wma_indicator(df.close, period)
        for period in periods:
            df['trix_'+str(period)] = trend.trix(df.close, period)
        for period in periods:
            df['cci_'+str(period)] = trend.cci(df.high, df.low, df.close, period)
        for period in periods:
            df['dpo_'+str(period)] = trend.dpo(df.close, period)  # trend occilator
        for period in periods:
            df['kst_'+str(period)] = trend.kst(df.close, period)  # know sure thing
        for period in periods:
            df['dmiPos_'+str(period)] = trend.adx_pos(df.high, df.low, df.close, period)  # dmi: directional movement index
        for period in periods:
            df['dmiNeg_'+str(period)] = trend.adx_neg(df.high, df.low, df.close, period)  # dmi: directional movement index
        for period in periods:
            df['adx_'+str(period)] = trend.adx(df.high, df.low, df.close, period)  # adx:strength of trend
        for period in periods:
            df['dpo_'+str(period)] = trend.dpo(df.close, period)  # trend occilator
        # for period in periods:
        #     df['icmBase_'+str(period)] = trend.ichimoku_base_line(df.high, df.low, period, int(period*2.89))
        # for period in periods:  
        #     df['icmLag_'+str(int(period*2.89))] = df.close.shift(int(period*2.89))

        for period in periods:
            df['icmBase_'+str(period)] = trend.ichimoku_base_line(df.high, df.low, period, int(period))
        for period in periods:  
            df['icmLag_'+str(int(period))] = df.close.shift(int(period))

    return df 

def calculate_CMO(series): 
    sum_gains = series[series >= 0].sum()
    sum_losses = np.abs(series[series < 0].sum())
    cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
    
    return np.round(cmo, 3)

def get_cmo(df, period=13):
    #tmp = df.close.diff().rolling(period).apply(calculate_CMO, args=(period,), raw=True)
    
    tmp = df.close.diff().rolling(period).apply(calculate_CMO, raw=True)
    return tmp
    

def get_dmi(df, intervals):
    tmp = pd.DataFrame([])
    for period in intervals:
        tmp['adx_pos'+str(period)]=adx_pos(df.High, df.Low, df.Close, period)
        tmp['adx_neg'+str(period)]=adx_neg(df.High, df.Low, df.Close, period)
        tmp['adx'+str(period)] = adx(df.High, df.Low, df.Close, period)

    return tmp

def get_ibr(df):
    
    ibr = (df.close - df.low) / (df.high - df.low)
    
    return ibr

from collections import Counter

def ta_frequency_lst(lst):
    """
    lst: ta names
    """
    tmp = lst.copy()
    tmp_s = [x.split('_')[0] for x in tmp ]
    tmp_p = [x.split('_')[1] for x in tmp if '_' in x]
    
    return Counter(tmp_s), Counter(tmp_p)

from statsmodels.tsa.stattools import adfuller 

def get_unstable_list(df):
    
    lst = []

    i = 0
    for i in range(0, df.shape[1]):
        try:
            tmp = adfuller(df.iloc[:, i], maxlag=1, regression='c', autolag=None)
            if tmp[0] > tmp[4]['5%']:
                lst.append(df.columns[i])
        except:
             pass

    return lst 

import numpy as np 
def remove_nan_zero(df):
    
    """
    Input:
        df: data frame
    Output:
        df: remove [np.inf, -np.inf, 0]        
    """
    tmp = df.copy()
    tmp = df.replace([np.inf, -np.inf, 0], np.nan).fillna(method='bfill')
    
    return tmp 
    
# momentum = [x for x in dir(ta.momentum)  if x[0].islower()]
# trend = [x for x in dir(ta.trend)  if x[0].islower()]
# volatility = [x for x in dir(ta.volatility)  if x[0].islower()]
# volume = [x for x in dir(ta.volume)  if x[0].islower()]


# ta_dict = dict(momentum=momentum, trend=trend, volatility=volatility, volume=volume)

# tmp1 = [{x:'momentum'} for x in momentum]
# tmp2 = [{x:'trend'} for x in trend]
# tmp3 = [{x:'trend'} for x in volatility]
# tmp4 = [{x:'trend'} for x in volume]

# tmp = tmp1 + tmp2 + tmp3 + tmp4

### categorical list
# momentum_lst = ['rsi', 'roc', 'cmo', 'wr']
# volume_lst = ['cmf', 'mfi', 'fi', 'eom']
# volatility_lst = ['bbh', 'bbl', 'bbp', 'atr']
# trend_lst = ['sma', 'smaO', 'ema', 'wma', 'trix', 'cci', 'dpo', 'kst', "dmiPos", 
# "dmiNeg", 'adx', "icmBase", 'icmLag']

