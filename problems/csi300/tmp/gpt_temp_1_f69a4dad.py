import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Components
    # Intraday Volatility
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    
    # Gap Volatility
    data['gap_vol'] = np.abs(data['open'] / data['close'].shift(1) - 1)
    
    # Trend Volatility
    data['trend_vol'] = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    
    # Volume Distribution Analysis
    # Volume Concentration
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).sum()
    
    # Volume Skewness
    rolling_vol = data['volume'].rolling(window=5)
    data['volume_skewness'] = (data['volume'] - rolling_vol.median()) / (rolling_vol.max() - rolling_vol.min())
    
    # Volume Persistence
    def calc_volume_persistence(series):
        if len(series) < 5:
            return np.nan
        vol_changes = series.diff().iloc[-4:]
        pos_count = (vol_changes > 0).sum()
        neg_count = (vol_changes < 0).sum()
        return pos_count - neg_count
    
    data['volume_persistence'] = data['volume'].rolling(window=5).apply(calc_volume_persistence, raw=False)
    
    # Volatility-Volume Interaction
    # Volatility Efficiency
    data['volatility_efficiency'] = ((data['close'] - data['close'].shift(1)) / 
                                   (data['high'] - data['low'])) * data['volume']
    
    # Volume-Volatility Divergence
    data['volume_vol_divergence'] = ((data['volume'] / data['volume'].shift(1) - 1) - 
                                   (data['intraday_vol'] / data['intraday_vol'].shift(1) - 1))
    
    # Gap Absorption
    data['gap_absorption'] = ((data['close'] - data['open']) / 
                            (data['open'] - data['close'].shift(1))) * data['volume']
    
    # Market State Detection
    # High Volatility State
    data['high_vol_state'] = (data['intraday_vol'] > 
                            data['intraday_vol'].rolling(window=11).median()).astype(float)
    
    # Volume Expansion State
    def percentile_70(x):
        if len(x) < 11:
            return np.nan
        return np.percentile(x, 70)
    
    data['volume_expansion_state'] = (data['volume'] > 
                                    data['volume'].rolling(window=11).apply(percentile_70, raw=True)).astype(float)
    
    # Trend State
    data['trend_state'] = (np.abs(data['close'] / data['close'].shift(4) - 1) > 0.02).astype(float)
    
    # Alpha Construction
    # Volatility-Adjusted Volume
    data['vol_adj_volume'] = (data['volume_concentration'] * 
                            data['volatility_efficiency'] * 
                            data['volume_persistence'])
    
    # Regime-Weighted Signal
    data['regime_weight'] = (data['high_vol_state'] * 1.5 + (1 - data['high_vol_state']) * 0.8) * \
                          (data['volume_expansion_state'] * 1.2 + (1 - data['volume_expansion_state']) * 0.9)
    
    data['regime_weighted_signal'] = data['vol_adj_volume'] * data['regime_weight']
    
    # Final Alpha
    data['alpha'] = (data['regime_weighted_signal'] * 
                   data['gap_absorption'] * 
                   (1 + data['volume_vol_divergence']))
    
    return data['alpha']
