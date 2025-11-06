import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Volatility Regime Classification
    # Calculate daily true range
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(data['high'], data['prev_close']) - np.minimum(data['low'], data['prev_close'])
    
    # Compute 20-day rolling median of true range
    data['tr_rolling_median'] = data['true_range'].rolling(window=20, min_periods=1).median()
    
    # Classify regime
    data['vol_regime'] = np.where(data['true_range'] > data['tr_rolling_median'], 'High Vol', 'Low Vol')
    
    # Price Efficiency Under Different Regimes
    # High Volatility Regime Analysis
    data['vol_adj_price_move'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_confirmation'] = np.sign(data['close'] - data['open']) * np.log(data['volume'] / (data['prev_volume'] + 1e-8))
    
    # Low Volatility Regime Analysis
    data['close_5d_ago'] = data['close'].shift(5)
    # Calculate rolling high-low range over 5 days
    data['high_5d'] = data['high'].rolling(window=5, min_periods=1).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=1).min()
    data['momentum_persistence'] = (data['close'] - data['close_5d_ago']) / (data['high_5d'] - data['low_5d'] + 1e-8)
    
    data['volume_20d_ago'] = data['volume'].shift(20)
    data['volume_breakout'] = data['volume'] / (data['volume_20d_ago'] + 1e-8) - 1
    
    # Market Microstructure Efficiency
    data['price_impact_efficiency'] = (data['close'] - data['open']) / (data['amount'] + 1e-8)
    
    # Volume clustering persistence
    data['volume_autocorr'] = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) == 3 else np.nan, raw=False
    )
    data['volume_clustering_strength'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Microstructure noise
    data['microstructure_noise'] = np.abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'] + 1e-8)
    
    # Regime-Adaptive Factor Synthesis
    # High Volatility Factor
    data['high_vol_factor'] = data['vol_adj_price_move'] * data['volume_confirmation']
    
    # Low Volatility Factor
    data['low_vol_factor'] = data['momentum_persistence'] * data['volume_breakout']
    
    # Microstructure Quality Score
    data['microstructure_quality'] = data['price_impact_efficiency'] / (1 + np.abs(data['microstructure_noise']))
    
    # Final Adaptive Factor
    data['adaptive_factor'] = np.where(
        data['vol_regime'] == 'High Vol',
        data['high_vol_factor'] * data['microstructure_quality'],
        data['low_vol_factor'] * data['microstructure_quality']
    )
    
    # Return the final factor series
    return data['adaptive_factor']
