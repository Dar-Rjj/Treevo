import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the input dataframe to avoid modifying the original
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR (20-day)
    data['atr_20'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    
    # Volume-Regime Adaptive Momentum
    # Price Efficiency Component
    data['efficiency_ratio'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['price_momentum'] = data['close'] / data['close'].shift(10) - 1
    
    # Volume Regime Component
    data['spread'] = data['high'] - data['low']
    data['volume_spread_corr'] = data['spread'].rolling(window=20, min_periods=10).corr(data['volume'])
    
    # Regime-Adaptive Weighting
    high_corr_mask = data['volume_spread_corr'] > 0.3
    low_corr_mask = data['volume_spread_corr'] <= 0.3
    
    data['regime_weighted_signal'] = 0.0
    data.loc[high_corr_mask, 'regime_weighted_signal'] = -data.loc[high_corr_mask, 'price_momentum'] * (1 - data.loc[high_corr_mask, 'efficiency_ratio'])
    data.loc[low_corr_mask, 'regime_weighted_signal'] = data.loc[low_corr_mask, 'price_momentum'] * data.loc[low_corr_mask, 'efficiency_ratio']
    
    data['volume_regime_factor'] = data['regime_weighted_signal'] / data['atr_20']
    
    # Amount-Weighted Volatility Divergence
    # Volatility Persistence
    data['tr_20_ma'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    data['tr_autocorr'] = data['tr_20_ma'].rolling(window=20, min_periods=10).apply(lambda x: x.autocorr(lag=1) if len(x) >= 20 else np.nan)
    
    # Price-Volume Divergence
    data['price_acceleration'] = (data['close'] / data['close'].shift(5) - 1) - (data['close'].shift(5) / data['close'].shift(10) - 1)
    data['volume_momentum'] = data['volume'] / data['volume'].shift(10) - 1
    data['divergence'] = data['price_acceleration'] - data['volume_momentum']
    
    # Amount Confirmation
    data['amount_20_ma'] = data['amount'].rolling(window=20, min_periods=10).mean()
    data['amount_ratio'] = data['amount'] / data['amount_20_ma']
    
    # Factor Integration
    data['base_signal'] = data['tr_autocorr'] * data['divergence']
    data['amount_weighted_factor'] = data['base_signal'] * data['amount_ratio'] / data['true_range']
    
    # Range Breakout Efficiency
    # Breakout Detection
    data['high_20'] = data['high'].rolling(window=20, min_periods=10).max()
    data['low_20'] = data['low'].rolling(window=20, min_periods=10).min()
    
    data['breakout_direction'] = 0.0
    data.loc[data['close'] > data['high_20'], 'breakout_direction'] = 1.0
    data.loc[data['close'] < data['low_20'], 'breakout_direction'] = -1.0
    
    # Efficiency Validation
    data['breakout_efficiency'] = data['efficiency_ratio']
    
    # Volume Confirmation
    data['volume_20_ma'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_20_ma']
    
    # Factor Construction
    data['breakout_factor'] = data['breakout_direction'] * data['breakout_efficiency'] * data['volume_ratio'] / data['atr_20']
    
    # Combine all factors with equal weighting
    data['combined_factor'] = (
        data['volume_regime_factor'].fillna(0) + 
        data['amount_weighted_factor'].fillna(0) + 
        data['breakout_factor'].fillna(0)
    ) / 3
    
    # Return the final factor series
    return data['combined_factor']
