import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic components
    df = df.copy()
    
    # True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Amplitude-Adjusted Efficiency
    # True Range Efficiency (10-day net/total true range movement)
    df['net_movement'] = (df['close'] - df['close'].shift(10)).abs()
    df['total_tr_movement'] = df['true_range'].rolling(window=10, min_periods=10).sum()
    df['tr_efficiency'] = df['net_movement'] / df['total_tr_movement']
    
    # Price Efficiency (Close-to-close efficiency ratio)
    df['close_abs_change'] = (df['close'] - df['close'].shift(1)).abs()
    df['close_total_movement'] = df['close_abs_change'].rolling(window=10, min_periods=10).sum()
    df['price_efficiency'] = df['net_movement'] / df['close_total_movement']
    
    # Volume-Amplitude Relationship
    # Volume-Amplitude Correlation (15-day correlation deviation)
    df['amplitude'] = (df['high'] - df['low']) / df['close']
    df['volume_amplitude_corr'] = df['amplitude'].rolling(window=15, min_periods=15).corr(df['volume'])
    df['corr_deviation'] = df['volume_amplitude_corr'] - df['volume_amplitude_corr'].rolling(window=30, min_periods=30).mean()
    
    # Volume Confirmation (Current/20-day average volume ratio)
    df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=20).mean()
    df['volume_confirmation'] = df['volume'] / df['volume_ma_20']
    
    # Volatility-Adjusted Momentum
    # Multi-period Momenta (5-day and 10-day close momentum)
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Momentum Difference (5-day minus 10-day, volatility-adjusted)
    df['momentum_diff'] = df['momentum_5'] - df['momentum_10']
    df['volatility_10'] = df['close'].pct_change().rolling(window=10, min_periods=10).std()
    df['vol_adj_momentum_diff'] = df['momentum_diff'] / (df['volatility_10'] + 1e-8)
    
    # Intraday Strength Persistence
    # Relative Close Position ((Close-Low)/(High-Low))
    df['relative_close'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Consecutive Strength Days tracking
    df['strong_close'] = (df['relative_close'] > 0.6).astype(int)
    df['streak'] = 0
    for i in range(1, len(df)):
        if df['strong_close'].iloc[i] == 1:
            df['streak'].iloc[i] = df['streak'].iloc[i-1] + 1
        else:
            df['streak'].iloc[i] = 0
    
    # Combine components with weights
    df['amplitude_efficiency'] = 0.4 * df['tr_efficiency'] + 0.6 * df['price_efficiency']
    df['volume_amplitude_factor'] = 0.7 * df['corr_deviation'] + 0.3 * df['volume_confirmation']
    df['momentum_factor'] = 0.6 * df['momentum_5'] + 0.4 * df['vol_adj_momentum_diff']
    df['intraday_strength'] = 0.5 * df['relative_close'] + 0.5 * (df['streak'] / 10)
    
    # Final factor combination
    factor = (0.3 * df['amplitude_efficiency'] + 
              0.25 * df['volume_amplitude_factor'] + 
              0.25 * df['momentum_factor'] + 
              0.2 * df['intraday_strength'])
    
    return factor
