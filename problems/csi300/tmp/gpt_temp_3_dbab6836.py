import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Period Momentum Elasticity
    df['momentum_short'] = df['close'].pct_change(periods=3)
    df['momentum_medium'] = df['close'].pct_change(periods=10)
    df['momentum_long'] = df['close'].pct_change(periods=20)
    df['momentum_elasticity'] = df['momentum_short'] * df['momentum_medium'] * df['momentum_long']
    
    # Volume Acceleration Analysis
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_shock'] = df['volume'] / df['volume_ma_20']
    
    # Volume persistence (5-day volume trend slope)
    volume_trend = []
    for i in range(len(df)):
        if i >= 4:
            y = df['volume'].iloc[i-4:i+1].values
            x = np.arange(5)
            slope = np.polyfit(x, y, 1)[0]
            volume_trend.append(slope)
        else:
            volume_trend.append(np.nan)
    df['volume_persistence'] = volume_trend
    df['volume_acceleration'] = df['volume_shock'] * df['volume_persistence']
    
    # Intraday Efficiency Context
    df['intraday_strength'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # True range calculation
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['range_efficiency'] = df['true_range'] / (df['high'] - df['low']).replace(0, np.nan)
    
    df['efficiency_composite'] = df['intraday_strength'] * df['range_efficiency']
    
    # Price-Level Volume Elasticity
    df['high_20'] = df['high'].rolling(window=20).max()
    df['low_20'] = df['low'].rolling(window=20).min()
    df['relative_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20']).replace(0, np.nan)
    df['volume_pressure_divergence'] = df['volume_acceleration'] * (1 - abs(df['relative_position'] - 0.5))
    df['level_adaptive_scaling'] = df['volume_pressure_divergence'] * df['efficiency_composite']
    
    # Liquidity-Weighted Momentum Decay
    df['liquidity_intensity'] = df['volume'] * df['amount']
    
    # Liquidity persistence (5-day liquidity trend slope)
    liquidity_trend = []
    for i in range(len(df)):
        if i >= 4:
            y = df['liquidity_intensity'].iloc[i-4:i+1].values
            x = np.arange(5)
            slope = np.polyfit(x, y, 1)[0]
            liquidity_trend.append(slope)
        else:
            liquidity_trend.append(np.nan)
    df['liquidity_persistence'] = liquidity_trend
    
    df['momentum_decay'] = df['momentum_elasticity'] / df['liquidity_persistence'].replace(0, np.nan)
    df['liquidity_momentum_interaction'] = df['momentum_decay'] * df['liquidity_intensity']
    
    # Combined Divergence Signal
    df['volume_price_efficiency_divergence'] = df['level_adaptive_scaling'] * df['liquidity_momentum_interaction']
    
    # Contrarian logic application
    extreme_high_mask = df['relative_position'] > 0.8
    extreme_low_mask = df['relative_position'] < 0.2
    df['final_alpha'] = df['volume_price_efficiency_divergence']
    df.loc[extreme_high_mask | extreme_low_mask, 'final_alpha'] = -df.loc[extreme_high_mask | extreme_low_mask, 'final_alpha']
    
    # Clean up intermediate columns
    result = df['final_alpha'].copy()
    
    return result
