import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Reversal Detection Component
    # Calculate recent price extremes
    data['highest_10d'] = data['high'].rolling(window=10, min_periods=10).max()
    data['lowest_10d'] = data['low'].rolling(window=10, min_periods=10).min()
    
    # Detect breakout reversal patterns
    data['break_high'] = (data['close'] > data['high'].shift(1)) & (data['close'] > data['highest_10d'].shift(1))
    data['break_low'] = (data['close'] < data['low'].shift(1)) & (data['close'] < data['lowest_10d'].shift(1))
    
    # Distance from extremes
    data['dist_from_high'] = (data['close'] - data['highest_10d']) / data['highest_10d']
    data['dist_from_low'] = (data['close'] - data['lowest_10d']) / data['lowest_10d']
    
    # Momentum divergence
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_divergence'] = data['momentum_5d'] - data['momentum_20d']
    
    # Volatility Expansion Analysis
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['avg_tr_10d'] = data['true_range'].rolling(window=10, min_periods=10).mean()
    data['tr_expansion'] = data['true_range'] / data['avg_tr_10d'] - 1
    
    # Price volatility (high-low based)
    data['daily_range'] = data['high'] - data['low']
    data['volatility_20d'] = data['daily_range'].rolling(window=20, min_periods=20).std()
    
    # Volatility-adjusted momentum
    data['momentum_5d_vol_adj'] = data['momentum_5d'] / (data['volatility_20d'] + 1e-8)
    data['momentum_20d_vol_adj'] = data['momentum_20d'] / (data['volatility_20d'] + 1e-8)
    
    # Volume Confirmation Component
    # Volume trend analysis
    data['volume_5d_slope'] = data['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    data['volume_20d_slope'] = data['volume'].rolling(window=20, min_periods=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
    )
    
    # Volume-price divergence detection
    data['volume_trend_divergence'] = np.sign(data['volume_5d_slope']) * np.sign(data['momentum_5d'])
    data['volume_divergence_strength'] = abs(data['volume_5d_slope']) * abs(data['momentum_5d'])
    
    # Integrated Alpha Factor Generation
    # Combine reversal and volatility signals
    data['reversal_signal'] = 0
    data.loc[data['break_high'], 'reversal_signal'] = -data['dist_from_high'] * data['tr_expansion']
    data.loc[data['break_low'], 'reversal_signal'] = -data['dist_from_low'] * data['tr_expansion']
    
    # Add volatility-adjusted momentum confirmation
    data['momentum_confirmation'] = data['momentum_5d_vol_adj'] - data['momentum_20d_vol_adj']
    
    # Apply volume divergence multiplier
    data['volume_multiplier'] = 1 + data['volume_divergence_strength'] * data['volume_trend_divergence']
    
    # Generate final alpha factor
    data['alpha_factor'] = (
        data['reversal_signal'] * 
        data['momentum_confirmation'] * 
        data['volume_multiplier']
    )
    
    # Clean up intermediate columns
    result = data['alpha_factor'].copy()
    
    return result
