import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-Period Momentum Elasticity Analysis
    # Short-term momentum efficiency
    data['short_mom_eff'] = (data['close'] - data['close'].shift(3)) / (
        data['high'].rolling(3).max() - data['low'].rolling(3).min())
    
    # Medium-term momentum efficiency
    data['medium_mom_eff'] = (data['close'] - data['close'].shift(10)) / (
        data['high'].rolling(10).max() - data['low'].rolling(10).min())
    
    # Long-term momentum efficiency
    data['long_mom_eff'] = (data['close'] - data['close'].shift(20)) / (
        data['high'].rolling(20).max() - data['low'].rolling(20).min())
    
    # Momentum elasticity divergence
    data['mom_elasticity'] = data['short_mom_eff'] * data['medium_mom_eff'] * data['long_mom_eff']
    
    # Volume-Price Divergence Context
    # Volume breakout detection
    data['vol_breakout'] = data['volume'] / data['volume'].rolling(20).mean()
    
    # Price efficiency ratio
    data['price_eff_ratio'] = (data['close'] - data['prev_close']) / data['true_range']
    
    # Volume persistence (5-day volume trend slope)
    data['vol_trend'] = data['volume'].rolling(5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0] if len(x) == 5 else np.nan)
    
    # Volume-price divergence
    data['vol_price_div'] = data['vol_breakout'] * data['price_eff_ratio'] * data['vol_trend']
    
    # Intraday Efficiency with Liquidity Context
    # Intraday strength ratio
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Range efficiency
    data['range_eff'] = data['true_range'] / (data['high'] - data['low'])
    
    # Daily liquidity intensity
    data['liquidity_intensity'] = data['volume'] * data['amount']
    
    # Efficiency-liquidity signal
    data['eff_liquidity_signal'] = data['intraday_strength'] * data['range_eff'] * data['liquidity_intensity']
    
    # Divergence Synthesis and Elasticity
    # Momentum-volume divergence
    data['mom_vol_div'] = data['mom_elasticity'] * data['vol_price_div']
    
    # Efficiency-weighted divergence
    data['eff_weighted_div'] = data['mom_vol_div'] * data['eff_liquidity_signal']
    
    # Level-adaptive scaling
    data['high_20d'] = data['high'].rolling(20).max()
    data['low_20d'] = data['low'].rolling(20).min()
    data['price_level'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'])
    data['level_adaptive_scale'] = 1 - abs(data['price_level'] - 0.5)
    data['level_adaptive_signal'] = data['eff_weighted_div'] * data['level_adaptive_scale']
    
    # Signal Generation with Adaptive Logic
    # Liquidity persistence adjustment (5-day liquidity trend)
    data['liquidity_trend'] = data['liquidity_intensity'].rolling(5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0] if len(x) == 5 else np.nan)
    
    # Apply contrarian logic (negative scaling for extreme price levels)
    extreme_condition = (data['price_level'] > 0.8) | (data['price_level'] < 0.2)
    data['contrarian_signal'] = np.where(extreme_condition, -data['level_adaptive_signal'], data['level_adaptive_signal'])
    
    # Volume-confirmation weighting
    data['vol_10d_avg'] = data['volume'].rolling(10).mean()
    data['vol_confirmation'] = data['volume'] / data['vol_10d_avg']
    
    # Final alpha
    alpha = data['contrarian_signal'] * data['liquidity_trend'] / data['vol_confirmation']
    
    # Return the alpha series with the same index as input
    return alpha
