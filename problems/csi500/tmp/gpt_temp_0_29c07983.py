import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Period Momentum Divergence
    # Short-term momentum components
    data['mom_3d'] = data['close'].pct_change(3)
    data['mom_5d'] = data['close'].pct_change(5)
    data['mom_10d_st'] = data['close'].pct_change(10)
    
    # Medium-term momentum components
    data['mom_10d_mt'] = data['close'].pct_change(10)
    data['mom_15d'] = data['close'].pct_change(15)
    data['mom_20d'] = data['close'].pct_change(20)
    
    # Divergence patterns
    data['div_3_10'] = np.sign(data['mom_3d']) * np.sign(data['mom_10d_st'])
    data['div_5_15'] = np.sign(data['mom_5d']) * np.sign(data['mom_15d'])
    data['div_10_20'] = np.sign(data['mom_10d_mt']) * np.sign(data['mom_20d'])
    
    # Divergence strength
    data['div_strength_short'] = (data['mom_3d'] - data['mom_10d_st']) * (data['div_3_10'] < 0)
    data['div_strength_medium'] = (data['mom_5d'] - data['mom_15d']) * (data['div_5_15'] < 0)
    data['div_strength_long'] = (data['mom_10d_mt'] - data['mom_20d']) * (data['div_10_20'] < 0)
    
    # Assess Volatility Regime Dynamics
    # Calculate daily range
    data['daily_range'] = data['high'] - data['low']
    
    # Average True Range (5-day)
    data['atr_5d'] = data['daily_range'].rolling(window=5).mean()
    
    # Range volatility (std of daily ranges)
    data['range_vol_5d'] = data['daily_range'].rolling(window=5).std()
    
    # Volatility regime classification
    data['vol_regime'] = 0  # Normal
    high_vol_threshold = data['atr_5d'].rolling(window=20).quantile(0.7)
    low_vol_threshold = data['atr_5d'].rolling(window=20).quantile(0.3)
    
    data.loc[data['atr_5d'] > high_vol_threshold, 'vol_regime'] = 1  # High volatility
    data.loc[data['atr_5d'] < low_vol_threshold, 'vol_regime'] = -1  # Low volatility
    
    # Volatility trend changes
    data['range_mom_10d'] = data['daily_range'].pct_change(10)
    data['atr_roc'] = data['atr_5d'].pct_change(5)
    
    # Regime-adaptive scaling weights
    data['regime_weight_short'] = np.where(data['vol_regime'] == 1, 1.5, 
                                          np.where(data['vol_regime'] == -1, 0.5, 1.0))
    data['regime_weight_medium'] = np.where(data['vol_regime'] == 1, 0.5, 
                                           np.where(data['vol_regime'] == -1, 1.5, 1.0))
    
    # Incorporate Liquidity Acceleration Dynamics
    # Volume acceleration patterns
    data['vol_slope_3d'] = data['volume'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / 2 if len(x) == 3 else np.nan, raw=False
    )
    data['vol_slope_5d'] = data['volume'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / 4 if len(x) == 5 else np.nan, raw=False
    )
    
    data['vol_accel_3d'] = data['vol_slope_3d'].diff(2)
    data['vol_accel_5d'] = data['vol_slope_5d'].diff(3)
    
    # Liquidity regime shifts
    data['liquidity_regime'] = np.sign(data['vol_accel_5d'])
    
    # Enhance divergence signals with liquidity
    data['div_short_liquidity'] = data['div_strength_short'] * data['vol_accel_3d']
    data['div_medium_liquidity'] = data['div_strength_medium'] * data['vol_accel_5d']
    
    # Regime-dependent liquidity weights
    data['liquidity_weight'] = np.where(data['liquidity_regime'] > 0, 1.2, 0.8)
    
    # Detect Acceleration-Weighted Breakout Conditions
    # Momentum acceleration
    data['mom_accel_3_5'] = data['mom_5d'].diff(3)
    data['mom_accel_5_10'] = data['mom_10d_st'].diff(5)
    
    # Acceleration consistency
    data['accel_consistency'] = np.sign(data['mom_accel_3_5']) * np.sign(data['mom_accel_5_10'])
    
    # Breakout strength
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    
    data['breakout_high'] = (data['close'] > data['high_20d'].shift(1)).astype(int)
    data['breakout_low'] = (data['close'] < data['low_20d'].shift(1)).astype(int)
    
    # Breakout persistence
    data['breakout_persistence'] = (data['breakout_high'] + data['breakout_low']).rolling(window=3).sum()
    
    # Breakout acceleration
    data['breakout_accel'] = data['close'].pct_change(3) * data['breakout_persistence']
    
    # Regime-specific breakout signals
    data['breakout_signal'] = data['breakout_accel'] * np.where(
        data['vol_regime'] == 1, 0.7, 
        np.where(data['vol_regime'] == -1, 1.3, 1.0)
    )
    
    # Combine Components with Dynamic Blending
    # Composite signal strength
    data['composite_divergence'] = (
        data['div_short_liquidity'] * data['regime_weight_short'] +
        data['div_medium_liquidity'] * data['regime_weight_medium']
    ) * data['liquidity_weight']
    
    data['composite_signal'] = data['composite_divergence'] * data['breakout_signal']
    
    # Directional probability weights
    bullish_condition = (data['composite_divergence'] > 0) & (data['vol_accel_3d'] > 0)
    bearish_condition = (data['composite_divergence'] < 0) & (data['vol_accel_3d'] < 0)
    
    data['directional_weight'] = 0
    data.loc[bullish_condition, 'directional_weight'] = data['composite_signal'] * 1.2
    data.loc[bearish_condition, 'directional_weight'] = data['composite_signal'] * 0.8
    
    # Apply non-linear response to extreme signals
    extreme_threshold = data['composite_signal'].abs().rolling(window=20).quantile(0.9)
    data['final_alpha'] = np.where(
        data['composite_signal'].abs() > extreme_threshold,
        np.tanh(data['composite_signal'] / extreme_threshold) * extreme_threshold,
        data['composite_signal']
    )
    
    # Final alpha factor
    alpha = data['final_alpha'].fillna(0)
    
    return alpha
