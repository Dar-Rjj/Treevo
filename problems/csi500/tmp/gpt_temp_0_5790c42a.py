import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate novel alpha factors based on price-volume interaction dynamics,
    multi-timeframe acceleration patterns, microstructure efficiency factors,
    and persistence/reversal dynamics.
    """
    df = data.copy()
    
    # Price-Volume Interaction Dynamics
    # Volume-Weighted Momentum Acceleration
    df['price_return_1d'] = df['close'] / df['close'].shift(1) - 1
    df['volume_growth_1d'] = df['volume'] / df['volume'].shift(1)
    df['vol_weighted_momentum'] = df['price_return_1d'] * df['volume_growth_1d']
    
    # Range Efficiency with Volume Confirmation
    df['efficiency_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['range_efficiency_vol'] = df['efficiency_ratio'] * df['volume']
    
    # Gap Pressure with Volume Intensity
    df['overnight_gap'] = df['open'] / df['close'].shift(1) - 1
    df['gap_pressure'] = abs(df['overnight_gap']) * df['volume'] * np.sign(df['overnight_gap'])
    
    # Multi-Timeframe Acceleration Patterns
    # Breakout Volume Acceleration
    df['high_5d_roll'] = df['high'].rolling(window=5).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    df['breakout_5d'] = df['close'] / df['high_5d_roll']
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean().shift(1)
    df['volume_accel_5d'] = df['volume'] / df['volume_5d_avg']
    df['breakout_vol_accel'] = df['breakout_5d'] * df['volume_accel_5d']
    
    # Momentum-Volume Divergence
    df['price_momentum_3d'] = df['close'] / df['close'].shift(3)
    df['volume_momentum_3d'] = df['volume'] / df['volume'].shift(3)
    df['momentum_volume_divergence'] = df['price_momentum_3d'] / df['volume_momentum_3d']
    
    # Range Expansion with Volume Confirmation
    df['range_today'] = df['high'] - df['low']
    df['range_5d_avg'] = (df['high'] - df['low']).rolling(window=5).mean().shift(1)
    df['range_expansion'] = df['range_today'] / df['range_5d_avg']
    df['volume_5d_avg_range'] = df['volume'].rolling(window=5).mean().shift(1)
    df['volume_expansion'] = df['volume'] / df['volume_5d_avg_range']
    df['range_vol_confirmation'] = df['range_expansion'] * df['volume_expansion']
    
    # Microstructure Efficiency Factors
    # Price Discovery Efficiency
    df['intraday_movement'] = abs(df['close'] - df['open'])
    df['available_range'] = df['high'] - df['low']
    df['price_discovery_efficiency'] = df['intraday_movement'] / df['available_range']
    
    # Volume-Weighted Price Premium
    df['vwap'] = df['amount'] / df['volume']
    df['vwap_premium'] = df['vwap'] - df['close']
    
    # Opening Session Momentum
    df['opening_gap'] = df['open'] - df['close'].shift(1)
    df['intraday_momentum'] = df['close'] - df['open']
    df['opening_session_momentum'] = df['opening_gap'] * df['intraday_momentum'] * df['volume']
    
    # Persistence and Reversal Dynamics
    # Volume-Dampened Mean Reversion
    df['close_5d_avg'] = df['close'].rolling(window=5).mean().shift(1)
    df['price_extension_5d'] = df['close'] / df['close_5d_avg']
    df['volume_5d_avg_rev'] = df['volume'].rolling(window=5).mean().shift(1)
    df['volume_ratio_5d'] = df['volume'] / df['volume_5d_avg_rev']
    df['volume_dampened_reversion'] = df['price_extension_5d'] * (1 / df['volume_ratio_5d'])
    
    # Breakout Volume Confirmation
    df['high_10d_roll'] = df['high'].rolling(window=10).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    df['new_high_breakout'] = (df['close'] > df['high_10d_roll']).astype(float)
    df['volume_10d_avg'] = df['volume'].rolling(window=10).mean().shift(1)
    df['volume_spike_10d'] = df['volume'] / df['volume_10d_avg']
    df['breakout_vol_confirmation'] = df['new_high_breakout'] * df['volume_spike_10d']
    
    # Accumulation Flow Momentum
    df['money_flow'] = ((df['high'] + df['low'] + df['close']) / 3) * df['volume']
    df['cumulative_flow_5d'] = df['money_flow'].rolling(window=5).sum()
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5)
    df['accumulation_flow_momentum'] = df['cumulative_flow_5d'] / df['price_momentum_5d']
    
    # Combine all factors using equal weighting
    factor_columns = [
        'vol_weighted_momentum', 'range_efficiency_vol', 'gap_pressure',
        'breakout_vol_accel', 'momentum_volume_divergence', 'range_vol_confirmation',
        'price_discovery_efficiency', 'vwap_premium', 'opening_session_momentum',
        'volume_dampened_reversion', 'breakout_vol_confirmation', 'accumulation_flow_momentum'
    ]
    
    # Standardize and combine factors
    combined_factor = pd.Series(index=df.index, dtype=float)
    for col in factor_columns:
        if col in df.columns:
            # Remove outliers and standardize
            factor_series = df[col].replace([np.inf, -np.inf], np.nan)
            factor_series = (factor_series - factor_series.mean()) / factor_series.std()
            combined_factor = combined_factor.add(factor_series, fill_value=0)
    
    return combined_factor
