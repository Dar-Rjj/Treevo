import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['close_ret'] = df['close'] / df['close'].shift(1) - 1
    df['high_low_range'] = df['high'] - df['low']
    df['upper_shadow'] = df['high'] - df['close']
    df['lower_shadow'] = df['close'] - df['low']
    df['body'] = df['close'] - df['open']
    df['vwap'] = df['amount'] / df['volume']
    
    # Timeframe Fractal Entropy
    df['short_term_entropy'] = (df['high_low_range'] / df['high_low_range'].shift(3)) * df['close_ret']
    df['medium_term_entropy'] = (df['high_low_range'] / df['high_low_range'].shift(10)) * (df['close'] / df['close'].shift(5) - 1)
    df['long_term_entropy'] = (df['high_low_range'] / df['high_low_range'].shift(20)) * (df['close'] / df['close'].shift(20) - 1)
    
    # Volume Entropy Memory
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['vwap_ratio'] = (df['vwap'] / df['vwap'].shift(1)).fillna(1)
    df['volume_entropy_memory'] = (df['volume'] / df['volume_ma_5']) * df['vwap_ratio']
    
    # Bid-Ask Entropy Pressure
    df['bid_entropy_pressure'] = df['lower_shadow'] * df['volume'] * df['close_ret']
    df['ask_entropy_pressure'] = df['upper_shadow'] * df['volume'] * df['close_ret']
    df['net_entropy_absorption'] = df['bid_entropy_pressure'] - df['ask_entropy_pressure']
    df['entropy_asymmetry_ratio'] = df['bid_entropy_pressure'] / df['ask_entropy_pressure'].replace(0, np.nan)
    
    # Multi-scale Entropy Efficiency
    df['intraday_efficiency'] = (abs(df['body']) / df['high_low_range']) * df['volume']
    df['gap_efficiency'] = (abs(df['open'] - df['close'].shift(1)) / df['high_low_range']) * df['volume_entropy_memory']
    df['efficiency_divergence'] = df['intraday_efficiency'] - df['gap_efficiency']
    df['entropy_absorption_momentum'] = (df['net_entropy_absorption'] / df['net_entropy_absorption'].shift(1) - 1).fillna(0)
    
    # Price-Level Entropy Memory
    df['resistance_entropy'] = (df['high'] / df['high'].rolling(window=5).max()) * (df['upper_shadow'] - df['lower_shadow'])
    df['support_entropy'] = (df['low'] / df['low'].rolling(window=5).min()) * (df['upper_shadow'] - df['lower_shadow'])
    df['entropy_memory_breakout'] = (df['resistance_entropy'] - 1) - (df['support_entropy'] - 1)
    df['entropy_memory_ratio'] = df['short_term_entropy'] / df['long_term_entropy'].replace(0, np.nan)
    
    # Entropy Absorption Dynamics
    df['trade_intensity_entropy'] = (df['amount'] / df['high_low_range']) * (df['upper_shadow'] - df['lower_shadow'])
    df['fractal_entropy_absorption'] = df['volume'] * df['net_entropy_absorption'] / df['high_low_range'].replace(0, np.nan)
    df['efficiency_adjusted_entropy'] = df['trade_intensity_entropy'] * df['volume_entropy_memory']
    
    # Multi-scale Entropy Momentum
    df['raw_entropy_momentum'] = df['close_ret']
    df['fractal_entropy_momentum'] = (df['close'] / df['close'].shift(3) - 1) - (df['close'].shift(3) / df['close'].shift(6) - 1)
    
    # Entropy Acceleration
    df['entropy_acceleration'] = ((df['entropy_memory_ratio'] / df['entropy_memory_ratio'].shift(3)) ** (1/3) - 1).fillna(0)
    
    # Core Entropy Velocity Components
    df['absorption_entropy_velocity'] = df['entropy_absorption_momentum'] * df['entropy_memory_breakout']
    df['volume_entropy_velocity'] = df['raw_entropy_momentum'] * df['efficiency_adjusted_entropy']
    
    # Fracture Breakout Entropy
    df['upside_fracture_breakout'] = (df['high'] / df['high'].shift(1) - 1) * df['upper_shadow'] * df['volume']
    df['downside_fracture_breakout'] = (df['low'] / df['low'].shift(1) - 1) * df['lower_shadow'] * df['volume']
    df['fracture_breakout_asymmetry'] = df['upside_fracture_breakout'] - df['downside_fracture_breakout']
    df['efficiency_breakout_entropy'] = df['fracture_breakout_asymmetry'] * df['gap_efficiency']
    
    # Asymmetric Entropy Momentum
    df['asymmetric_entropy_momentum'] = df['fractal_entropy_momentum'] * df['entropy_asymmetry_ratio']
    
    # Divergence and Validation
    df['entropy_fractal_alignment'] = np.sign(df['net_entropy_absorption']) * np.sign(df['entropy_absorption_momentum'])
    df['price_entropy_consistency'] = df['entropy_memory_breakout'] * df['high_low_range'] / (df['entropy_memory_breakout'].shift(1) * df['high_low_range'].shift(1)).replace(0, np.nan)
    
    # Microstructure-Confirmed Entropy
    df['microstructure_confirmed_entropy'] = df['absorption_entropy_velocity'] * df['entropy_fractal_alignment']
    
    # Volume-Efficiency Entropy
    df['volume_efficiency_entropy'] = df['volume_entropy_velocity'] * np.sign(df['efficiency_adjusted_entropy']) * np.sign(df['entropy_absorption_momentum'])
    
    # Breakout Efficiency Entropy
    df['breakout_efficiency_entropy'] = df['efficiency_breakout_entropy'] * np.sign(df['intraday_efficiency'] - df['intraday_efficiency'].shift(1)) * np.sign(df['entropy_acceleration'])
    
    # Range-Enhanced Entropy
    range_expansion = (df['high_low_range'] / df['high_low_range'].shift(1) > 1.2) * df['net_entropy_absorption']
    range_contraction = (df['high_low_range'] / df['high_low_range'].shift(1) < 0.8) * df['net_entropy_absorption']
    df['mean_reversion_entropy'] = (1 - abs(df['close_ret']) / (df['high_low_range'] / df['close'].shift(1))) * df['net_entropy_absorption']
    df['range_enhanced_entropy'] = df['asymmetric_entropy_momentum'] * (range_expansion - range_contraction)
    
    # Final composite alpha construction
    primary_factor = df['microstructure_confirmed_entropy'] * df['price_entropy_consistency']
    secondary_factor = df['volume_efficiency_entropy'] * df['raw_entropy_momentum'].rolling(window=3).apply(lambda x: np.sum(np.sign(x) == np.sign(x.shift(1))) / 3 if len(x) == 3 else np.nan)
    tertiary_factor = df['breakout_efficiency_entropy'] * df['intraday_efficiency'].rolling(window=3).apply(lambda x: np.sum(np.sign(x.diff()) == np.sign(x.diff().shift(1))) / 3 if len(x) == 3 else np.nan)
    quaternary_factor = df['range_enhanced_entropy'] * df['mean_reversion_entropy']
    
    # Composite entropy alpha with entropy-weighted combination
    composite_alpha = (
        0.4 * primary_factor.fillna(0) +
        0.3 * secondary_factor.fillna(0) +
        0.2 * tertiary_factor.fillna(0) +
        0.1 * quaternary_factor.fillna(0)
    )
    
    # Apply volume-based enhancements
    volume_spike = df['volume_entropy_memory'] > 1.5 * (df['upper_shadow'] / df['lower_shadow'].replace(0, np.nan))
    volume_drought = df['volume_entropy_memory'] < 0.7 * (df['upper_shadow'] / df['lower_shadow'].replace(0, np.nan))
    
    composite_alpha = np.where(volume_spike, composite_alpha * 1.3, composite_alpha)
    composite_alpha = np.where(volume_drought, composite_alpha * 0.7, composite_alpha)
    
    result = pd.Series(composite_alpha, index=df.index)
    return result
