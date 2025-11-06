import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price features
    df['range'] = df['high'] - df['low']
    df['gap'] = abs(df['open'] - df['close'].shift(1))
    df['close_open_diff'] = df['close'] - df['open']
    df['mid_price'] = (df['high'] + df['low']) / 2
    
    # Calculate rolling statistics
    df['vol_5d'] = df['close'].pct_change().rolling(5).std()
    df['vol_10d'] = df['close'].pct_change().rolling(10).std()
    df['vol_20d'] = df['close'].pct_change().rolling(20).std()
    df['ret_5d'] = df['close'].pct_change(5)
    df['avg_volume_5d'] = df['volume'].rolling(5).mean()
    df['avg_range_20d'] = df['range'].rolling(20).mean()
    
    # Calculate volume-based features
    df['up_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['down_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    
    # Calculate fractal dimensions (simplified)
    def calculate_fractal(series, window=3):
        return series.rolling(window).apply(lambda x: (x.max() - x.min()) / (x.std() + 1e-8))
    
    df['price_fractal_3d'] = calculate_fractal(df['close'], 3)
    df['volume_fractal_3d'] = calculate_fractal(df['volume'], 3)
    
    # Calculate peer averages (using rolling mean as proxy for peer average)
    df['peer_range'] = df['range'].rolling(10).mean()
    df['peer_gap'] = df['gap'].rolling(10).mean()
    df['peer_vol_5d'] = df['vol_5d'].rolling(10).mean()
    df['peer_vol_20d'] = df['vol_20d'].rolling(10).mean()
    df['peer_volume_ratio'] = (df['volume'] / df['avg_volume_5d']).rolling(10).mean()
    df['peer_volume_fractal_mom'] = df['volume_fractal_3d'].pct_change(3).rolling(10).mean()
    
    # Cross-Asset Volatility Structure
    df['relative_intraday_vol_ratio'] = df['range'] / (df['peer_range'] + 1e-8)
    df['cross_asset_gap_efficiency'] = df['gap'] / (df['peer_gap'] + 1e-8)
    df['sector_volatility_persistence'] = (df['vol_5d'] / (df['vol_10d'] + 1e-8)) * np.sign(df['ret_5d'])
    
    # Cross-Asset Volume-Volatility Integration
    df['relative_volume_vol_divergence'] = (df['volume'] / (df['avg_volume_5d'] + 1e-8)) - df['peer_volume_ratio']
    volume_pressure = (df['up_volume'] - df['down_volume']) / (df['up_volume'] + df['down_volume'] + 1e-8)
    df['sector_volume_pressure'] = volume_pressure * df['close_open_diff']
    df['cross_asset_volume_fractal'] = df['volume_fractal_3d'] * df['price_fractal_3d'] * df['ret_5d']
    
    # Cross-Asset Microstructure Dynamics
    df['opening_range'] = (df['open'] - df['low']) / (df['range'] + 1e-8)
    df['relative_opening_range_capture'] = df['opening_range'] * df['opening_range'].rolling(10).mean()
    df['sector_midday_momentum'] = abs(df['mid_price'] - df['open']) / (abs(df['close'] - df['mid_price']) + 1e-8)
    df['cross_asset_closing_efficiency'] = (abs(df['close_open_diff']) / (df['range'] + 1e-8)) * np.sign(df['close_open_diff'])
    
    # Cross-Asset Regime Momentum Framework
    df['sector_high_vol_momentum'] = df['ret_5d'] * (df['vol_5d'] / (df['peer_vol_5d'] + 1e-8))
    df['cross_asset_low_vol_momentum'] = df['ret_5d'] * (df['peer_vol_20d'] / (df['vol_5d'] + 1e-8))
    df['sector_transition_signal'] = np.sign(df['vol_5d'] - df['peer_vol_5d']) * (df['range'] / (df['avg_range_20d'] + 1e-8))
    
    # Cross-Asset Fractal Integration
    df['sector_volume_price_fractal'] = df['volume_fractal_3d'].rolling(5).corr(df['price_fractal_3d'])
    df['cross_asset_fractal_divergence'] = df['volume_fractal_3d'].pct_change(3) - df['peer_volume_fractal_mom']
    
    # Relative volume-position calculation
    df['min_low_3d'] = df['low'].rolling(3).min()
    df['max_high_3d'] = df['high'].rolling(3).max()
    df['relative_volume_position'] = ((df['close'] - df['min_low_3d']) / 
                                    (df['max_high_3d'] - df['min_low_3d'] + 1e-8)) * df['volume_fractal_3d']
    
    # Breakout calculations
    df['breakout_strength'] = (df['close'] - df['low'].rolling(5).min()) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-8)
    df['volume_breakout_efficiency'] = df['volume'] / df['avg_volume_5d']
    df['transition_confidence'] = df['sector_transition_signal'].abs()
    
    # Peer averages for breakout metrics
    df['peer_breakout_strength'] = df['breakout_strength'].rolling(10).mean()
    df['peer_volume_breakout'] = df['volume_breakout_efficiency'].rolling(10).mean()
    df['peer_transition_confidence'] = df['transition_confidence'].rolling(10).mean()
    
    # Relative breakout metrics
    df['relative_breakout_momentum'] = df['breakout_strength'] / (df['peer_breakout_strength'] + 1e-8)
    df['cross_asset_volume_breakout'] = df['volume_breakout_efficiency'] - df['peer_volume_breakout']
    df['sector_transition_validation'] = df['transition_confidence'] * df['peer_transition_confidence']
    
    # Breakout multipliers
    df['breakout_leadership_multiplier'] = 1 + df['breakout_strength'] * 0.15
    df['volume_breakout_multiplier'] = 1 + df['volume_breakout_efficiency'] * 0.1
    df['transition_leadership'] = 1 + df['transition_confidence'] * 0.08
    
    # Core Cross-Fractal Components
    volatility_volume_core = df['sector_volatility_persistence'] * df['relative_volume_vol_divergence']
    microstructure_momentum = df['cross_asset_closing_efficiency'] * df['sector_midday_momentum']
    fractal_alignment = df['sector_volume_price_fractal'] * df['relative_volume_position']
    
    # Primary signal and multipliers
    primary_signal = volatility_volume_core * microstructure_momentum * fractal_alignment
    breakout_multiplier = df['breakout_leadership_multiplier'] * df['volume_breakout_multiplier']
    transition_multiplier = df['transition_leadership'] * df['sector_transition_signal']
    
    # Regime filters
    sector_vol_filter = ((df['vol_5d'] / (df['peer_vol_5d'] + 1e-8)).between(0.7, 1.3)).astype(float)
    volume_persistence = (df['volume'] / (df['avg_volume_5d'] + 1e-8)).between(0.8, 1.2).astype(float)
    regime_weight = np.where((sector_vol_filter == 1) & (volume_persistence == 1), 1.2, 0.8)
    
    # Final signal construction
    raw_cross_alpha = primary_signal * breakout_multiplier * transition_multiplier
    filtered_cross_alpha = raw_cross_alpha * regime_weight
    signal_strength = filtered_cross_alpha * df['cross_asset_fractal_divergence']
    
    # Final alpha
    cross_asset_vol_fractal_momentum = signal_strength * df['relative_intraday_vol_ratio']
    final_alpha = cross_asset_vol_fractal_momentum * df['cross_asset_gap_efficiency']
    
    # Fill NaN values and return
    result = final_alpha.fillna(0)
    return result
