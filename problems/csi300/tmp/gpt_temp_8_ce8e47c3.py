import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Horizon Momentum Analysis
    df['high_freq_momentum'] = df['close'] / df['close'].shift(1) - 1
    df['medium_freq_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['low_freq_momentum'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_acceleration'] = (df['high_freq_momentum'] - df['medium_freq_momentum']) / 4
    
    # Volume Pressure Component
    df['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
    df['intraday_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['cumulative_pressure'] = df['intraday_pressure'].rolling(window=3, min_periods=1).sum()
    
    # Volume persistence: count of consecutive same-sign volume changes over 3 days
    volume_change = df['volume'] - df['volume'].shift(1)
    volume_sign_change = volume_change * volume_change.shift(1)
    df['volume_persistence'] = volume_sign_change.rolling(window=3, min_periods=1).apply(
        lambda x: sum((x > 0) & (x.notna())), raw=False
    )
    
    # Divergence Core Calculation
    df['basic_divergence'] = -df['momentum_acceleration'] * df['volume_momentum']
    
    close_change_sign = np.sign(df['close'] - df['close'].shift(1))
    volume_change_sign = np.sign(df['volume'] - df['volume'].shift(1))
    df['volume_coherent_divergence'] = df['basic_divergence'] * close_change_sign * volume_change_sign
    
    df['pressure_enhanced_divergence'] = df['volume_coherent_divergence'] * df['cumulative_pressure']
    
    # Price Efficiency Context
    df['intraday_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['overnight_efficiency'] = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    df['total_price_efficiency'] = df['intraday_efficiency'] + df['overnight_efficiency']
    df['efficiency_consistency'] = df['total_price_efficiency'].rolling(window=5, min_periods=1).std()
    
    # Volatility Regime Classification
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['range_efficiency'] = df['daily_range'] / df['daily_range'].rolling(window=5, min_periods=1).mean()
    df['volatility_ratio'] = df['close'].rolling(window=5, min_periods=1).std() / df['close'].rolling(window=10, min_periods=1).std()
    
    high_vol_condition1 = df['volatility_ratio'] > 1
    high_vol_condition2 = df['daily_range'] > df['daily_range'].rolling(window=20, min_periods=1).mean()
    df['high_volatility'] = high_vol_condition1 | high_vol_condition2
    
    # Regime-Adaptive Enhancement
    # High Volatility Enhancement
    df['range_breakout'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_significance'] = df['volume'] / df['volume'].rolling(window=10, min_periods=1).mean()
    df['high_vol_enhanced'] = df['pressure_enhanced_divergence'] * df['range_breakout'] * df['volume_significance']
    
    # Low Volatility Enhancement
    df['trend_persistence'] = np.sign(df['close'] - df['close'].shift(1)).rolling(window=5, min_periods=1).sum()
    df['low_vol_enhanced'] = df['pressure_enhanced_divergence'] * df['trend_persistence'] * df['total_price_efficiency']
    
    # Synchronization Strength Assessment
    strong_sync = (df['volume_coherent_divergence'] > 0) & (df['volume_persistence'] >= 2)
    weak_sync = (df['volume_coherent_divergence'] <= 0) | (df['volume_persistence'] < 2)
    
    # Final enhancement based on volatility regime
    df['regime_enhanced_factor'] = np.where(
        df['high_volatility'] & strong_sync,
        df['high_vol_enhanced'],
        np.where(
            ~df['high_volatility'] & strong_sync,
            df['low_vol_enhanced'],
            df['pressure_enhanced_divergence']  # fallback for weak sync
        )
    )
    
    # Signal Refinement
    # Apply efficiency consistency weighting
    efficiency_weight = 1 / (1 + df['efficiency_consistency'])
    weighted_factor = df['regime_enhanced_factor'] * efficiency_weight
    
    # Multiply by absolute momentum convergence for directional strength
    momentum_convergence = (df['high_freq_momentum'].abs() + df['medium_freq_momentum'].abs() + df['low_freq_momentum'].abs()) / 3
    directional_factor = weighted_factor * momentum_convergence
    
    # Smooth with 3-day moving average
    final_alpha = directional_factor.rolling(window=3, min_periods=1).mean()
    
    return final_alpha
