import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a composite alpha factor combining dynamic volatility-adjusted momentum,
    volume-confirmed breakouts, efficiency-based reversals, and multi-scale momentum alignment.
    """
    df = data.copy()
    
    # Dynamic Volatility-Adjusted Momentum
    # Multi-timeframe momentum
    df['mom_5'] = df['close'].shift(1) / df['close'].shift(5) - 1
    df['mom_10'] = df['close'].shift(1) / df['close'].shift(10) - 1
    df['mom_20'] = df['close'].shift(1) / df['close'].shift(20) - 1
    
    # Volatility regime
    df['daily_range'] = (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    df['rolling_vol'] = df['close'].shift(1).rolling(window=5).std()
    
    # Volatility regime classification
    df['range_percentile'] = df['daily_range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x[:-1], 80)) if len(x) > 1 else 0, raw=False
    )
    df['low_vol_regime'] = df['daily_range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] < np.percentile(x[:-1], 20)) if len(x) > 1 else 0, raw=False
    )
    
    # Adaptive momentum signal
    df['vol_adj_mom'] = np.where(
        df['low_vol_regime'] == 1,
        df['mom_10'] / (df['rolling_vol'] + 1e-8),
        np.where(
            df['range_percentile'] == 1,
            -df['mom_10'] / (df['rolling_vol'] + 1e-8),
            df['mom_10'] / (df['rolling_vol'] + 1e-8)
        )
    )
    
    # Volume-Confirmed Breakout Factor
    df['prev_high_max'] = df['high'].shift(1).rolling(window=9).max()
    df['prev_low_min'] = df['low'].shift(1).rolling(window=9).min()
    
    # Breakout detection
    df['upper_breakout'] = (df['close'].shift(1) > df['prev_high_max']).astype(int)
    df['lower_breakout'] = (df['close'].shift(1) < df['prev_low_min']).astype(int)
    
    # Breakout magnitude
    df['breakout_magnitude'] = np.where(
        df['upper_breakout'] == 1,
        (df['close'].shift(1) - df['prev_high_max']) / df['prev_high_max'],
        np.where(
            df['lower_breakout'] == 1,
            (df['prev_low_min'] - df['close'].shift(1)) / df['prev_low_min'],
            0
        )
    )
    
    # Volume confirmation
    df['avg_volume_5'] = df['volume'].shift(1).rolling(window=4).mean()
    df['volume_acceleration'] = df['volume'].shift(1) / (df['avg_volume_5'] + 1e-8)
    
    # Volume persistence (simplified)
    df['volume_above_avg'] = (df['volume'].shift(1) > df['volume'].shift(2).rolling(window=5).mean()).astype(int)
    df['volume_persistence'] = df['volume_above_avg'].rolling(window=5).sum()
    
    # Breakout strength
    df['breakout_strength'] = df['breakout_magnitude'] * df['volume_acceleration']
    
    # Efficiency-Based Reversal Indicator
    df['daily_efficiency'] = abs(df['close'].shift(1) - df['open'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)
    
    # Efficiency persistence (simplified)
    df['efficient_day'] = (df['daily_efficiency'] > 0.7).astype(int)
    df['efficiency_persistence'] = df['efficient_day'].rolling(window=5).sum()
    
    # Reversal signal based on efficiency divergence
    df['price_change'] = df['close'].shift(1) / df['close'].shift(2) - 1
    df['efficiency_reversal'] = np.where(
        (df['daily_efficiency'] > 0.8) & (df['price_change'].abs() > 0.02),
        -df['price_change'] * df['volume_acceleration'],
        0
    )
    
    # Multi-Scale Momentum Alignment
    # Short-term alignment (3-day, 5-day)
    df['mom_3'] = df['close'].shift(1) / df['close'].shift(3) - 1
    df['short_term_alignment'] = (df['mom_3'] * df['mom_5']) / (abs(df['mom_3']) + abs(df['mom_5']) + 1e-8)
    
    # Medium-term alignment (8-day, 13-day)
    df['mom_8'] = df['close'].shift(1) / df['close'].shift(8) - 1
    df['mom_13'] = df['close'].shift(1) / df['close'].shift(13) - 1
    df['medium_term_alignment'] = (df['mom_8'] * df['mom_13']) / (abs(df['mom_8']) + abs(df['mom_13']) + 1e-8)
    
    # Cross-timeframe signals
    df['momentum_convergence'] = (df['short_term_alignment'] * df['medium_term_alignment']) * df['volume_acceleration']
    
    # Regime-Specific Alpha Composite
    df['low_vol_composite'] = (
        df['vol_adj_mom'].rolling(window=5).mean() +  # Momentum persistence
        df['breakout_strength'] * 0.5 +  # Volume-confirmed trend
        df['daily_efficiency'] * df['mom_10']  # Efficiency-weighted momentum
    )
    
    df['high_vol_composite'] = (
        -df['mom_10'].rolling(window=3).mean() +  # Mean reversion
        (df['volume_acceleration'] - df['volume_persistence']/5) * 2 +  # Volume divergence
        df['efficiency_reversal'] * df['rolling_vol']  # Volatility-scaled reversals
    )
    
    # Final composite alpha factor
    df['alpha_factor'] = np.where(
        df['low_vol_regime'] == 1,
        df['low_vol_composite'],
        np.where(
            df['range_percentile'] == 1,
            df['high_vol_composite'],
            (df['low_vol_composite'] + df['high_vol_composite']) / 2
        )
    ) + df['momentum_convergence'] * 0.3
    
    return df['alpha_factor']
