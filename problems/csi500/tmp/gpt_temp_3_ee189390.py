import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Momentum Regime Factor
    Combines momentum persistence, volume concentration, price-volume divergence,
    and liquidity conditions to identify sustainable price movements.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price and volume metrics
    df = df.copy()
    df['returns_5d'] = df['close'].pct_change(5)
    df['volume_20d_median'] = df['volume'].rolling(window=20, min_periods=10).median()
    df['volume_spike'] = df['volume'] / df['volume_20d_median']
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    # Directional Momentum Strength Analysis
    df['momentum_direction'] = np.sign(df['returns_5d'])
    
    # Calculate momentum continuation rate (5-day window)
    momentum_continuation = []
    for i in range(len(df)):
        if i < 5:
            momentum_continuation.append(0)
            continue
        window = df['momentum_direction'].iloc[i-5:i+1]
        continuation_rate = (window == window.iloc[-1]).sum() / 6
        momentum_continuation.append(continuation_rate)
    df['momentum_continuation'] = momentum_continuation
    
    # Upside momentum duration
    upside_duration = []
    current_streak = 0
    for i in range(len(df)):
        if i == 0:
            upside_duration.append(0)
            continue
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            current_streak += 1
        else:
            current_streak = 0
        upside_duration.append(current_streak)
    df['upside_duration'] = upside_duration
    
    # Volume Asymmetry and Concentration
    df['advancing_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['declining_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    
    # 5-day volume concentration ratio
    df['volume_concentration'] = (
        df['advancing_volume'].rolling(window=5, min_periods=3).sum() / 
        (df['volume'].rolling(window=5, min_periods=3).sum() + 1e-8)
    )
    
    # Volume clustering ratio (max volume / total volume in 5-day window)
    df['volume_clustering'] = (
        df['volume'].rolling(window=5, min_periods=3).max() / 
        (df['volume'].rolling(window=5, min_periods=3).sum() + 1e-8)
    )
    
    # Price-Volume Divergence Measurement
    df['volume_momentum_5d'] = df['volume'].pct_change(5)
    df['price_volume_divergence'] = (
        df['returns_5d'] - df['volume_momentum_5d'].fillna(0)
    )
    
    # Volume confirmation analysis
    df['volume_confirmation'] = np.where(
        (df['returns_5d'] > 0) & (df['volume_momentum_5d'] > 0), 1,
        np.where((df['returns_5d'] < 0) & (df['volume_momentum_5d'] < 0), -1, 0)
    )
    
    # Regime Transition Detection
    df['momentum_acceleration'] = df['returns_5d'].diff(3)
    
    # Volume-based regime shifts
    df['volume_regime_shift'] = (
        (df['volume_spike'] > 2.0) & 
        (df['volume_spike'].shift(1) <= 2.0)
    ).astype(int)
    
    # Integrated Factor Generation
    
    # Volume-weighted momentum scoring
    momentum_strength = (
        df['returns_5d'].abs() * 
        df['momentum_continuation'] * 
        (1 + df['volume_concentration'])
    )
    
    # Divergence signal integration
    divergence_signal = (
        -df['price_volume_divergence'] *  # Negative divergence suggests reversal
        df['volume_confirmation'].abs() *  # Strength of volume confirmation
        (1 - df['daily_range'])  # Adjust for liquidity conditions
    )
    
    # Regime transition weighting
    regime_weight = (
        1 + 
        df['volume_regime_shift'] * 0.5 +  # Boost during regime transitions
        df['momentum_acceleration'].abs() * 0.3  # Account for momentum changes
    )
    
    # Final factor calculation
    factor = (
        momentum_strength * 
        (1 + divergence_signal) * 
        regime_weight * 
        np.sign(df['returns_5d'])  # Preserve direction
    )
    
    # Fill NaN values and return
    result = factor.fillna(0)
    return result
