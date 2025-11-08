import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return Pattern
    intraday_return = (df['close'] - df['open']) / df['open']
    trading_range = df['high'] - df['low']
    normalized_return = intraday_return / (trading_range / df['open'])
    
    # Apply Volume-Based Persistence Filter
    # Calculate volume-weighted rolling average of intraday return pattern
    volume_weighted_return = (normalized_return * df['volume']).rolling(window=5, min_periods=3).sum() / df['volume'].rolling(window=5, min_periods=3).sum()
    
    # Compute persistence measure - count consecutive same-sign periods
    sign_changes = np.sign(volume_weighted_return).diff().fillna(0)
    persistence = sign_changes.groupby((sign_changes != 0).cumsum()).cumcount() + 1
    persistence = persistence * np.sign(volume_weighted_return)
    
    # Weight by volume trend consistency
    volume_trend = df['volume'].rolling(window=5, min_periods=3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0, raw=False)
    persistence_weighted = persistence * volume_trend.fillna(0)
    
    # Incorporate Asymmetric Volatility Weighting
    intraday_range = df['high'] - df['low']
    prev_range = intraday_range.shift(1)
    range_ratio = intraday_range / prev_range
    
    # Calculate Close-to-Open Gap
    close_open_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_direction = np.sign(close_open_gap)
    
    # Apply Asymmetric Adjustment
    volatility_adjustment = np.where(
        (range_ratio > 1) & (gap_direction == np.sign(persistence_weighted)),
        range_ratio,  # Amplify for expanding volatility with directional gaps
        np.where(
            (range_ratio < 1) & (gap_direction != np.sign(persistence_weighted)),
            1 / range_ratio,  # Dampen for contracting volatility with opposing gaps
            1.0  # Neutral adjustment
        )
    )
    
    # Final factor calculation
    factor = persistence_weighted * volatility_adjustment
    
    return factor
