import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Momentum Components
    short_term_return = df['close'] / df['close'].shift(5) - 1
    medium_term_return = df['close'] / df['close'].shift(20) - 1
    
    # Calculate Volume Momentum Components
    short_term_volume_change = df['volume'] / df['volume'].shift(5) - 1
    medium_term_volume_change = df['volume'] / df['volume'].shift(20) - 1
    
    # Calculate Volatility Components
    # True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Volatility Persistence (autocorrelation of true range)
    tr_lag1 = true_range.shift(1)
    tr_lag2 = true_range.shift(2)
    tr_lag3 = true_range.shift(3)
    
    # Calculate rolling autocorrelation coefficient
    volatility_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window_data = true_range.iloc[i-20:i]
        if len(window_data) >= 3:
            autocorr = window_data.autocorr(lag=1)
            volatility_persistence.iloc[i] = autocorr if not np.isnan(autocorr) else 0.0
    
    # Identify Divergence Patterns
    divergence_score = pd.Series(0.0, index=df.index)
    
    # Positive price momentum with negative volume momentum
    pos_condition = (short_term_return > 0) & (short_term_volume_change < 0)
    divergence_score[pos_condition] = 1.0
    
    # Negative price momentum with positive volume momentum
    neg_condition = (short_term_return < 0) & (short_term_volume_change > 0)
    divergence_score[neg_condition] = -1.0
    
    # Combine with Volatility Weighting
    # Apply volatility scaling
    volatility_scaled_divergence = divergence_score * true_range
    
    # Scale by volatility persistence (absolute value to maintain direction)
    volatility_persistence_scaled = volatility_persistence.fillna(0).abs()
    volatility_weighted_divergence = volatility_scaled_divergence * volatility_persistence_scaled
    
    # Incorporate Trend Consistency
    # Compare short-term and medium-term price returns
    price_trend_alignment = np.sign(short_term_return) == np.sign(medium_term_return)
    price_alignment_strength = price_trend_alignment.astype(float) * (abs(short_term_return) + abs(medium_term_return)) / 2
    
    # Compare short-term and medium-term volume changes
    volume_trend_alignment = np.sign(short_term_volume_change) == np.sign(medium_term_volume_change)
    volume_alignment_strength = volume_trend_alignment.astype(float) * (abs(short_term_volume_change) + abs(medium_term_volume_change)) / 2
    
    # Combined alignment strength
    alignment_strength = (price_alignment_strength + volume_alignment_strength) / 2
    
    # Generate Final Alpha Factor
    final_factor = volatility_weighted_divergence * alignment_strength
    
    return final_factor
