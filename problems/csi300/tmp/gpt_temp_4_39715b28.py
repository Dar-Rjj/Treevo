import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Parameters
    N = 20  # Lookback period for momentum
    M = 10  # Lookback period for volume average
    top_pct = 0.2  # Top 20% by momentum
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate N-day momentum
    momentum = (df['close'] - df['close'].shift(N)) / df['close'].shift(N)
    
    # Calculate M-day average volume
    avg_volume = df['volume'].rolling(window=M, min_periods=1).mean()
    
    # Calculate volume surge (current volume relative to average)
    volume_surge = df['volume'] / avg_volume - 1
    
    # Rank stocks by momentum and identify top performers
    momentum_rank = momentum.rank(pct=True)
    high_momentum = momentum_rank > (1 - top_pct)
    
    # Calculate correlation between momentum and volume surge using rolling window
    # Use shorter window for correlation to capture recent relationship
    corr_window = min(5, M)
    momentum_volume_corr = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= corr_window - 1:
            window_momentum = momentum.iloc[i-corr_window+1:i+1]
            window_volume_surge = volume_surge.iloc[i-corr_window+1:i+1]
            if len(window_momentum) >= 2 and len(window_volume_surge) >= 2:
                corr = window_momentum.corr(window_volume_surge)
                momentum_volume_corr.iloc[i] = corr if not np.isnan(corr) else 0
            else:
                momentum_volume_corr.iloc[i] = 0
        else:
            momentum_volume_corr.iloc[i] = 0
    
    # Identify negative correlation (divergence)
    negative_corr = momentum_volume_corr < 0
    
    # Calculate strength of divergence
    # Higher momentum with lower volume surge indicates stronger reversal signal
    divergence_strength = -momentum * (1 - volume_surge.clip(lower=0))
    
    # Generate reversal signal
    # High momentum + negative correlation = potential reversal
    reversal_signal = high_momentum & negative_corr
    
    # Weight by strength of divergence
    factor_values = reversal_signal.astype(float) * divergence_strength
    
    # Handle NaN values
    factor_values = factor_values.fillna(0)
    
    return factor_values
