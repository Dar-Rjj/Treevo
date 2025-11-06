import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volatility-Normalized Momentum
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term momentum: (close[t] / close[t-5] - 1)
    momentum_5d = (close / close.shift(5)) - 1
    
    # Daily volatility proxy: (high[t] - low[t]) / close[t-1]
    volatility_proxy = (high - low) / close.shift(1)
    
    # Volatility normalization: momentum / volatility_proxy
    volatility_normalized_momentum = momentum_5d / volatility_proxy.replace(0, np.nan)
    
    # Volatility regime detection using 20-day rolling average of volatility_proxy
    volatility_20d_avg = volatility_proxy.rolling(window=20).mean()
    historical_vol_median = volatility_20d_avg.median()
    
    # Volume-Price Divergence Analysis
    volume_divergence = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        # Volume trend calculation: linear regression slope of volume[t-4:t]
        volume_window = volume.iloc[i-4:i+1]
        if len(volume_window) >= 2:
            slope, _, _, _, _ = linregress(range(len(volume_window)), volume_window)
            volume_slope = slope
        else:
            volume_slope = 0
        
        # Price momentum direction: sign(momentum)
        price_direction = np.sign(momentum_5d.iloc[i]) if not pd.isna(momentum_5d.iloc[i]) else 0
        
        # Volume trend direction: sign(volume_slope)
        volume_direction = np.sign(volume_slope)
        
        # Convergence: price and volume moving same direction
        # Divergence: price and volume moving opposite directions
        if price_direction == volume_direction and price_direction != 0:
            volume_divergence.iloc[i] = 2.0  # Convergence
        elif price_direction != volume_direction and price_direction != 0 and volume_direction != 0:
            volume_divergence.iloc[i] = 0.5  # Divergence
        else:
            volume_divergence.iloc[i] = 1.0  # Neutral
    
    # Volume adjustment: base_signal * volume_divergence
    volume_adjusted_signal = volatility_normalized_momentum * volume_divergence
    
    # Regime-Aware Signal Weighting
    regime_weight = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.isna(volatility_20d_avg.iloc[i]):
            regime_weight.iloc[i] = 1.0
            continue
            
        # High volatility regime: volatility_20d_avg > historical median
        if volatility_20d_avg.iloc[i] > historical_vol_median:
            # High vol weighting: emphasize mean reversion (negative momentum signals)
            if volume_adjusted_signal.iloc[i] > 0:
                regime_weight.iloc[i] = 0.7  # Reduce positive signals
            else:
                regime_weight.iloc[i] = 1.3  # Amplify negative signals
        else:
            # Low volatility regime: volatility_20d_avg <= historical median
            # Low vol weighting: emphasize momentum continuation (positive momentum signals)
            if volume_adjusted_signal.iloc[i] > 0:
                regime_weight.iloc[i] = 1.3  # Amplify positive signals
            else:
                regime_weight.iloc[i] = 0.7  # Reduce negative signals
    
    # Final alpha factor: regime_adjusted_signal
    regime_adjusted_signal = volume_adjusted_signal * regime_weight
    
    return regime_adjusted_signal
