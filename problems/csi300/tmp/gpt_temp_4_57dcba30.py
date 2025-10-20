import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Decaying Momentum Factor
    Combines short-term (5-day) and medium-term (20-day) momentum with volume confirmation
    and volatility scaling for risk-adjusted signals.
    """
    close = df['close']
    volume = df['volume']
    
    # Calculate daily returns
    returns = close.pct_change()
    
    # Short-term momentum component (5-day)
    short_window = 5
    short_weights = np.array([0.8 ** i for i in range(short_window)][::-1])
    short_weights = short_weights / short_weights.sum()
    
    short_momentum = pd.Series(index=close.index, dtype=float)
    for i in range(short_window, len(close)):
        window_returns = returns.iloc[i-short_window+1:i+1].values
        if len(window_returns) == short_window:
            short_momentum.iloc[i] = np.dot(window_returns, short_weights)
    
    # Medium-term momentum component (20-day)
    medium_window = 20
    medium_weights = np.array([0.9 ** i for i in range(medium_window)][::-1])
    medium_weights = medium_weights / medium_weights.sum()
    
    medium_momentum = pd.Series(index=close.index, dtype=float)
    for i in range(medium_window, len(close)):
        window_returns = returns.iloc[i-medium_window+1:i+1].values
        if len(window_returns) == medium_window:
            medium_momentum.iloc[i] = np.dot(window_returns, medium_weights)
    
    # Volume confirmation
    # Calculate volume percentiles (5-day rolling)
    volume_pct_5d = volume.rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    # Volume trend consistency
    volume_ma_5d = volume.rolling(window=5).mean()
    volume_ma_20d = volume.rolling(window=20).mean()
    volume_trend_ratio = volume_ma_5d / volume_ma_20d
    
    # Volume confirmation score
    volume_confirmation = volume_pct_5d * np.tanh(volume_trend_ratio - 1)
    
    # Volatility scaling
    volatility_20d = returns.rolling(window=20).std()
    
    # Signal integration
    # Combine short and medium-term momentum with their decay weights
    combined_momentum = (0.6 * short_momentum + 0.4 * medium_momentum)
    
    # Apply volume confirmation
    volume_adjusted_momentum = combined_momentum * (1 + volume_confirmation)
    
    # Final volatility scaling
    volatility_scaling = 1 / (1 + volatility_20d)
    final_factor = volume_adjusted_momentum * volatility_scaling
    
    return final_factor
