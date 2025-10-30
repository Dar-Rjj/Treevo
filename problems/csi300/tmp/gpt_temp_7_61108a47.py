import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining momentum decay-adjusted volume profile with volatility adjustment.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Momentum Decay Factor
    # Compute recent price momentum with exponential decay weights
    close_prices = data['close']
    
    # Calculate momentum over 5-day period
    momentum_5d = close_prices.pct_change(periods=5)
    
    # Apply exponential decay with half-life of 3 days
    decay_weights = np.exp(-np.arange(5) * np.log(2) / 3)
    decay_weights = decay_weights / decay_weights.sum()  # Normalize
    
    # Create momentum decay factor using rolling window
    momentum_decay = close_prices.rolling(window=5).apply(
        lambda x: np.sum(x.pct_change().dropna() * decay_weights[:len(x.pct_change().dropna())])
        if len(x.pct_change().dropna()) > 0 else 0
    )
    
    # 2. Analyze Volume Profile
    volume_data = data['volume']
    
    # Calculate volume distribution over 20-day window
    volume_percentiles = volume_data.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan
    )
    
    # Identify high volume periods (above 80th percentile)
    high_volume_flag = (volume_percentiles > 0.8).astype(int)
    
    # Count consecutive high volume days
    consecutive_high_volume = high_volume_flag * (high_volume_flag.groupby((high_volume_flag != high_volume_flag.shift(1)).cumsum()).cumcount() + 1)
    
    # Create volume profile score
    volume_profile = consecutive_high_volume * volume_percentiles
    
    # 3. Combine Signals
    # Multiply momentum decay by volume profile
    combined_signal = momentum_decay * volume_profile
    
    # Adjust for recent volatility (10-day price range)
    high_prices = data['high']
    low_prices = data['low']
    
    # Calculate 10-day price range volatility
    price_range = (high_prices.rolling(window=10).max() - low_prices.rolling(window=10).min()) / close_prices.rolling(window=10).mean()
    
    # Avoid division by zero
    volatility_adjusted = combined_signal / (price_range.replace(0, np.nan).fillna(method='ffill') + 1e-8)
    
    # Generate final alpha factor
    alpha_factor = volatility_adjusted
    
    # Clean and return the factor
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return alpha_factor
