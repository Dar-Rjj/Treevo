import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def adaptive_ema(data, period, smoothing=2):
    return data.ewm(span=period, adjust=False).mean()

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    
    # Smooth using Adaptive Exponential Moving Average (AEMA)
    recent_volatility = intraday_return.rolling(window=14).std()
    aema_period = 14 / (1 + recent_volatility)
    smoothed_factor = adaptive_ema(combined_factor, aema_period)
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    prev_day_close_gap = df['open'].shift(-1) - df['close']
    volume_weighted_factor += prev_day_close_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    true_range = df['high'] - df['low']
    atr = true_range.rolling(window=14).mean()
    combined_volatility = (rolling_std + atr) / 2
    
    # Adjust Volatility Component with Volume
    adjusted_volatility = combined_volatility * df['volume']
    
    # Incorporate Market Regime
    long_term_trend = df['close'].pct_change(50).rolling(window=50).mean()
    short_term_momentum = df['close'].pct_change(5).rolling(window=5).mean()
    
    market_regime = pd.Series(index=df.index, dtype=float)
    market_regime[(long_term_trend > 0) & (short_term_momentum > 0)] = 1  # Bull
    market_regime[(long_term_trend < 0) & (short_term_momentum < 0)] = -1  # Bear
    market_regime[(long_term_trend > 0) & (short_term_momentum < 0) | (long_term_trend < 0) & (short_term_momentum > 0)] = 0  # Sideways
    
    # Final Factor Calculation
    final_factor = (
        volume_weighted_factor + 
        prev_day_close_gap + 
        normalized_long_term_return + 
        adjusted_volatility
    )
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor)
    
    # Adjust Final Factor Based on Market Regime
    final_factor[market_regime == 1] *= 1.2  # More weight in Bull markets
    final_factor[market_regime == -1] *= 0.8  # Less weight in Bear markets
    final_factor[market_regime == 0] *= 1.0  # No change in Sideways markets
    
    return final_factor
