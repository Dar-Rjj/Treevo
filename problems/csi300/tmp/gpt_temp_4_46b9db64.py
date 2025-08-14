import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def adaptive_ema(series, span=14):
    volatility = series.rolling(window=20).std()
    alpha = 2 / (span + 1 + volatility)
    alpha = alpha.ffill().fillna(0.0)
    ema = series.ewm(alpha=alpha).mean()
    return ema

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    smoothed_factor = adaptive_ema(combined_factor, span=14)
    
    # Apply Volume Weighting
    volume_weighted_smoothed_factor = smoothed_factor * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    previous_day_close_gap = df['open'].shift(-1) - df['close'].shift(-1)
    volume_weighted_smoothed_factor_gap = volume_weighted_smoothed_factor + previous_day_close_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Sector-Specific Momentum (Assuming sector prices are available in the DataFrame)
    if 'sector_open' in df.columns and 'sector_close' in df.columns and 'sector_high' in df.columns and 'sector_low' in df.columns:
        sector_intraday_return = df['sector_close'] - df['sector_open']
        sector_long_term_return = df['sector_close'] - df['sector_close'].shift(50)
        sector_high_low_range = df['sector_high'] - df['sector_low']
        normalized_sector_long_term_return = sector_long_term_return / sector_high_low_range
    else:
        normalized_sector_long_term_return = pd.Series(0, index=df.index)
    
    # Include Enhanced Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    atr = df['high'] - df['low'].shift(1)
    atr = atr.abs().rolling(window=14).mean()
    combined_volatility = (rolling_std + atr) / 2
    volume_adjusted_volatility = combined_volatility * df['volume']
    
    # Integrate Market Sentiment (Assuming market sentiment is available in the DataFrame)
    if 'market_sentiment' in df.columns:
        market_sentiment_volatility = df['market_sentiment'] * volume_adjusted_volatility
    else:
        market_sentiment_volatility = pd.Series(0, index=df.index)
    
    # Final Factor Calculation
    final_factor = (
        volume_weighted_smoothed_factor_gap + 
        normalized_long_term_return + 
        normalized_sector_long_term_return + 
        market_sentiment_volatility
    )
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor)
    
    return final_factor
