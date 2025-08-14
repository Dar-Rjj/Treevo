import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    combined_factor_ema = combined_factor.ewm(span=14, adjust=False).mean()
    
    # Apply Volume Weighting
    volume_weighted_factor = combined_factor_ema * df['volume']
    
    # Incorporate Previous Day's Closing Gap
    prev_day_close = df['close'].shift(1)
    closing_gap = df['open'] - prev_day_close
    volume_weighted_smoothed_factor = volume_weighted_factor + closing_gap
    
    # Integrate Long-Term Momentum
    long_term_return = df['close'] - df['close'].shift(50)
    normalized_long_term_return = long_term_return / high_low_range
    
    # Include Enhanced Dynamic Volatility Component
    intraday_returns = (df['close'] - df['open']) / df['open']
    rolling_std = intraday_returns.rolling(window=20).std()
    atr = df[['high', 'low', 'close']].apply(lambda x: np.max(x) - np.min(x), axis=1).rolling(window=14).mean()
    combined_volatility = (rolling_std + atr) / 2
    volume_adjusted_volatility = combined_volatility * df['volume']
    
    # Incorporate Market Breadth
    positive_returns = (df['close'] > df['close'].shift(1)).astype(int)
    negative_returns = (df['close'] < df['close'].shift(1)).astype(int)
    market_breadth = positive_returns.rolling(window=1).sum() - negative_returns.rolling(window=1).sum()
    
    # Incorporate Sector-Specific Performance
    # Assuming sector information is available in a column named 'sector'
    sector_average_return = df.groupby('sector')['close'].pct_change().groupby(df['sector']).mean()
    sector_volatility = df.groupby('sector')['close'].pct_change().groupby(df['sector']).rolling(window=20).std().reset_index(level=0, drop=True)
    # Assuming sector sentiment scores are available in a column named 'sector_sentiment'
    sector_sentiment = df.groupby('sector')['sector_sentiment'].rolling(window=7).mean().reset_index(level=0, drop=True)
    combined_sector_components = sector_average_return + sector_volatility + sector_sentiment
    
    # Final Factor Calculation
    final_factor = (
        volume_weighted_smoothed_factor +
        normalized_long_term_return +
        volume_adjusted_volatility +
        market_breadth +
        combined_sector_components
    )
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor)
    
    return final_factor
