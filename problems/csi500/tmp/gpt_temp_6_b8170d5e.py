import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame, sector_data: pd.Series, macro_data: pd.DataFrame) -> pd.Series:
    # Adaptive momentum considering both short and long-term trends with exponential smoothing
    short_term_momentum = df['close'].pct_change(periods=5).ewm(span=5, adjust=False).mean()
    long_term_momentum = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_momentum = 0.7 * short_term_momentum + 0.3 * long_term_momentum

    # Dynamic volatility using the standard deviation of logarithmic returns over the last 14 days
    log_returns = np.log(df['close'] / df['close'].shift(1))
    dynamic_volatility = log_returns.rolling(window=14).std().ewm(span=14, adjust=False).mean()

    # Normalized volume to account for varying liquidity conditions
    normalized_volume = (df['volume'] - df['volume'].mean()) / df['volume'].std()

    # Market sentiment using the ratio of high to low prices as a proxy
    market_sentiment = (df['high'] - df['low']) / df['close']

    # Price-Volume Correlation: A measure of the relationship between price and volume
    price_volume_correlation = df['close'].pct_change().rolling(window=14).corr(df['volume'].pct_change())

    # Trend Strength: Using the difference between the highest and lowest closing price in the last 10 days
    trend_strength = df['close'].rolling(window=10).max() - df['close'].rolling(window=10).min()

    # Sector-specific indicator: Relative strength against the sector
    sector_relative_strength = df['close'].pct_change(periods=30) - sector_data.pct_change(periods=30)

    # Macroeconomic data integration: GDP growth rate
    gdp_growth_rate = macro_data['GDP_Growth_Rate']
    gdp_growth_impact = df['close'].pct_change(periods=30) * gdp_growth_rate

    # Adaptive weights based on recent performance
    recent_performance = df['close'].pct_change(periods=30).ewm(span=30, adjust=False).mean()
    adaptive_weights = {
        'adaptive_momentum': 0.3 + 0.1 * recent_performance,
        'dynamic_volatility': 0.2 - 0.1 * recent_performance,
        'normalized_volume': 0.2 + 0.05 * recent_performance,
        'market_sentiment': 0.1 + 0.05 * recent_performance,
        'price_volume_correlation': 0.1 + 0.05 * recent_performance,
        'trend_strength': 0.1 + 0.05 * recent_performance,
        'sector_relative_strength': 0.1,
        'gdp_growth_impact': 0.1
    }

    # Combining the factors into a single alpha factor
    factor_values = (
        adaptive_weights['adaptive_momentum'] * adaptive_momentum
        - adaptive_weights['dynamic_volatility'] * dynamic_volatility
        + adaptive_weights['normalized_volume'] * normalized_volume
        + adaptive_weights['market_sentiment'] * market_sentiment
        + adaptive_weights['price_volume_correlation'] * price_volume_correlation
        + adaptive_weights['trend_strength'] * trend_strength
        + adaptive_weights['sector_relative_strength'] * sector_relative_strength
        + adaptive_weights['gdp_growth_impact'] * gdp_growth_impact
    )

    return factor_values
