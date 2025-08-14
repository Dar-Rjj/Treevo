import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the exponential moving average of close prices for a 5-day and 20-day window
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate the difference between today's close and the 5-day EMA, then divide by today's volume
    momentum_adjusted = (df['close'] - ema_5) / df['volume']
    
    # Calculate the difference between today's close and the 20-day EMA, then divide by today's volume
    long_momentum = (df['close'] - ema_20) / df['volume']
    
    # Calculate the weighted price using amount and volume to identify the money flow over a 3-day window
    money_flow = (df['amount'] / df['volume']).rolling(window=3).mean()
    
    # Calculate the ratio of the current day's range to the 5-day average range
    daily_range = df['high'] - df['low']
    avg_daily_range = daily_range.rolling(window=5).mean()
    range_ratio = daily_range / avg_daily_range
    
    # Adaptive window for volatility: use the standard deviation of the last 10 days
    volatility = df['close'].rolling(window=10).std()
    
    # Additional feature: market sentiment (e.g., percentage change in volume)
    volume_change = df['volume'].pct_change().fillna(0)
    
    # Incorporate logarithmic returns as a measure of robust, responsive signals
    log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    
    # Dynamic weights based on the adaptive volatility
    weights = 1 / (1 + volatility)
    
    # Combine all factors with a weighted sum
    alpha_factor = (
        0.3 * momentum_adjusted * weights +
        0.2 * long_momentum * weights +
        0.1 * money_flow * weights +
        0.1 * range_ratio * weights +
        0.1 * volume_change * weights +
        0.2 * log_returns * weights
    )
    
    # Introduce a new factor: the interaction between log returns and momentum adjusted
    log_momentum_interaction = log_returns * momentum_adjusted * weights
    
    # Add the new interaction term to the alpha factor
    alpha_factor += 0.1 * log_momentum_interaction
    
    # Consider seasonality: calculate the average return for each month
    monthly_seasonality = df['close'].resample('M').mean().pct_change().reindex(df.index, method='ffill').fillna(0)
    
    # Add the seasonal factor to the alpha factor
    alpha_factor += 0.05 * monthly_seasonality * weights
    
    # External indicators: use a simple moving average of an external indicator (e.g., VIX)
    vix_sma_20 = df['vix'].rolling(window=20).mean()
    vix_factor = (df['vix'] - vix_sma_20) / vix_sma_20
    alpha_factor += 0.05 * vix_factor * weights
    
    # Cross-asset correlation: calculate the correlation between the stock and a benchmark (e.g., S&P 500)
    corr_with_sp500 = df['close'].rolling(window=20).corr(df['sp500'])
    cross_asset_factor = (corr_with_sp500 - 0.5) * 2
    alpha_factor += 0.05 * cross_asset_factor * weights
    
    # Smooth the alpha factor using an exponential moving average with a span of 5
    alpha_factor_smoothed = alpha_factor.ewm(span=5, adjust=False).mean()
    
    return alpha_factor_smoothed
