import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the price momentum as the change in closing price over the last 10 days
    price_momentum = df['close'].pct_change(periods=10)
    
    # Calculate the money flow ratio as the average of (high - low) * volume over the last 5 days
    # This aims to capture the strength of money inflow or outflow over recent trading days
    money_flow_ratio = ((df['high'] - df['low']) * df['volume']).rolling(window=5).mean()
    
    # Calculate the volatility as the standard deviation of the closing price over the last 20 days
    volatility = df['close'].rolling(window=20).std()
    
    # Adaptive window for moving average based on the volatility
    adaptive_window = (volatility / df['close'].rolling(window=20).mean()).fillna(20).astype(int).clip(lower=5, upper=40)
    adaptive_moving_average = df['close'].rolling(window=adaptive_window).mean()
    
    # Incorporate cross-asset correlations (assuming a correlation matrix is available, otherwise, use self-correlation)
    # For simplicity, we will use the rolling correlation with the 10-day delayed close price
    cross_asset_correlation = df['close'].rolling(window=20).corr(df['close'].shift(10))
    
    # Combine the factors into a single alpha factor
    # Weighted sum based on heuristic importance assigned to each component
    factor_values = 0.4 * price_momentum + 0.3 * money_flow_ratio + 0.2 * (df['close'] - adaptive_moving_average) + 0.1 * cross_asset_correlation

    return factor_values
