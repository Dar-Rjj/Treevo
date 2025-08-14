import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the price momentum as the change in closing price over the last 10 days
    price_momentum = df['close'].pct_change(periods=10)
    
    # Calculate the money flow ratio as the average of (high - low) * volume over the last 5 days
    # This aims to capture the strength of money inflow or outflow over recent trading days
    money_flow_ratio = ((df['high'] - df['low']) * df['volume']).rolling(window=5).mean()
    
    # Calculate a simple oscillator as the difference between 5-day and 10-day moving averages
    # This may highlight reversals or confirm trends
    short_ma = df['close'].rolling(window=5).mean()
    long_ma = df['close'].rolling(window=10).mean()
    simple_oscillator = short_ma - long_ma
    
    # Calculate the relative strength index (RSI) with a 14-day window
    # This helps identify overbought or oversold conditions
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Incorporate volatility using the standard deviation of the closing prices over the last 20 days
    volatility = df['close'].rolling(window=20).std()
    
    # Use adaptive windows for the momentum calculation
    adaptive_momentum = df['close'].pct_change(periods=volatility.rolling(window=20).mean().astype(int))
    
    # Consider market microstructure by calculating the high-low spread
    high_low_spread = df['high'] - df['low']
    
    # Combine the factors into a single alpha factor
    # Weighted sum based on heuristic importance assigned to each component
    factor_values = (
        0.3 * price_momentum + 
        0.2 * money_flow_ratio + 
        0.2 * simple_oscillator + 
        0.1 * rsi + 
        0.1 * volatility + 
        0.1 * adaptive_momentum
    )
    
    return factor_values
