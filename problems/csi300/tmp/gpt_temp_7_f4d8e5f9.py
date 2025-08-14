import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily return
    daily_return = df['close'].pct_change()
    
    # Dynamic span for momentum based on the 30-day standard deviation of returns
    std_30 = daily_return.rolling(window=30).std()
    ewma_momentum_span = (std_30 * 20 + 1).astype(int)  # Ensure span is at least 1
    ewma_momentum = daily_return.ewm(span=ewma_momentum_span, adjust=False).mean()
    
    # Adaptive volatility scaling
    ewmstd_volatility = daily_return.ewm(span=ewma_momentum_span, adjust=False).std()
    volatility_scaled_momentum = ewma_momentum / (ewmstd_volatility + 1e-7)
    
    # Dynamic span for volume liquidity
    ewma_volume_span = (std_30 * 20 + 1).astype(int)  # Ensure span is at least 1
    ewma_volume = df['volume'].ewm(span=ewma_volume_span, adjust=False).mean()
    volume_scaled_momentum = 1 / (ewma_volume + 1e-7)
    
    # Combine momentum and volume scaled factors
    alpha_factor = (volatility_scaled_momentum * volume_scaled_momentum)
    
    # Incorporate a longer-term trend by using a 60-day EWMA of the close price
    ewma_close_60 = df['close'].ewm(span=60, adjust=False).mean()
    long_term_trend = df['close'] / ewma_close_60 - 1
    alpha_factor += long_term_trend
    
    # Refine risk adjustment by using a 20-day EWMA of the ratio of amount to volume
    risk_adjustment = df['amount'] / df['volume']
    ewma_risk_adjustment_20 = risk_adjustment.ewm(span=20, adjust=False).mean()
    alpha_factor *= ewma_risk_adjustment_20
    
    # Add a VWAP component to the alpha factor
    vwap = (df['amount'] / df['volume']).ewm(span=20, adjust=False).mean()
    vwap_deviation = (df['close'] - vwap) / vwap
    alpha_factor += vwap_deviation
    
    # Rank the alpha factor to avoid overfitting
    alpha_factor = alpha_factor.rank(pct=True)
    
    return alpha_factor
