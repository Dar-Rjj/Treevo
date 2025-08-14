import pandas as pd
def heuristics_v2(df: pd.DataFrame, sector_ratios: pd.DataFrame, macro_indicators: pd.DataFrame) -> pd.Series:
    # Calculate the daily return
    daily_return = df['close'].pct_change()
    
    # Calculate the 5-day, 20-day, and 60-day Exponentially Weighted Moving Average (EWMA) of the returns for momentum
    ewma_momentum_5 = daily_return.ewm(span=5, adjust=False).mean()
    ewma_momentum_20 = daily_return.ewm(span=20, adjust=False).mean()
    ewma_momentum_60 = daily_return.ewm(span=60, adjust=False).mean()
    
    # Calculate the 10-day and 30-day Exponentially Weighted Moving Standard Deviation (EWMSTD) of the returns for volatility
    ewmstd_volatility_10 = daily_return.ewm(span=10, adjust=False).std()
    ewmstd_volatility_30 = daily_return.ewm(span=30, adjust=False).std()
    
    # Calculate the 5-day and 20-day Exponentially Weighted Moving Average (EWMA) of the volume for liquidity
    ewma_volume_5 = df['volume'].ewm(span=5, adjust=False).mean()
    ewma_volume_20 = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Calculate cumulative sum of the daily returns for market sentiment
    cumulative_return = daily_return.cumsum()
    
    # Simplify the factor to combine momentum, volatility, and liquidity
    alpha_factor = (ewma_momentum_5 / (ewmstd_volatility_10 + 1e-7)) * (1 / (ewma_volume_5 + 1e-7)) \
                  + (ewma_momentum_20 / (ewmstd_volatility_30 + 1e-7)) * (1 / (ewma_volume_20 + 1e-7)) \
                  + cumulative_return
    
    # Incorporate a longer-term trend by using a 60-day EWMA of the close price
    ewma_close_60 = df['close'].ewm(span=60, adjust=False).mean()
    long_term_trend = df['close'] / ewma_close_60 - 1
    alpha_factor += long_term_trend
    
    # Refine risk adjustment by using a 20-day EWMA of the ratio of amount to volume
    risk_adjustment = df['amount'] / df['volume']
    ewma_risk_adjustment_20 = risk_adjustment.ewm(span=20, adjust=False).mean()
    alpha_factor *= ewma_risk_adjustment_20
    
    # Integrate VWAP (Volume-Weighted Average Price)
    vwap = (df['amount'] / df['volume']).rolling(window=20).mean()
    vwap_deviation = (df['close'] - vwap) / vwap
    alpha_factor += vwap_deviation
    
    # Add a term for high-low range as a measure of intraday volatility
    high_low_range = df['high'] - df['low']
    high_low_ratio = high_low_range / df['close']
    alpha_factor -= high_low_ratio
    
    # Incorporate sector-specific ratios
    sector_alpha = (sector_ratios['price_to_earnings'] - sector_ratios['price_to_book']) / sector_ratios['dividend_yield']
    alpha_factor += sector_alpha
    
    # Incorporate macroeconomic indicators
    macro_alpha = (macro_indicators['gdp_growth_rate'] - macro_indicators['inflation_rate']) / (macro_indicators['unemployment_rate'] + 1e-7)
    alpha_factor += macro_alpha
    
    # Rank the alpha factor to avoid overfitting
    alpha_factor = alpha_factor.rank(pct=True)
    
    return alpha_factor
