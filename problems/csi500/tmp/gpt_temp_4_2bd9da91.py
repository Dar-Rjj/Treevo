import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price persistence via autocorrelation breakpoint
    returns = close.pct_change()
    autocorr_short = returns.rolling(5).apply(lambda x: x.autocorr(), raw=False)
    autocorr_long = returns.rolling(15).apply(lambda x: x.autocorr(), raw=False)
    persistence_signal = autocorr_short - autocorr_long
    
    # Volume-synchronized volatility efficiency
    dollar_volume = volume * close
    vol_weighted_volatility = (high - low).rolling(8).std() * dollar_volume.rolling(8).mean()
    market_volatility = (high - low).rolling(8).std() * amount.rolling(8).mean()
    volatility_efficiency = vol_weighted_volatility / (market_volatility + 1e-8)
    
    # Momentum regime filtering using price acceleration divergence
    price_accel_short = close.pct_change(3) - close.pct_change(6)
    price_accel_long = close.pct_change(8) - close.pct_change(15)
    regime_filter = np.arctan(price_accel_short - price_accel_long)
    
    # Combined factor
    heuristics_matrix = persistence_signal * volatility_efficiency * regime_filter
    
    return heuristics_matrix
