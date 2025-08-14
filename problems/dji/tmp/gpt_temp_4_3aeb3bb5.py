import pandas as pd
import pandas as pd

def heuristics(df):
    # Intraday Volatility
    intraday_volatility = df['High'] - df['Low']
    
    # Price Momentum
    price_momentum = df['Close'].pct_change(periods=5).rolling(window=5).sum()
    
    # Volume Momentum
    volume_momentum = (df['Volume'].diff(periods=5).abs()).rolling(window=5).sum()
    
    # Short-Term Return
    short_term_return = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    
    # Long-Term Return
    long_term_return = (df['Close'] - df['Close'].shift(60)) / df['Close'].shift(60)
    
    # Adjusted Momentum
    adjusted_price_momentum = price_momentum / intraday_volatility
    adjusted_volume_momentum = volume_momentum / intraday_volatility
    
    # Combined Adjusted Momentum
    combined_adjusted_momentum = (adjusted_price_momentum + adjusted_volume_momentum + df['Volume'] * short_term_return) / 3.0
    
    # Volume-Weighted Long-Term Return
    volume_weighted_long_term_return = df['Volume'] * long_term_return
    
    # Momentum Factor
    momentum_factor = combined_adjusted_momentum - volume_weighted_long_term_return
    
    # Reversal Indicator
    reversal_indicator = -momentum_factor
    
    return reversal_indicator
