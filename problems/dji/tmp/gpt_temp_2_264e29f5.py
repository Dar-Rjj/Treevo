import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_spread = df['high'] - df['low']
    
    # Compute Intraday Range Weighted Average Price (IRWAP)
    irwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
    
    # Evaluate IRWAP Difference
    irwap_diff = df['close'] - irwap
    
    # Calculate Long-Term Moving Average (LMA)
    lma = df['close'].rolling(window=200).mean()
    
    # Calculate Short-Term Moving Average (SMA)
    sma = df['close'].rolling(window=50).mean()
    
    # Calculate Difference Between LMA and SMA
    ma_diff = lma - sma
    
    # Apply Volume Weighted Filter
    volume_weighted_diff = ma_diff * df['volume']
    sum_volume_weighted_diff = volume_weighted_diff.sum()
    smoothed_vol_weighted_diff = sum_volume_weighted_diff.ewm(span=10).mean()
    
    # Adjust Daily Return by Intraday Volatility
    daily_return = df['close'].pct_change()
    adjusted_return = daily_return / intraday_spread
    
    # Smooth the Adjusted Return
    smoothed_adjusted_return = adjusted_return.rolling(window=5).mean()
    
    # Calculate Short-Term Momentum
    log_returns = np.log(df['close']) - np.log(df['close'].shift(1))
    short_term_momentum = log_returns.rolling(window=7).sum()
    
    # Calculate Long-Term Momentum
    long_term_momentum = log_returns.rolling(window=25).sum()
    
    # Combine Short and Long-Term Momentum
    combined_momentum = short_term_momentum - long_term_momentum
    
    # Introduce Price Volatility Component
    price_volatility = (df['high'] - df['low']).rolling(window=10).std()
    adjusted_combined_momentum = combined_momentum / price_volatility
    
    # Confirm with Volume Trends
    avg_volume_20 = df['volume'].rolling(window=20).mean()
    volume_trend_score = np.where(df['volume'] > avg_volume_20, 1, -1)
    
    # Introduce Open-Price Based Momentum
    open_log_returns = np.log(df['open']) - np.log(df['open'].shift(1))
    open_price_momentum = open_log_returns.rolling(window=7).sum()
    
    # Combine Open-Price Based Momentum
    open_combined_momentum = open_price_momentum - long_term_momentum
    
    # Integrate Intraday Range and Volume
    intraday_range = df['high'] - df['low']
    volume_ratio = df['volume'] / df['volume'].rolling(window=20).mean()
    integrated_intraday = intraday_range * volume_ratio
    
    # Introduce Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Final Factor Calculation
    final_factor = (
        irwap_diff + 
        smoothed_adjusted_return + 
        adjusted_combined_momentum + 
        smoothed_vol_weighted_diff + 
        open_combined_momentum + 
        integrated_intraday + 
        rsi
    )
    
    return final_factor
