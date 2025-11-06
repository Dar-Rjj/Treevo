import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration with Volume Confirmation
    # Calculate 5-day and 20-day momentum from Close
    mom_5 = data['close'] / data['close'].shift(5) - 1
    mom_20 = data['close'] / data['close'].shift(20) - 1
    
    # Compute acceleration as momentum difference
    momentum_acceleration = mom_5 - mom_20
    
    # Multiply by volume trend direction (5-day volume change)
    volume_trend = data['volume'] / data['volume'].shift(5) - 1
    factor1 = momentum_acceleration * np.sign(volume_trend)
    
    # Volatility-Regime Price Reversal
    # Calculate 10-day volatility from High-Low
    daily_range = (data['high'] - data['low']) / data['close']
    volatility_10 = daily_range.rolling(window=10, min_periods=5).std()
    
    # Identify price extremes using Close (5-day z-score)
    price_zscore = (data['close'] - data['close'].rolling(window=5).mean()) / data['close'].rolling(window=5).std()
    reversal_signal = -price_zscore  # Negative for mean reversion
    
    # Weight reversal signals by volatility
    factor2 = reversal_signal * volatility_10
    
    # Liquidity-Adjusted Trend Persistence
    # Compute 15-day price trend from Close
    price_trend = data['close'] / data['close'].shift(15) - 1
    
    # Multiply by volume-to-amount liquidity ratio (normalized)
    liquidity_ratio = data['volume'] / (data['amount'] + 1e-8)
    liquidity_ratio_norm = (liquidity_ratio - liquidity_ratio.rolling(window=20).mean()) / liquidity_ratio.rolling(window=20).std()
    factor3 = price_trend * liquidity_ratio_norm
    
    # Asymmetric Volume-Price Divergence
    # Calculate volume-price correlations for up/down days
    returns = data['close'].pct_change()
    volume_change = data['volume'].pct_change()
    
    # Separate up and down days
    up_days = returns > 0
    down_days = returns < 0
    
    # Rolling correlation for up days (10-day window)
    up_corr = pd.Series(index=data.index, dtype=float)
    for i in range(10, len(data)):
        window_data = data.iloc[i-9:i+1]
        window_up = up_days.iloc[i-9:i+1]
        if window_up.sum() >= 3:
            up_corr.iloc[i] = window_data.loc[window_up, 'close'].pct_change().corr(
                window_data.loc[window_up, 'volume'].pct_change()
            )
    
    # Rolling correlation for down days (10-day window)
    down_corr = pd.Series(index=data.index, dtype=float)
    for i in range(10, len(data)):
        window_data = data.iloc[i-9:i+1]
        window_down = down_days.iloc[i-9:i+1]
        if window_down.sum() >= 3:
            down_corr.iloc[i] = window_data.loc[window_down, 'close'].pct_change().corr(
                window_data.loc[window_down, 'volume'].pct_change()
            )
    
    # Measure divergence persistence (difference in correlations)
    corr_divergence = up_corr - down_corr
    factor4 = corr_divergence.rolling(window=5).mean()
    
    # Intraday Pressure Accumulation
    # Calculate opening gap (Open vs previous Close)
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Multiply by intraday momentum (Close vs Open)
    intraday_momentum = (data['close'] - data['open']) / data['open']
    factor5 = opening_gap * intraday_momentum
    
    # Combine all factors with equal weights
    combined_factor = (
        factor1.fillna(0) + 
        factor2.fillna(0) + 
        factor3.fillna(0) + 
        factor4.fillna(0) + 
        factor5.fillna(0)
    )
    
    # Normalize the final factor
    final_factor = (combined_factor - combined_factor.rolling(window=20).mean()) / combined_factor.rolling(window=20).std()
    
    return final_factor
