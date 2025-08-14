import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Reversal
    intraday_range = df['high'] - df['low']
    intraday_move = df['close'] - df['open']
    
    # Weight by Trade Intensity and Amount
    vwap = (df['amount'] / df['volume']).fillna(0)
    average_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    trade_intensity = vwap / average_price
    
    trade_amount_ratio = df['amount'].sum() / df['volume'].sum()
    trade_amount_ratio = trade_amount_ratio.diff().fillna(0)
    
    weighted_intraday_reversal = intraday_move * trade_intensity * trade_amount_ratio
    
    # Calculate Daily Volatility
    daily_volatility = df['high'] - df['low']
    
    # Calculate Volume Change Ratio
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_change_ratio = df['volume'] / volume_ma_20
    
    # Calculate Weighted Average Price
    weighted_average_price = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    weighted_average_price = weighted_average_price * volume_change_ratio
    
    # Calculate Price Momentum
    n_days = 5
    price_momentum = df['close'].pct_change(periods=n_days)
    
    # Identify Volume Trends
    m_days = 20
    avg_volume = df['volume'].rolling(window=m_days).mean()
    volume_trend = (df['volume'] > avg_volume).astype(int)
    adjusted_momentum = price_momentum + volume_trend * 0.01  # Add a fixed value if volume trend is positive
    
    # Calculate Smoothed Momentum
    smoothed_momentum = adjusted_momentum.ewm(span=5, adjust=False).mean()
    
    # Calculate Final Factor
    final_factor = smoothed_momentum * daily_volatility
    
    return final_factor
