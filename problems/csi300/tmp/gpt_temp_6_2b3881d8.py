import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Intraday Volatility (Average True Range over a period)
    true_range = df[['high', 'low']].diff(axis=1).iloc[:, 0].abs()
    true_range = true_range + (df['high'] - df['close'].shift(1)).abs() + (df['low'] - df['close'].shift(1)).abs()
    true_range = true_range.max(axis=1)
    intraday_volatility = true_range.rolling(window=14).mean()
    
    # Calculate Intraday Momentum
    intraday_momentum = intraday_high_low_spread - intraday_high_low_spread.shift(1)
    
    # Calculate Intraday Reversal
    previous_close_open_diff = (df['close'].shift(1) - df['open'].shift(1))
    intraday_reversal = (df['close'] - df['open']) - previous_close_open_diff * intraday_momentum
    
    # Calculate Volume Trend (Exponential Moving Average of Volume)
    volume_ema = df['volume'].ewm(span=7, adjust=False).mean()
    volume_trend = df['volume'] - volume_ema
    
    # Compute Volume Reversal Component
    volume_reversal_component = np.where(volume_trend > 0, 1, -1) * intraday_return
    
    # Introduce Intraday Gap Factor
    intraday_gap_factor = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Incorporate Transaction Amount
    amount_sma = df['amount'].rolling(window=7).mean()
    transaction_amount_component = np.where(df['amount'] > amount_sma, 1, -1) * intraday_return
    
    # Calculate Intraday Price Efficiency
    intraday_price_efficiency = (df['high'] - df['close']) / (df['high'] - df['low'])
    intraday_price_efficiency_component = (intraday_price_efficiency - intraday_price_efficiency.shift(1)) * intraday_return
    
    # Compute N-day Close Price Percent Change
    n_day_percent_change = (df['close'] / df['close'].shift(7) - 1)
    
    # Apply M-day Exponential Moving Average to N-day Percent Change
    m_day_ema_n_day_percent_change = n_day_percent_change.ewm(span=3, adjust=False).mean()
    
    # Calculate Volume-Adjusted Intraday Return
    volume_adjusted_intraday_return = (df['high'] - df['low']) / df['low'] / df['volume']
    
    # Combine All Components
    factor = (intraday_return * intraday_range * intraday_volatility +
              intraday_reversal + volume_reversal_component + 
              intraday_gap_factor + transaction_amount_component + 
              intraday_price_efficiency_component + 
              m_day_ema_n_day_percent_change * volume_adjusted_intraday_return)
    
    # Set to zero if the absolute value is below a threshold (e.g., 0.5)
    factor = np.where(np.abs(factor) < 0.5, 0, factor)
    
    return factor
