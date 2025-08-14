import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term Logarithmic Return
    short_term_log_return = (df['close'] / df['close'].shift(5)).apply(lambda x: 0 if x <= 0 else math.log(x))
    
    # Calculate Long-Term Logarithmic Return
    long_term_log_return = (df['close'] / df['close'].shift(20)).apply(lambda x: 0 if x <= 0 else math.log(x))
    
    # Calculate Volume-Weighted Short-Term Logarithmic Return
    volume_weighted_short_term_return = short_term_log_return * df['volume']
    
    # Calculate Volume-Weighted Long-Term Logarithmic Return
    volume_weighted_long_term_return = long_term_log_return * df['volume']
    
    # Calculate Short-Term Volatility
    tr = df[['high', 'low']].assign(close=df['close'].shift(1)).max(axis=1) - df[['high', 'low']].assign(close=df['close'].shift(1)).min(axis=1)
    atr = tr.rolling(window=5).mean()
    
    # Adjust for Volatility
    adjusted_short_term_return = (volume_weighted_short_term_return / atr) - volume_weighted_long_term_return
    
    # Determine Gain and Loss
    gain = (df['close'] - df['close'].shift(1)).apply(lambda x: x if x > 0 else 0)
    loss = (df['close'] - df['close'].shift(1)).apply(lambda x: -x if x < 0 else 0)
    
    # Aggregate Gains and Losses
    sum_gains = gain.rolling(window=14).sum()
    sum_losses = loss.rolling(window=14).sum()
    
    # Calculate Relative Strength
    rs = sum_gains / sum_losses.replace(0, 1)
    
    # Convert to ARSI
    rsi = 100 - (100 / (1 + rs))
    arsi = rsi * (df['volume'] / df['volume'].rolling(window=14).mean()) * (df['close'] / df['open'])
    
    # Calculate Intraday Price Movement Ratio
    high_low_ratio = df['high'] / df['low']
    open_close_ratio = df['open'] / df['close']
    intraday_movement_ratio = (high_low_ratio + open_close_ratio) / 2
    
    # Measure Price-Volatility Alignment
    intraday_volatility = (df['high'] - df['low']) * df['volume']
    alignment = intraday_movement_ratio - intraday_volatility
    sentiment = alignment.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Calculate Price Oscillator
    price_oscillator = short_term_log_return - long_term_log_return
    
    # Combine ARSI, Adjusted Returns, and Price Oscillator
    combined_factor = arsi * adjusted_short_term_return + price_oscillator - volume_weighted_long_term_return
    
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Compute VWMA of High-Low Spread
    vwma_high_low_spread = (high_low_spread * df['volume']).rolling(window=20).mean() / df['volume'].rolling(window=20).mean()
    
    # Incorporate Close-to-Close Return
    close_to_close_return = df['close'] - df['close'].shift(1)
    vwma_close_to_close = vwma_high_low_spread * close_to_close_return
    
    # Final Adjustment
    volume_change = (df['volume'] / df['volume'].shift(1)) - 1
    final_factor = adjusted_short_term_return + combined_factor + vwma_close_to_close * volume_change
    
    return final_factor
