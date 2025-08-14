import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Adjust Intraday High-Low Spread by Volume
    volume_ema_span = 10  # Set EMA span
    volume_ema = df['volume'].ewm(span=volume_ema_span, adjust=False).mean()
    adjusted_volume = df['volume'] / volume_ema
    adjusted_intraday_spread = intraday_high_low_spread * adjusted_volume
    
    # Further Adjustment by Open Price Volatility
    open_price_volatility = df['open'].pct_change().std() * np.sqrt(252)
    adjusted_intraday_spread_volatility = adjusted_intraday_spread / open_price_volatility
    
    # Calculate Short-Term and Long-Term Volume-Weighted Return
    short_term_window = 5
    long_term_window = 20
    returns = df['close'].pct_change()
    short_term_volume_weighted_return = (returns * df['volume']).rolling(window=short_term_window).mean()
    long_term_volume_weighted_return = (returns * df['volume']).rolling(window=long_term_window).mean()
    
    # Calculate Open-Close Range
    open_close_range = df['close'] - df['open']
    
    # Aggregate Intraday Volatility
    intraday_volatility = (adjusted_intraday_spread + open_close_range) * df['volume']
    
    # Calculate Relative Strength Indicator (RSI)
    rsi_window = 14
    gain = returns.where(returns > 0, 0)
    loss = -returns.where(returns < 0, 0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Price Reversal Indicator
    local_highs = df['high'].rolling(window=rsi_window, center=True).max()
    local_lows = df['low'].rolling(window=rsi_window, center=True).min()
    time_since_local_high = (df.index.to_series() - df.loc[df['high'] == local_highs, :].index.to_series()).dt.days
    time_since_local_low = (df.index.to_series() - df.loc[df['low'] == local_lows, :].index.to_series()).dt.days
    reversal_indicator = (time_since_local_high - time_since_local_low) / (rsi_window * 2)
    
    # Combine Factors for Final Alpha
    short_minus_long_term = short_term_volume_weighted_return - long_term_volume_weighted_return
    intraday_volatility_ma = intraday_volatility.rolling(window=short_term_window).mean()
    adjusted_rsi = rsi - rsi.rolling(window=short_term_window).mean()
    alpha_factor = short_minus_long_term + intraday_volatility_ma + adjusted_rsi + reversal_indicator
    
    return alpha_factor
