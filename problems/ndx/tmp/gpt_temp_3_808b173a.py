import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Movement Ratio
    high_low_ratio = df['high'] / df['low']
    open_close_ratio = df['open'] / df['close']
    intraday_price_movement_ratio = (high_low_ratio + open_close_ratio) / 2

    # Measure Price-Volatility Alignment
    intraday_volatility = (df['high'] - df['low']) * df['volume']
    alignment = intraday_price_movement_ratio / intraday_volatility
    sentiment = np.where(alignment > 0, 1, -1)

    # Calculate Gain and Loss
    gain_loss = df['close'].diff()
    gain = np.where(gain_loss > 0, gain_loss, 0)
    loss = np.where(gain_loss < 0, -gain_loss, 0)

    # Aggregate Gains and Losses over a period (e.g., 14 days)
    sum_gains = gain.rolling(window=14).sum()
    sum_losses = loss.rolling(window=14).sum()

    # Calculate Relative Strength and ARSI
    rs = sum_gains / sum_losses
    arsi = 100 - (100 / (1 + rs))
    arsi = arsi * df['volume'].rolling(window=14).mean() * (df['close'] / df['open'])

    # Calculate Short-Term and Long-Term Logarithmic Returns
    short_term_log_return = np.log(df['close'] / df['close'].shift(5))
    long_term_log_return = np.log(df['close'] / df['close'].shift(20))

    # Calculate Volume-Weighted Logarithmic Returns
    vw_short_term_log_return = df['volume'] * short_term_log_return
    vw_long_term_log_return = df['volume'] * long_term_log_return

    # Calculate Short-Term Volatility
    short_term_volatility = df[['high', 'low']].rolling(window=5).apply(lambda x: np.mean(np.abs(x.diff().dropna())), raw=True)

    # Adjust for Volatility
    adjusted_vw_short_term_log_return = vw_short_term_log_return / short_term_volatility

    # Calculate Price Oscillator
    price_oscillator = short_term_log_return - long_term_log_return

    # Combine ARSI, Adjusted Returns, and Price Oscillator
    combined_factor = (arsi * adjusted_vw_short_term_log_return + price_oscillator - vw_long_term_log_return)

    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']

    # Compute VWMA of High-Low Spread
    vwma_high_low_spread = (high_low_spread * df['volume']).rolling(window=14).mean()

    # Incorporate Close-to-Close Return
    close_to_close_return = df['close'].diff()
    final_factor = vwma_high_low_spread * close_to_close_return + combined_factor

    return final_factor
