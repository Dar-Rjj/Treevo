import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['open']
    
    # Adjust for Volume
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    adjusted_intraday_return = (df['volume'] - volume_ma_20) * intraday_return
    
    # Incorporate Price Volatility
    close_std_10 = df['close'].rolling(window=10).std()
    volatility_adjustment = 1.5 if close_std_10 > close_std_10.mean() else 0.7
    adjusted_intraday_return *= volatility_adjustment
    
    # Calculate Daily Price Movement Range
    daily_range = df['high'] - df['low']
    
    # Calculate Volume Weighted Average Price (VWAP)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Determine Daily Return Deviation from VWAP
    deviation_from_vwap = df['close'] - vwap
    
    # Identify Trend Reversal Potential
    previous_deviation = deviation_from_vwap.shift(1)
    trend_reversal_potential = (deviation_from_vwap > previous_deviation) & (df['volume'] > df['volume'].rolling(window=14).mean())
    
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Price Breakout Ratio
    price_breakout_ratio = (df['high'] - df['open']) / high_low_range
    
    # Calculate Volume Breakout Indicator
    volume_breakout_indicator = df['volume'] * (df['close'] - df['open'])
    
    # Aggregate Breakout Indicators
    breakout_ratio_weighted = price_breakout_ratio * volume_breakout_indicator
    breakout_aggregated = breakout_ratio_weighted.cumsum()
    
    # Smooth with Exponential Moving Average
    smoothing_factor = 0.5
    smoothed_breakout = breakout_aggregated.ewm(alpha=smoothing_factor, adjust=False).mean()
    
    # Generate Final Alpha Factor
    final_alpha_factor = pd.Series(index=df.index)
    for i in range(len(df)):
        if trend_reversal_potential.iloc[i] and adjusted_intraday_return.iloc[i] > 0:
            final_alpha_factor.iloc[i] = smoothed_breakout.iloc[i] * 1.5
        else:
            final_alpha_factor.iloc[i] = smoothed_breakout.iloc[i]
    
    return final_alpha_factor
