import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Reversal
    intraday_reversal = 2 * (df['high'] - df['low']) / (df['close'] + df['open'])
    
    # Adjust for Open Interest
    volume_diff = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    adjusted_intraday_reversal = intraday_reversal * (1 + volume_diff)
    
    # Calculate Intraday High-Low Difference
    intraday_high_low_diff = df['high'] - df['low']
    
    # Volume Weighted Close-Open Change
    close_open_diff = df['close'] - df['open']
    volume_weighted_close_open_change = close_open_diff * df['volume']
    
    # Combine Intraday Metrics
    combined_intraday_metrics = intraday_high_low_diff + volume_weighted_close_open_change
    
    # Moving Average Comparison
    ma_5 = combined_intraday_metrics.rolling(window=5).mean()
    moving_avg_comparison = (combined_intraday_metrics > ma_5).astype(int) * 2 - 1
    
    # Calculate Daily Price Momentum
    daily_price_momentum = df['close'] - df['close'].shift(10)
    
    # Calculate Volume Surprise
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_surprise = df['volume'] - volume_ma_10
    
    # Consider Day-to-Day Open Price Change
    open_price_change = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Intraday Midpoint
    intraday_midpoint = (df['high'] + df['low']) / 2
    
    # Calculate Close to Midpoint Deviation
    close_to_midpoint_deviation = df['close'] - intraday_midpoint
    
    # Combine Metrics
    combined_metrics = intraday_range + close_to_midpoint_deviation + open_price_change
    
    # Compute Volume Influence Ratio
    upward_volume = df[df['close'] > df['open']]['volume'].sum()
    downward_volume = df[df['close'] < df['open']]['volume'].sum()
    volume_influence_ratio = (upward_volume / downward_volume).fillna(1)
    
    # Intraday Volatility
    intraday_volatility = (df['high'] - df['low']) / df['close']
    
    # Synthesize Final Alpha Factor
    final_alpha_factor = (
        adjusted_intraday_reversal * combined_metrics +
        daily_price_momentum * volume_surprise * intraday_range +
        intraday_volatility * volume_influence_ratio
    )
    
    return final_alpha_factor
