import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Price Breakout Ratio
    price_breakout_ratio = (df['high'] - df['open']) / high_low_range
    
    # Calculate Volume Breakout Indicator
    volume_breakout_indicator = (df['close'] - df['open']) * df['volume']
    
    # Aggregate Breakout Indicators
    aggregated_breakout_indicators = price_breakout_ratio * df['volume']
    
    # Smooth with Moving Average (e.g., 5 days)
    window_size = 5
    smoothed_aggregated_breakout = aggregated_breakout_indicators.rolling(window=window_size).mean()
    
    # Calculate Daily Returns
    daily_returns = df['close'].diff()
    
    # Compute Momentum Score (e.g., 20 days)
    reference_period = 20
    momentum_score = df['close'] - df['close'].shift(reference_period)
    
    # Adjust for Volume Volatility
    lookback_days = 20
    volume_moving_average = df['volume'].rolling(window=lookback_days).mean()
    volume_deviation = df['volume'] - volume_moving_average
    volume_adjustment_factor = volume_deviation + 1e-6  # Small constant to avoid division by zero
    adjusted_momentum_score = momentum_score / volume_adjustment_factor
    
    # Combine Breakout and Momentum Indicators
    breakout_momentum_combined = smoothed_aggregated_breakout + adjusted_momentum_score
    
    # Calculate Volume Weight
    average_volume = df['volume'].rolling(window=lookback_days).mean()
    volume_weight = df['volume'] / average_volume
    
    # Calculate Volume Weighted Average Price (VWAP)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Determine Daily Return Deviation from VWAP
    daily_return_deviation = df['close'] - vwap
    
    # Identify Trend Reversal Potential
    trend_reversal_potential = (daily_return_deviation > daily_return_deviation.shift(1)) & (df['volume'] > df['volume'].rolling(window=14).mean())
    
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['open']
    
    # Adjust for Volume
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_adjusted_intraday_return = (df['volume'] - volume_ma_10) * intraday_return
    
    # Incorporate Price Volatility
    close_price_std = df['close'].rolling(window=5).std()
    volatility_factor = 1.2 if close_price_std > close_price_std.median() else 0.8
    adjusted_intraday_return = volume_adjusted_intraday_return * volatility_factor
    
    # Combine Indicators
    combined_indicator = trend_reversal_potential * adjusted_intraday_return
    
    # Final Alpha Factor
    final_alpha_factor = combined_indicator * volume_weight
    
    return final_alpha_factor
