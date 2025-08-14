import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Price Difference
    high_low_diff = df['high'] - df['low']
    
    # Compute Volume-Weighted High-Low Price Difference
    vol_weighted_high_low_diff = high_low_diff * df['volume']
    
    # Summarize Daily Volume
    total_daily_volume = df['volume'].sum()
    
    # Compute Price Change
    price_change = df['close'] - df['close'].shift(1)
    
    # Incorporate Volume Impact Factor
    volume_impact_factor = df['volume'] * abs(price_change)
    
    # Integrate Historical High-Low Range and Momentum Contributions
    last_5_days_high_low_range = high_low_diff.rolling(window=5).sum()
    momentum_contributions = (vol_weighted_high_low_diff * (df['close'] - df['close'].shift(1))).rolling(window=10).sum()
    
    # Adjust for Market Sentiment
    volatility_threshold = (df['high'] - df['low']) / df['close']
    average_volatility = volatility_threshold.rolling(window=5).mean()
    integrated_value = momentum_contributions.where(price_change > 0, 0).sum()
    sentiment_adjusted_value = integrated_value.where(integrated_value > average_volatility, integrated_value - (average_volatility - integrated_value))
    
    # Calculate Intraday Return Ratio
    intraday_high_over_low = df['high'] / df['low']
    close_over_open = df['close'] / df['open']
    
    # Evaluate Overnight Sentiment
    log_volume = np.log(df['volume'])
    overnight_return = np.log(df['open'] / df['close'].shift(1))
    
    # Integrate Intraday and Overnight Signals
    avg_intraday_return = (intraday_high_over_low + close_over_open) / 2
    intraday_overnight_signal = (avg_intraday_return - overnight_return) * (df['volume'] - df['volume'].rolling(window=10).mean())
    
    # Calculate Intraday High-Low Spread
    intraday_high_low_spread = df['high'] - df['low']
    
    # Calculate Deviation of Current Intraday High-Low Spread
    historical_avg_high_low_spread = intraday_high_low_spread.rolling(window=10).mean()
    deviation_current_high_low_spread = intraday_high_low_spread - historical_avg_high_low_spread
    
    # Calculate Volume Weighted Average Price (VWAP)
    vwap = ((df['open'] + df['high'] + df['low'] + df['close']) / 4) * df['volume']
    total_vwap = vwap.sum() / df['volume'].sum()
    
    # Calculate Deviation of VWAP from Close
    vwap_deviation = total_vwap - df['close']
    
    # Generate Alpha Factor
    high_low_range_over_volume = high_low_diff / total_daily_volume
    
    # Synthesize Overall Alpha Factor
    combined_momentum_high_low_range = sentiment_adjusted_value + intraday_overnight_signal
    volume_adjusted_reversal_potential = (deviation_current_high_low_spread + vwap_deviation) * (df['volume'] - df['volume'].rolling(window=10).mean())
    
    final_alpha_factor = combined_momentum_high_low_range + volume_adjusted_reversal_potential
    final_alpha_factor = final_alpha_factor * (df['close'] / df['close'].shift(1))
    
    return final_alpha_factor
