import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close to Midpoint Deviation
    midpoint_deviation = df['close'] - (df['high'] + df['low']) / 2
    
    # Calculate Adjusted Intraday Reversal with Momentum Adjustment
    intraday_reversal = 2 * (df['high'] - df['low']) / (df['close'] + df['open'])
    momentum_adjustment = 1 + (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    adjusted_intraday_reversal = intraday_reversal * momentum_adjustment
    
    # Calculate Previous Day Return
    previous_day_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Generate Intermediate Alpha Factor 1
    intermediate_alpha_factor_1 = (intraday_range * midpoint_deviation) - previous_day_return
    
    # Calculate Daily Volume Surprise
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    daily_volume_surprise = df['volume'] - volume_ma_5
    
    # Combine Price Momentum and Volume Surprise
    price_momentum = (df['close'] - df['close'].shift(5))
    combined_price_vol_momentum = price_momentum * daily_volume_surprise
    
    # Calculate High-Low Price Difference
    high_low_diff = df['high'] - df['low']
    
    # Compute Volume Influence Ratio
    upward_volume = df.loc[df['close'] > df['open'], 'volume'].sum()
    downward_volume = df.loc[df['close'] < df['open'], 'volume'].sum()
    volume_influence_ratio = upward_volume / (downward_volume + 1e-6)  # Adding small value to avoid division by zero
    
    # Adjust Momentum by Volume and Amount
    weighted_high_low_return = high_low_diff * df['volume'] * volume_influence_ratio
    volume_adjusted_momentum = weighted_high_low_return.sum()
    
    # Introduce Time Decay
    decay_rate = 0.95
    volume_adjusted_momentum *= decay_rate ** (df.index.to_series().diff().dt.days.fillna(0).cumsum())
    
    # Intermediate Alpha Factor Synthesis 1
    intermediate_alpha_factor_synthesis_1 = combined_price_vol_momentum * high_low_diff
    
    # Compute Price Momentum
    lookback_period = 5
    price_momentum_smoothed = (df['close'] - df['close'].shift(lookback_period)).ewm(span=lookback_period).mean()
    
    # Adjust by Volume Change
    volume_ratio = (df['volume'] / df['volume'].shift(lookback_period)).rolling(window=lookback_period).mean()
    combined_momentum_vol_ratio = price_momentum_smoothed * volume_ratio
    
    # Compute Intraday Close-Open Return
    close_open_return = (df['close'] - df['open']) / df['open']
    
    # Combine High-Low Spread and Close-Open Return
    intraday_momentum = high_low_diff * close_open_return
    
    # Consider Day-to-Day Open Price Change
    open_price_change = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Adjust for Open Interest
    adjusted_intraday_reversal_with_open_interest = adjusted_intraday_reversal * (1 + (df['volume'] - df['volume'].shift(5)) / (df['volume'].shift(5) + 1e-6))
    
    # Intermediate Alpha Factor Synthesis 2
    intermediate_alpha_factor_synthesis_2 = combined_momentum_vol_ratio * high_low_diff * close_open_return * (1 + (df['volume'] - df['volume'].shift(5)) / (df['volume'].shift(5) + 1e-6))
    
    # Cumulative Directional Volume Impact
    cumulative_directional_volume_impact = high_low_diff * volume_influence_ratio
    
    # Final Alpha Factor
    final_alpha_factor = (intermediate_alpha_factor_1 + intermediate_alpha_factor_synthesis_1 + intermediate_alpha_factor_synthesis_2) * cumulative_directional_volume_impact
    
    return final_alpha_factor
