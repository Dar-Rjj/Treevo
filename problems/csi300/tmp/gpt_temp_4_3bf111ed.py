import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining volatility-scaled momentum, volume-confirmed signals,
    range-efficiency measures, amount-flow momentum, and volatility-volume integration.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility-Scaled Momentum components
    mom_5 = close / close.shift(5) - 1
    mom_10 = close / close.shift(10) - 1
    daily_vol = (high - low) / close
    
    short_term_vol_scaled = mom_5 / daily_vol
    medium_term_vol_scaled = mom_10 / daily_vol
    multi_timeframe_vol_scaled = (mom_5 + mom_10) / daily_vol
    
    # Volume-Confirmed Momentum components
    volume_trend = volume / volume.shift(5)
    
    # Calculate volume persistence (count of days where volume > previous day's volume over past 5 days)
    volume_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window = volume.iloc[i-5:i+1]
        persistence_count = sum(window.iloc[j] > window.iloc[j-1] for j in range(1, len(window)))
        volume_persistence.iloc[i] = persistence_count
    
    volume_confirmed_mom = mom_5 * volume_trend
    multi_timeframe_volume_signal = (mom_5 + mom_10) * volume_trend
    persistent_volume_mom = mom_5 * volume_persistence
    
    # Range-Efficiency Signals components
    daily_range = high - low
    range_expansion = daily_range / (high.shift(1) - low.shift(1))
    
    single_day_efficiency = abs(close - close.shift(1)) / daily_range
    
    # 3-day cumulative efficiency
    cum_efficiency = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if i >= 2:
            price_change = abs(close.iloc[i] - close.iloc[i-3])
            range_sum = sum(high.iloc[i-j] - low.iloc[i-j] for j in range(3))
            cum_efficiency.iloc[i] = price_change / range_sum if range_sum > 0 else 0
    
    range_compressed_mom = mom_5 / daily_range
    efficiency_weighted_return = (close / close.shift(1) - 1) * single_day_efficiency
    
    # Efficiency persistence signal
    efficiency_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        if i >= 3:
            persistence_count = sum(
                abs(close.iloc[i-j] - close.iloc[i-j-1]) / (high.iloc[i-j] - low.iloc[i-j]) > 0.5 
                for j in range(3)
            )
            efficiency_persistence.iloc[i] = (close.iloc[i] / close.iloc[i-1] - 1) * persistence_count
    
    # Amount-Flow Momentum components
    amount_trend = amount / amount.shift(5)
    
    # Flow persistence (count of consecutive same-sign price changes)
    flow_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        if i >= 5:
            persistence_count = sum(
                np.sign(close.iloc[i-j] - close.iloc[i-j-1]) == np.sign(close.iloc[i-j-1] - close.iloc[i-j-2])
                for j in range(5)
            )
            flow_persistence.iloc[i] = persistence_count
    
    amount_confirmed_mom = mom_5 * amount_trend
    flow_persistent_trend = mom_5 * flow_persistence
    multi_timeframe_flow_signal = (mom_5 + mom_10) * amount_trend
    
    # Volatility-Volume Integration components
    volatility_ratio = daily_vol / ((high.shift(1) - low.shift(1) + high.shift(2) - low.shift(2)) / 2)
    
    vol_volume_mom = mom_5 * volume_trend / daily_vol
    multi_timeframe_vol_volume = (mom_5 + mom_10) * volume_trend / daily_vol
    persistent_vol_volume = mom_5 * volume_persistence / daily_vol
    
    # Combine all factors with equal weighting
    factors = [
        short_term_vol_scaled, medium_term_vol_scaled, multi_timeframe_vol_scaled,
        volume_confirmed_mom, multi_timeframe_volume_signal, persistent_volume_mom,
        range_compressed_mom, efficiency_weighted_return, efficiency_persistence,
        amount_confirmed_mom, flow_persistent_trend, multi_timeframe_flow_signal,
        vol_volume_mom, multi_timeframe_vol_volume, persistent_vol_volume
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        normalized_factor = (factor - factor.mean()) / factor.std()
        combined_factor += normalized_factor
    
    # Final normalization
    combined_factor = (combined_factor - combined_factor.mean()) / combined_factor.std()
    
    return combined_factor
