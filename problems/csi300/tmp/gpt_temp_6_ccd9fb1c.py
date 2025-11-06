import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        # Current day data
        open_t = df.iloc[i]['open']
        high_t = df.iloc[i]['high']
        low_t = df.iloc[i]['low']
        close_t = df.iloc[i]['close']
        volume_t = df.iloc[i]['volume']
        
        # Previous day data
        open_t1 = df.iloc[i-1]['open']
        high_t1 = df.iloc[i-1]['high']
        low_t1 = df.iloc[i-1]['low']
        close_t1 = df.iloc[i-1]['close']
        volume_t1 = df.iloc[i-1]['volume']
        
        # Two days ago data
        high_t2 = df.iloc[i-2]['high']
        low_t2 = df.iloc[i-2]['low']
        
        # Avoid division by zero
        range_t = high_t - low_t if high_t != low_t else 1e-8
        range_t1 = high_t1 - low_t1 if high_t1 != low_t1 else 1e-8
        range_t2 = high_t2 - low_t2 if high_t2 != low_t2 else 1e-8
        
        # Order Flow Imbalance
        price_impact = ((close_t - open_t) / range_t) * (volume_t / volume_t1 if volume_t1 != 0 else 1)
        bid_ask_pressure = ((close_t - low_t) / range_t) - ((high_t - close_t) / range_t)
        volume_imbalance = ((volume_t - volume_t1) / range_t) * (close_t - open_t)
        
        microstructure_score = price_impact * bid_ask_pressure * volume_imbalance
        
        # Price Rejection Patterns
        upper_shadow_rejection = ((high_t - close_t) / range_t) * ((high_t - open_t) / range_t)
        lower_shadow_support = ((close_t - low_t) / range_t) * ((open_t - low_t) / range_t)
        gap_filling = ((open_t - close_t1) / range_t1) * ((close_t - open_t) / range_t)
        
        rejection_strength = upper_shadow_rejection * lower_shadow_support * gap_filling
        
        # Volatility Clustering
        volatility_momentum = (range_t / range_t1) * (range_t1 / range_t2)
        
        # 5-day lookback for range expansion
        if i >= 5:
            high_5d = max(df.iloc[i-5:i]['high'])
            low_5d = min(df.iloc[i-5:i]['low'])
            volume_t5 = df.iloc[i-5]['volume']
            range_5d = high_5d - low_5d if high_5d != low_5d else 1e-8
            range_expansion = (range_t / range_5d) * (volume_t / volume_t5 if volume_t5 != 0 else 1)
        else:
            range_expansion = 1.0
            
        volatility_reversal = (range_t / range_t1) * ((close_t - close_t1) / range_t)
        
        volatility_clustering = volatility_momentum * range_expansion * volatility_reversal
        
        # Liquidity Dynamics
        # 5-day volume average
        if i >= 5:
            volume_avg_5d = df.iloc[i-4:i+1]['volume'].mean()
            volume_concentration = (volume_t / volume_avg_5d) * ((close_t - open_t) / range_t)
        else:
            volume_concentration = 1.0
            
        price_volume_divergence = ((close_t - close_t1) / range_t) - ((volume_t - volume_t1) / volume_t1 if volume_t1 != 0 else 0)
        
        # Efficiency decay using previous day
        efficiency_current = abs(close_t - open_t) / range_t
        efficiency_previous = abs(close_t1 - open_t1) / range_t1
        efficiency_decay = efficiency_current * efficiency_previous
        
        liquidity_dynamics = volume_concentration * price_volume_divergence * efficiency_decay
        
        # Market Quality
        market_quality = volatility_clustering * liquidity_dynamics
        
        # Final Alpha
        alpha = microstructure_score * rejection_strength * market_quality
        
        result.iloc[i] = alpha
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
