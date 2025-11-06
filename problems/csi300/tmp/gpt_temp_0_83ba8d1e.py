import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Microstructure Momentum Divergence factor combining intraday patterns,
    volume distribution, price-level effects, order flow signals, and cross-asset spillovers.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only current and past data
        
        # Intraday Price Efficiency Patterns
        # Opening Gap Momentum Persistence
        if i >= 1:
            prev_close = current_data['close'].iloc[i-1]
            open_price = current_data['open'].iloc[i]
            close_price = current_data['close'].iloc[i]
            gap_momentum = (close_price - open_price) / (abs(open_price - prev_close) + 1e-8) * np.sign(close_price - open_price)
        else:
            gap_momentum = 0
            
        # Midday Reversal Pressure
        high = current_data['high'].iloc[i]
        low = current_data['low'].iloc[i]
        current_volume = current_data['volume'].iloc[i]
        
        if i >= 5:
            avg_volume_5d = current_data['volume'].iloc[i-5:i+1].mean()
            reversal_pressure = ((high + low - 2 * open_price) / (high - low + 1e-8)) * (current_volume / (avg_volume_5d + 1e-8))
        else:
            reversal_pressure = 0
            
        # Closing Auction Dominance (simplified)
        close_auction = (close_price - (high + low)/2) / ((high - low)/2 + 1e-8)
        
        # Multi-timeframe Volume Distribution (simplified proxies)
        # Early Session Accumulation proxy
        if i >= 1:
            daily_range = high - low
            early_session = (current_data['volume'].iloc[i] / (current_data['volume'].iloc[i] + 1e-8)) * ((close_price - open_price) / (daily_range + 1e-8))
        else:
            early_session = 0
            
        # Price-Level Memory Effects
        # Recent High/Low Attraction
        recent_5d = current_data.iloc[i-4:i+1]
        high_5d = recent_5d['high'].max()
        low_5d = recent_5d['low'].min()
        range_5d = high_5d - low_5d + 1e-8
        
        high_attraction = (close_price - high_5d) / range_5d
        low_attraction = (close_price - low_5d) / range_5d
        price_level_effect = high_attraction - low_attraction
        
        # Round Number Magnetism (simplified)
        recent_10d = current_data.iloc[max(0, i-9):i+1]
        round_touches = 0
        for j in range(len(recent_10d)):
            price_levels = [recent_10d['high'].iloc[j], recent_10d['low'].iloc[j], recent_10d['close'].iloc[j]]
            for price in price_levels:
                last_two_digits = int((price * 100) % 100)
                if last_two_digits in [0, 50]:
                    round_touches += 1
        
        round_magnetism = round_touches / (len(recent_10d) * 3 + 1e-8)
        
        # Fibonacci Retracement Clustering
        recent_20d = current_data.iloc[max(0, i-19):i+1]
        high_20d = recent_20d['high'].max()
        low_20d = recent_20d['low'].min()
        range_20d = high_20d - low_20d + 1e-8
        
        fib_levels = [0.382, 0.5, 0.618]
        min_distance = float('inf')
        for level in fib_levels:
            fib_price = low_20d + level * range_20d
            distance = abs(close_price - fib_price) / (range_20d + 1e-8)
            min_distance = min(min_distance, distance)
        
        fib_clustering = -min_distance  # Negative because closer is better
        
        # Order Flow Imbalance Signals (proxies)
        # Bid-Ask Spread Dynamics proxy
        spread_proxy = (high - low) / (current_data['close'].iloc[i] + 1e-8)
        
        # Combine components with weights
        factor_value = (
            0.15 * gap_momentum +
            0.12 * reversal_pressure +
            0.10 * close_auction +
            0.08 * early_session +
            0.15 * price_level_effect +
            0.12 * round_magnetism +
            0.13 * fib_clustering +
            0.15 * spread_proxy
        )
        
        result.iloc[i] = factor_value
    
    return result
