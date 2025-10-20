import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Recent-Emphasized Price Components
    price_3d = df['close'] / df['close'].shift(3) - 1
    price_5d = df['close'] / df['close'].shift(5) - 1
    price_10d = df['close'] / df['close'].shift(10) - 1
    
    # Recent-Emphasized Volume Components
    volume_3d = df['volume'] / df['volume'].shift(3) - 1
    volume_5d = df['volume'] / df['volume'].shift(5) - 1
    volume_10d = df['volume'] / df['volume'].shift(10) - 1
    
    # Asymmetric Persistence Scoring
    # Price persistence assessment
    ultra_to_short_price = price_3d.sign() * price_5d.sign()
    short_to_medium_price = price_5d.sign() * price_10d.sign()
    total_price_persistence = 0.7 * ultra_to_short_price + 0.3 * short_to_medium_price
    
    # Volume persistence assessment
    ultra_to_short_volume = volume_3d.sign() * volume_5d.sign()
    short_to_medium_volume = volume_5d.sign() * volume_10d.sign()
    total_volume_persistence = 0.8 * ultra_to_short_volume + 0.2 * short_to_medium_volume
    
    # Recent-Weighted Signal Construction
    weighted_price_trend = 0.6 * price_3d + 0.3 * price_5d + 0.1 * price_10d
    weighted_volume_trend = 0.7 * volume_3d + 0.2 * volume_5d + 0.1 * volume_10d
    
    # Persistence-Amplified Base Signal
    base_trend_signal = weighted_price_trend * weighted_volume_trend
    persistence_multiplier = 1.0 + (0.25 * total_price_persistence) + (0.2 * total_volume_persistence)
    amplified_base = base_trend_signal * persistence_multiplier
    
    # Additive Recent Enhancement
    ultra_recent_price_boost = 2.0 * price_3d
    ultra_recent_volume_boost = 1.5 * volume_3d
    final_factor = amplified_base + ultra_recent_price_boost + ultra_recent_volume_boost
    
    return final_factor
