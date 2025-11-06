import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Multi-Timeframe Price Efficiency
    # 5-day price efficiency
    price_change_5d = close - close.shift(5)
    
    # Calculate maximum true range for 5-day period
    max_tr_5d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            period_high = high.iloc[i-4:i+1].max()
            period_low = low.iloc[i-4:i+1].min()
            prev_close = close.iloc[i-5]
            tr_range1 = period_high - period_low
            tr_range2 = abs(period_high - prev_close)
            tr_range3 = abs(period_low - prev_close)
            max_tr_5d.iloc[i] = max(tr_range1, tr_range2, tr_range3)
    
    price_efficiency_5d = price_change_5d / max_tr_5d.replace(0, np.nan)
    
    # 10-day price efficiency
    price_change_10d = close - close.shift(10)
    
    # Calculate maximum true range for 10-day period
    max_tr_10d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 9:
            period_high = high.iloc[i-9:i+1].max()
            period_low = low.iloc[i-9:i+1].min()
            prev_close = close.iloc[i-10]
            tr_range1 = period_high - period_low
            tr_range2 = abs(period_high - prev_close)
            tr_range3 = abs(period_low - prev_close)
            max_tr_10d.iloc[i] = max(tr_range1, tr_range2, tr_range3)
    
    price_efficiency_10d = price_change_10d / max_tr_10d.replace(0, np.nan)
    
    # Multi-Timeframe Volume Efficiency
    # 5-day volume efficiency
    volume_change_5d = volume - volume.shift(5)
    volume_efficiency_5d = volume_change_5d / max_tr_5d.replace(0, np.nan)
    
    # 10-day volume efficiency
    volume_change_10d = volume - volume.shift(10)
    volume_efficiency_10d = volume_change_10d / max_tr_10d.replace(0, np.nan)
    
    # Intraday Momentum Efficiency
    morning_efficiency = (high - open_price) / true_range.replace(0, np.nan)
    afternoon_efficiency = (close - high) / true_range.replace(0, np.nan)
    intraday_asymmetry = morning_efficiency - afternoon_efficiency
    
    # Multi-Scale Efficiency Divergence
    short_term_divergence_score = pd.Series(0, index=df.index)
    short_term_divergence_score[price_efficiency_5d > volume_efficiency_5d] = 1
    short_term_divergence_score[price_efficiency_5d < volume_efficiency_5d] = -1
    
    medium_term_divergence_score = pd.Series(0, index=df.index)
    medium_term_divergence_score[price_efficiency_10d > volume_efficiency_10d] = 1
    medium_term_divergence_score[price_efficiency_10d < volume_efficiency_10d] = -1
    
    # Volatility-Efficiency Context
    volatility_regime = true_range / true_range.shift(8)
    
    # Efficiency Strength Assessment
    short_term_efficiency_strength = abs(price_efficiency_5d) + abs(volume_efficiency_5d)
    medium_term_efficiency_strength = abs(price_efficiency_10d) + abs(volume_efficiency_10d)
    
    # Regime-Adaptive Signal Integration
    short_term_weight = pd.Series(0.6, index=df.index)  # Normal volatility default
    medium_term_weight = pd.Series(0.4, index=df.index)
    
    # Volatility-based weighting
    short_term_weight[volatility_regime > 1.2] = 0.7
    medium_term_weight[volatility_regime > 1.2] = 0.3
    
    short_term_weight[volatility_regime < 0.8] = 0.5
    medium_term_weight[volatility_regime < 0.8] = 0.5
    
    # Efficiency-enhanced adjustment
    short_term_weight[short_term_efficiency_strength > 1.5] += 0.1
    medium_term_weight[medium_term_efficiency_strength > 2.0] += 0.1
    
    # Ensure weights sum to 1
    total_weight = short_term_weight + medium_term_weight
    short_term_weight = short_term_weight / total_weight
    medium_term_weight = medium_term_weight / total_weight
    
    # Divergence Strength Measurement
    short_term_divergence_magnitude = abs(price_efficiency_5d - volume_efficiency_5d)
    medium_term_divergence_magnitude = abs(price_efficiency_10d - volume_efficiency_10d)
    avg_divergence_magnitude = (short_term_divergence_magnitude + medium_term_divergence_magnitude) / 2
    
    # Composite Alpha Factor Generation
    regime_weighted_score = (short_term_divergence_score * short_term_weight + 
                           medium_term_divergence_score * medium_term_weight)
    
    efficiency_divergence_gap = avg_divergence_magnitude * (2 - volatility_regime)
    
    intraday_momentum_scale = 1 + (intraday_asymmetry * 0.3)
    
    # Final alpha factor
    raw_factor = regime_weighted_score * efficiency_divergence_gap * intraday_momentum_scale
    
    return raw_factor
