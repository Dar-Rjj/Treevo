import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Momentum Asymmetry
    # Micro
    micro_fma = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8) - \
                (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Meso (2-day window)
    high_2d = data['high'].rolling(window=3, min_periods=1).max()
    low_2d = data['low'].rolling(window=3, min_periods=1).min()
    meso_fma = (high_2d - data['open']) / (high_2d - low_2d + 1e-8) - \
               (data['open'] - low_2d) / (high_2d - low_2d + 1e-8)
    
    # Macro (5-day window)
    high_5d = data['high'].rolling(window=6, min_periods=1).max()
    low_5d = data['low'].rolling(window=6, min_periods=1).min()
    macro_fma = (high_5d - data['open']) / (high_5d - low_5d + 1e-8) - \
                (data['open'] - low_5d) / (high_5d - low_5d + 1e-8)
    
    fma_cascade = micro_fma * meso_fma * macro_fma
    
    # Volume-Momentum Alignment
    # Volume Direction
    price_direction = np.sign(data['close'] - data['close'].shift(1))
    volume_change = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    volume_direction = price_direction * volume_change
    
    # Volume Spike
    volume_avg_5d = data['volume'].shift(1).rolling(window=5, min_periods=1).mean()
    volume_spike = data['volume'] / (volume_avg_5d + 1e-8)
    
    # Volume Trend
    volume_trend = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            count = 0
            for j in range(i-4, i+1):
                if j > 0 and data['volume'].iloc[j] > data['volume'].iloc[j-1]:
                    count += 1
            volume_trend.iloc[i] = count
    
    # Momentum Range Efficiency
    # Micro
    micro_mre = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Meso (5-day window)
    high_5d_range = data['high'].rolling(window=6, min_periods=1).max()
    low_5d_range = data['low'].rolling(window=6, min_periods=1).min()
    meso_mre = (data['close'] - data['close'].shift(5)) / (high_5d_range - low_5d_range + 1e-8)
    
    # Macro (13-day window)
    high_13d = data['high'].rolling(window=14, min_periods=1).max()
    low_13d = data['low'].rolling(window=14, min_periods=1).min()
    macro_mre = (data['close'] - data['close'].shift(13)) / (high_13d - low_13d + 1e-8)
    
    mre_cascade = micro_mre * meso_mre * macro_mre
    
    # Momentum Persistence
    # Short-term persistence
    short_term_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        if i >= 4:
            count = 0
            for j in range(i-4, i+1):
                if j >= 2:
                    momentum1 = data['close'].iloc[j] - data['close'].iloc[j-1]
                    momentum2 = data['close'].iloc[j-1] - data['close'].iloc[j-2]
                    if momentum1 * momentum2 > 0:
                        count += 1
            short_term_persistence.iloc[i] = count
    
    # Medium-term persistence
    medium_term_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            pos_count = 0
            neg_count = 0
            for j in range(i-4, i+1):
                if j >= 0:
                    fma_val = micro_fma.iloc[j]
                    if fma_val > 0:
                        pos_count += 1
                    elif fma_val < 0:
                        neg_count += 1
            medium_term_persistence.iloc[i] = pos_count - neg_count
    
    persistence_ratio = short_term_persistence / (medium_term_persistence + 1e-8)
    
    # Price-Volume Divergence
    # Negative Divergence
    price_change_5d = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8)
    volume_change_5d = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    negative_divergence = price_change_5d - volume_change_5d
    
    # Divergence Reversal
    price_dir_prev = np.sign(data['close'].shift(1) - data['close'].shift(2))
    divergence_reversal = price_dir_prev * (negative_divergence - negative_divergence.shift(1))
    
    # Volume-Price Efficiency
    vol_price_efficiency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            price_sum = 0
            for j in range(i-4, i+1):
                if j > 0:
                    price_sum += abs(data['close'].iloc[j] - data['close'].iloc[j-1])
            vpe = (data['close'].iloc[i] - data['close'].iloc[i-5]) / (price_sum + 1e-8)
            vol_ratio = data['volume'].iloc[i] / (data['volume'].iloc[i-5] + 1e-8)
            vol_price_efficiency.iloc[i] = vpe * vol_ratio
    
    # Regime Integration
    # High Efficiency
    high_efficiency = mre_cascade * volume_direction * fma_cascade
    
    # Low Efficiency
    low_efficiency = -mre_cascade * np.sign(data['close'] - data['open']) * \
                    (data['high'] - data['low']) * micro_fma
    
    # Transition
    transition = (mre_cascade - mre_cascade.shift(3)) * \
                (volume_direction - volume_direction.shift(3)) * \
                persistence_ratio
    
    # Core regime integration (simple weighted combination)
    core = 0.4 * high_efficiency + 0.3 * low_efficiency + 0.3 * transition
    
    # Volatility Adjustment
    volatility_adjusted = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            vol_sum = 0
            for j in range(i-4, i+1):
                if j >= 0:
                    range_val = data['high'].iloc[j] - data['low'].iloc[j]
                    high_close = abs(data['high'].iloc[j] - data['close'].iloc[j-1]) if j > 0 else 0
                    low_close = abs(data['low'].iloc[j] - data['close'].iloc[j-1]) if j > 0 else 0
                    vol_sum += max(range_val, high_close, low_close)
            volatility = vol_sum / 5
            volatility_adjusted.iloc[i] = core.iloc[i] / (volatility + 1e-8)
    
    # Volume Confirmed
    volume_confirmed = volatility_adjusted * volume_spike * volume_trend
    
    # Final Alpha
    final_alpha = volume_confirmed * divergence_reversal * np.sign(core)
    
    return final_alpha
