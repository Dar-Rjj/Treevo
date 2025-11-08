import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility regime switching, fractal analysis,
    momentum transfer, volume asymmetry, price elasticity, and range-volume convexity.
    """
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # 1. Intraday Volatility Regime Switching
    daily_range = data['high'] - data['low']
    range_ratio = daily_range / daily_range.shift(1)
    
    # Regime detection
    high_vol_regime = (range_ratio > 1.8).astype(int)
    low_vol_regime = (range_ratio < 0.6).astype(int)
    normal_regime = ((range_ratio >= 0.6) & (range_ratio <= 1.8)).astype(int)
    
    # Regime persistence
    regime_persistence = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if high_vol_regime.iloc[i] == 1:
            regime_persistence.iloc[i] = regime_persistence.iloc[i-1] + 1 if high_vol_regime.iloc[i-1] == 1 else 1
        elif low_vol_regime.iloc[i] == 1:
            regime_persistence.iloc[i] = regime_persistence.iloc[i-1] + 1 if low_vol_regime.iloc[i-1] == 1 else 1
        else:
            regime_persistence.iloc[i] = regime_persistence.iloc[i-1] + 1 if normal_regime.iloc[i-1] == 1 else 1
    
    # 2. Price-Volume Fractal Dimension
    price_changes = data['close'].diff().abs()
    volume_changes = data['volume'].diff().abs()
    
    # Rolling fractal calculations (5-day window)
    window = 5
    price_fractal = pd.Series(index=data.index, dtype=float)
    volume_fractal = pd.Series(index=data.index, dtype=float)
    
    for i in range(window, len(data)):
        window_data = data.iloc[i-window:i+1]
        price_sum = price_changes.iloc[i-window+1:i+1].sum()
        volume_sum = volume_changes.iloc[i-window+1:i+1].sum()
        
        if price_sum > 0 and len(window_data) > 1:
            price_fractal.iloc[i] = np.log(price_sum) / np.log(len(window_data) - 1)
        if volume_sum > 0 and len(window_data) > 1:
            volume_fractal.iloc[i] = np.log(volume_sum) / np.log(len(window_data) - 1)
    
    fractal_complexity = price_fractal * volume_fractal
    
    # 3. Session Boundary Momentum Transfer
    prev_range = data['high'].shift(1) - data['low'].shift(1)
    gap_momentum = (data['open'] - data['close'].shift(1)) / prev_range.replace(0, np.nan)
    
    # Gap filling speed approximation (using intraday range)
    gap_fill_speed = (data['high'] - data['low']) / prev_range.replace(0, np.nan)
    
    # 4. Volume Asymmetry Accumulation
    up_volume = data['volume'].where(data['close'] > data['open'], 0)
    down_volume = data['volume'].where(data['close'] < data['open'], 0)
    neutral_threshold = data['open'] * 0.002
    neutral_volume = data['volume'].where(abs(data['close'] - data['open']) < neutral_threshold, 0)
    
    # 3-day rolling sums
    up_volume_dominance = up_volume.rolling(window=3).sum() / data['volume'].rolling(window=3).sum()
    down_volume_concentration = down_volume.rolling(window=3).sum() / data['volume'].rolling(window=3).sum()
    neutral_volume_ratio = neutral_volume.rolling(window=3).sum() / data['volume'].rolling(window=3).sum()
    
    volume_asymmetry = up_volume_dominance - down_volume_concentration
    
    # 5. Price Elasticity Hysteresis
    up_move_elasticity = (data['high'] - data['open']) / up_volume.replace(0, np.nan)
    down_move_elasticity = (data['open'] - data['low']) / down_volume.replace(0, np.nan)
    
    hysteresis_gap = abs(up_move_elasticity - down_move_elasticity)
    
    # 6. Range-Volume Convexity
    range_expansion = daily_range / daily_range.shift(1)
    volume_expansion = data['volume'] / data['volume'].shift(1)
    
    convexity_score = range_expansion / volume_expansion.replace(0, np.nan)
    
    # Combine all components into final alpha factor
    for i in range(2, len(data)):
        if pd.notna(price_fractal.iloc[i]) and pd.notna(volume_fractal.iloc[i]):
            # Weighted combination of factors
            regime_factor = -regime_persistence.iloc[i] * high_vol_regime.iloc[i]  # Negative for high volatility persistence
            fractal_factor = fractal_complexity.iloc[i] * 0.5
            momentum_factor = gap_momentum.iloc[i] * gap_fill_speed.iloc[i] if pd.notna(gap_momentum.iloc[i]) else 0
            volume_factor = volume_asymmetry.iloc[i] * 2 if pd.notna(volume_asymmetry.iloc[i]) else 0
            elasticity_factor = hysteresis_gap.iloc[i] * np.sign(up_move_elasticity.iloc[i] - down_move_elasticity.iloc[i]) if pd.notna(hysteresis_gap.iloc[i]) else 0
            convexity_factor = -convexity_score.iloc[i] if pd.notna(convexity_score.iloc[i]) else 0  # Negative for high convexity
            
            alpha.iloc[i] = (regime_factor + fractal_factor + momentum_factor + 
                           volume_factor + elasticity_factor + convexity_factor)
    
    # Normalize the alpha factor
    alpha = (alpha - alpha.rolling(window=20).mean()) / alpha.rolling(window=20).std()
    
    return alpha.fillna(0)
