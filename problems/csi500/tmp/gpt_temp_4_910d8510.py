import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum with Volume Confirmation and Intraday Efficiency
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    # Calculate daily range
    daily_range = (data['high'] - data['low']) / data['close']
    
    # Calculate short-term volatility (5-day average range)
    short_term_vol = daily_range.rolling(window=5).mean()
    
    # Calculate medium-term volatility (20-day average range)
    medium_term_vol = daily_range.rolling(window=20).mean()
    
    # Classify regime
    high_vol_regime = short_term_vol > (medium_term_vol * 1.5)
    low_vol_regime = short_term_vol < (medium_term_vol * 0.7)
    normal_vol_regime = ~(high_vol_regime | low_vol_regime)
    
    # Multi-Timeframe Momentum Persistence
    # Calculate raw momentum
    ultra_short_momentum = data['close'] - data['close'].shift(3)
    short_term_momentum = data['close'] - data['close'].shift(8)
    medium_term_momentum = data['close'] - data['close'].shift(21)
    
    # Momentum persistence scoring
    positive_momentum = pd.DataFrame({
        'ultra_short': (ultra_short_momentum > 0).astype(int),
        'short_term': (short_term_momentum > 0).astype(int),
        'medium_term': (medium_term_momentum > 0).astype(int)
    })
    
    negative_momentum = pd.DataFrame({
        'ultra_short': (ultra_short_momentum < 0).astype(int),
        'short_term': (short_term_momentum < 0).astype(int),
        'medium_term': (medium_term_momentum < 0).astype(int)
    })
    
    net_persistence = (positive_momentum.sum(axis=1) - negative_momentum.sum(axis=1)) / 3
    
    # Volume-Price Divergence Confirmation
    # Price momentum acceleration
    price_3d = data['close'] - data['close'].shift(3)
    price_8d = data['close'] - data['close'].shift(8)
    price_accel = price_3d - price_8d
    
    # Volume momentum acceleration
    volume_3d = data['volume'] - data['volume'].shift(3)
    volume_8d = data['volume'] - data['volume'].shift(8)
    volume_accel = volume_3d - volume_8d
    
    # Divergence detection
    bullish_divergence = (price_accel < 0) & (volume_accel > 0)
    bearish_divergence = (price_accel > 0) & (volume_accel < 0)
    no_divergence = ~(bullish_divergence | bearish_divergence)
    
    # Factor Integration & Enhancement
    # Core Signal Construction
    # Initialize weighted momentum
    weighted_momentum = pd.Series(index=data.index, dtype=float)
    
    # Apply regime-adaptive weighting
    for idx in data.index:
        if high_vol_regime.loc[idx]:
            weights = [0.5, 0.3, 0.2]  # Ultra-Short, Short-Term, Medium-Term
        elif low_vol_regime.loc[idx]:
            weights = [0.3, 0.4, 0.3]
        else:  # normal volatility
            weights = [0.4, 0.35, 0.25]
        
        weighted_momentum.loc[idx] = (
            weights[0] * ultra_short_momentum.loc[idx] +
            weights[1] * short_term_momentum.loc[idx] +
            weights[2] * medium_term_momentum.loc[idx]
        )
    
    # Persistence adjustment (capped at ±0.5)
    persistence_adj = np.clip(net_persistence, -0.5, 0.5)
    adjusted_momentum = weighted_momentum * (1 + persistence_adj)
    
    # Divergence multiplier
    divergence_multiplier = pd.Series(1.0, index=data.index)
    divergence_multiplier[bullish_divergence] = 1.5
    divergence_multiplier[bearish_divergence] = -1.5
    
    core_signal = adjusted_momentum * divergence_multiplier
    
    # Intraday Efficiency Enhancement
    # Range efficiency calculation
    mid_price = (data['high'] + data['low']) / 2
    absolute_move = np.abs(data['close'] - mid_price)
    daily_range_abs = data['high'] - data['low']
    efficiency = absolute_move / daily_range_abs
    
    # Efficiency multiplier
    efficiency_multiplier = pd.Series(1.0, index=data.index)
    efficiency_multiplier[efficiency > 0.7] = 1.2
    efficiency_multiplier[efficiency < 0.3] = 0.8
    
    # Final factor value
    final_factor = core_signal * efficiency_multiplier
    
    return final_factor
