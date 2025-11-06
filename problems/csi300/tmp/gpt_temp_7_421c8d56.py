import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Momentum-Volume Divergence with Dynamic Regime Adjustment
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Price Momentum Calculation
    momentum_3 = df['close'] / df['close'].shift(3) - 1
    momentum_5 = df['close'] / df['close'].shift(5) - 1
    momentum_10 = df['close'] / df['close'].shift(10) - 1
    momentum_20 = df['close'] / df['close'].shift(20) - 1
    
    # Momentum Quality Assessment
    momentum_signals = pd.DataFrame({
        'm3': momentum_3 > 0,
        'm5': momentum_5 > 0,
        'm10': momentum_10 > 0,
        'm20': momentum_20 > 0
    })
    
    positive_count = momentum_signals.sum(axis=1)
    negative_count = 4 - positive_count
    net_momentum_direction = (positive_count - negative_count) / 4
    
    # Momentum acceleration
    short_term_acceleration = momentum_5 - momentum_10
    medium_term_acceleration = momentum_10 - momentum_20
    
    # Volume-Price Divergence Framework
    # Volume Momentum Analysis
    volume_momentum_3 = df['volume'] / df['volume'].shift(3) - 1
    volume_momentum_5 = df['volume'] / df['volume'].shift(5) - 1
    volume_momentum_10 = df['volume'] / df['volume'].shift(10) - 1
    volume_momentum_20 = df['volume'] / df['volume'].shift(20) - 1
    
    # Divergence Detection & Quantification
    timeframe_weights = [0.4, 0.3, 0.2, 0.1]  # [3,5,10,20]-day weights
    price_momentums = [momentum_3, momentum_5, momentum_10, momentum_20]
    volume_momentums = [volume_momentum_3, volume_momentum_5, volume_momentum_10, volume_momentum_20]
    
    divergence_scores = []
    divergence_directions = []
    
    for i, (price_mom, vol_mom, weight) in enumerate(zip(price_momentums, volume_momentums, timeframe_weights)):
        # Bullish divergence: price up, volume down
        bullish_div = (price_mom > 0) & (vol_mom < 0)
        bullish_score = (abs(price_mom) + abs(vol_mom)) * weight
        
        # Bearish divergence: price down, volume up
        bearish_div = (price_mom < 0) & (vol_mom > 0)
        bearish_score = -(abs(price_mom) + abs(vol_mom)) * weight
        
        # Confirmation: price and volume same direction
        confirmation = (price_mom > 0) & (vol_mom > 0)
        confirmation_score = 0.5 * (price_mom + vol_mom) * weight
        
        # Combine scores
        timeframe_score = pd.Series(np.zeros(len(df)), index=df.index)
        timeframe_score[bullish_div] = bullish_score[bullish_div]
        timeframe_score[bearish_div] = bearish_score[bearish_div]
        timeframe_score[confirmation] = confirmation_score[confirmation]
        
        divergence_scores.append(timeframe_score)
        divergence_directions.append(timeframe_score.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)))
    
    # Cross-timeframe divergence consistency
    aggregated_divergence_score = sum(divergence_scores)
    
    # Divergence confirmation strength
    divergence_direction_df = pd.concat(divergence_directions, axis=1)
    same_direction_count = divergence_direction_df.apply(
        lambda row: sum(row == row.mode().iloc[0]) if len(row.mode()) > 0 else 0, axis=1
    )
    confirmation_strength = 1 + (same_direction_count / 4)
    
    # Dynamic Market Regime Detection
    # Volatility Regime Analysis
    short_term_vol = (df['high'] - df['low']).rolling(5).mean()
    medium_term_vol = (df['high'] - df['low']).rolling(10).mean()
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Volatility regime classification
    volatility_regime = pd.Series('normal', index=df.index)
    volatility_regime[volatility_ratio > 1.2] = 'high'
    volatility_regime[volatility_ratio < 0.8] = 'low'
    
    # Trend Regime Analysis
    price_slope = (df['close'] - df['close'].shift(20)) / 20
    avg_daily_range = (df['high'] - df['low']).rolling(20).mean()
    trend_to_noise_ratio = abs(price_slope) / avg_daily_range
    
    # Trend regime classification
    trend_regime = pd.Series('moderate', index=df.index)
    trend_regime[trend_to_noise_ratio > 0.5] = 'strong'
    trend_regime[trend_to_noise_ratio < 0.2] = 'weak'
    
    # Adaptive Alpha Signal Generation
    # Base Signal Construction
    momentum_weights = [0.3, 0.4, 0.2, 0.1]  # [3,5,10,20]-day weights
    combined_momentum = sum(mom * weight for mom, weight in zip(price_momentums, momentum_weights))
    
    # Divergence enhancement
    base_signal = combined_momentum * (1 + aggregated_divergence_score) * confirmation_strength
    
    # Regime-Based Signal Adjustment
    # Volatility regime multipliers
    volatility_multiplier = pd.Series(1.0, index=df.index)
    volatility_multiplier[volatility_regime == 'high'] = 0.7
    volatility_multiplier[volatility_regime == 'low'] = 1.3
    
    # Trend regime multipliers
    trend_multiplier = pd.Series(1.0, index=df.index)
    trend_multiplier[trend_regime == 'strong'] = 1.4
    trend_multiplier[trend_regime == 'weak'] = 0.8
    trend_multiplier[trend_regime == 'moderate'] = 1.1
    
    # Final Alpha Signal
    final_signal = base_signal * volatility_multiplier * trend_multiplier
    
    return final_signal
