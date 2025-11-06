import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Multi-Timeframe Momentum Convergence with Efficiency Breakout
    # Calculate momentum across timeframes
    mom_3 = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_8 = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    mom_13 = (df['close'] - df['close'].shift(13)) / df['close'].shift(13)
    mom_21 = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
    
    # Assess momentum convergence strength
    momentum_signs = pd.DataFrame({
        'mom_3': np.sign(mom_3),
        'mom_8': np.sign(mom_8),
        'mom_13': np.sign(mom_13),
        'mom_21': np.sign(mom_21)
    })
    
    convergence_score = np.zeros(len(df))
    momentum_magnitude = np.zeros(len(df))
    
    for i in range(len(df)):
        if i < 21:
            convergence_score[i] = 0
            momentum_magnitude[i] = 0
            continue
            
        signs = momentum_signs.iloc[i]
        positive_count = (signs > 0).sum()
        negative_count = (signs < 0).sum()
        
        if positive_count == 4 or negative_count == 4:
            convergence_score[i] = 2.0  # Full convergence
        elif positive_count == 3 or negative_count == 3:
            convergence_score[i] = 1.5  # Strong convergence
        elif positive_count == 2 or negative_count == 2:
            convergence_score[i] = 1.0  # Weak convergence
        else:
            convergence_score[i] = 0.5  # No convergence
            
        # Calculate average momentum magnitude
        current_moms = [mom_3.iloc[i], mom_8.iloc[i], mom_13.iloc[i], mom_21.iloc[i]]
        momentum_magnitude[i] = np.mean(np.abs(current_moms))
    
    # Evaluate intraday efficiency breakout
    intraday_efficiency = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate rolling highs and lows for breakout levels
    rolling_high = df['high'].rolling(window=20, min_periods=1).max()
    rolling_low = df['low'].rolling(window=20, min_periods=1).min()
    
    # Compute distance to nearest level
    distance_to_high = np.abs(df['close'] - rolling_high) / rolling_high
    distance_to_low = np.abs(df['close'] - rolling_low) / rolling_low
    min_distance = np.minimum(distance_to_high, distance_to_low)
    
    # Assess breakout strength
    breakout_multiplier = np.ones(len(df))
    for i in range(len(df)):
        if i < 20:
            breakout_multiplier[i] = 1.0
            continue
            
        if distance_to_high.iloc[i] < distance_to_low.iloc[i]:
            # Near resistance - amplify upward convergence if momentum is positive
            if momentum_signs.iloc[i].mean() > 0:
                breakout_multiplier[i] = 1.2
            else:
                breakout_multiplier[i] = 0.8
        else:
            # Near support - amplify downward convergence if momentum is negative
            if momentum_signs.iloc[i].mean() < 0:
                breakout_multiplier[i] = 1.2
            else:
                breakout_multiplier[i] = 0.8
    
    # Generate combined signal for first factor
    base_factor1 = convergence_score * momentum_magnitude
    factor1 = base_factor1 * (1 + intraday_efficiency) * (1 + min_distance * np.sign(momentum_signs.mean(axis=1)))
    
    # Volume-Price Divergence with Momentum Acceleration
    # Calculate multi-timeframe volume trends
    volume_mom_3 = df['volume'] / df['volume'].shift(3)
    volume_ratio_5 = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Calculate volume persistence
    volume_persistence = np.zeros(len(df))
    above_avg_count = 0
    for i in range(len(df)):
        if i == 0:
            volume_persistence[i] = 0
            continue
        if volume_ratio_5.iloc[i] > 1:
            above_avg_count = min(5, above_avg_count + 1)
        else:
            above_avg_count = 0
        volume_persistence[i] = above_avg_count
    
    # Assess momentum acceleration
    mom_acceleration = mom_3 - mom_8  # Short vs medium momentum difference
    
    # Identify divergence patterns
    divergence_base = np.zeros(len(df))
    for i in range(len(df)):
        if i < 8:
            divergence_base[i] = 0
            continue
            
        volume_trend = volume_mom_3.iloc[i]
        mom_accel = mom_acceleration.iloc[i]
        
        # Volume increasing while momentum decelerating (negative divergence)
        if volume_trend > 1.1 and mom_accel < -0.01:
            divergence_base[i] = -1.0
        # Volume decreasing while momentum accelerating (positive divergence)
        elif volume_trend < 0.9 and mom_accel > 0.01:
            divergence_base[i] = 1.0
        # Volume-momentum alignment
        elif (volume_trend > 1 and mom_accel > 0) or (volume_trend < 1 and mom_accel < 0):
            divergence_base[i] = 0.5 * np.sign(mom_accel)
        else:
            divergence_base[i] = 0
    
    # Generate divergence factor
    divergence_strength = np.abs(mom_acceleration) * volume_ratio_5
    persistence_score = 1 + (volume_persistence / 5)
    factor2 = divergence_base * persistence_score * divergence_strength
    
    # Gap Quality with Liquidity-Weighted Momentum
    gap_percentage = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Classify gap size
    gap_size = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            gap_size[i] = 0
            continue
            
        gap_pct = abs(gap_percentage.iloc[i])
        if gap_pct > 0.015:
            gap_size[i] = 3.0  # Large gap
        elif gap_pct > 0.005:
            gap_size[i] = 2.0  # Medium gap
        else:
            gap_size[i] = 1.0  # Small gap
    
    # Evaluate liquidity and momentum confirmation
    liquidity_efficiency = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    liquidity_efficiency = liquidity_efficiency.replace([np.inf, -np.inf], 0).fillna(0)
    
    effective_spread = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    avg_spread_5 = effective_spread.rolling(window=5, min_periods=1).mean()
    spread_ratio = effective_spread / avg_spread_5
    
    # Momentum alignment for gap confirmation
    gap_momentum_alignment = np.zeros(len(df))
    for i in range(len(df)):
        if i < 3:
            gap_momentum_alignment[i] = 0
            continue
            
        gap_dir = np.sign(gap_percentage.iloc[i])
        mom_3_dir = np.sign(mom_3.iloc[i])
        
        if gap_dir == mom_3_dir:
            gap_momentum_alignment[i] = 1.0  # Confirmation
        elif gap_dir == -mom_3_dir:
            gap_momentum_alignment[i] = -1.0  # Contradiction
        else:
            gap_momentum_alignment[i] = 0.0  # Neutral
    
    # Generate gap quality score
    base_gap_factor = gap_percentage * volume_ratio_5
    factor3 = base_gap_factor * (1 + liquidity_efficiency) * (1 + gap_momentum_alignment)
    
    # Range-Based Reversion with Volatility-Regime Adaptation
    normalized_range = (df['high'] - df['low']) / df['close']
    
    # Calculate True Range
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    avg_true_range_10 = true_range.rolling(window=10, min_periods=1).mean()
    volatility_ratio = true_range / avg_true_range_10
    
    # Previous day's return
    prev_return = df['close'].pct_change(1)
    
    # Generate adaptive reversion signal
    base_reversion = normalized_range * prev_return.shift(1)
    
    # Apply volatility-regime adjustment
    volatility_adjustment = np.ones(len(df))
    for i in range(len(df)):
        if i < 10:
            volatility_adjustment[i] = 1.0
            continue
            
        vol_ratio = volatility_ratio.iloc[i]
        if vol_ratio > 1.5:
            volatility_adjustment[i] = 0.7  # High volatility: reduce sensitivity
        elif vol_ratio < 0.7:
            volatility_adjustment[i] = 1.3  # Low volatility: amplify strength
        else:
            volatility_adjustment[i] = 1.0  # Normal volatility
    
    factor4 = base_reversion * volatility_adjustment / (1 + volatility_ratio)
    
    # Efficiency-Weighted Momentum with Volume Persistence
    price_efficiency = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    price_efficiency = price_efficiency.replace([np.inf, -np.inf], 0).fillna(0)
    
    momentum_efficiency = momentum_magnitude / normalized_range
    momentum_efficiency = momentum_efficiency.replace([np.inf, -np.inf], 0).fillna(0)
    
    volume_efficiency = volume_ratio_5 * (volume_persistence / 5)
    
    # Assess momentum-volume alignment
    alignment_strength = np.zeros(len(df))
    for i in range(len(df)):
        if i < 21:
            alignment_strength[i] = 0
            continue
            
        mom_dir = np.sign(momentum_signs.iloc[i].mean())
        volume_dir = np.sign(volume_mom_3.iloc[i] - 1)
        
        if mom_dir == volume_dir:
            alignment_strength[i] = 1.5  # Strong alignment
        elif mom_dir == 0 or volume_dir == 0:
            alignment_strength[i] = 1.0  # Neutral
        else:
            alignment_strength[i] = 0.7  # Weak alignment
    
    # Generate efficiency-weighted signal
    base_momentum = convergence_score * momentum_magnitude
    efficiency_weighting = price_efficiency * volume_efficiency * alignment_strength
    
    # Apply efficiency weighting with regime awareness
    final_efficiency = np.ones(len(df))
    for i in range(len(df)):
        eff_weight = efficiency_weighting.iloc[i]
        vol_regime = volatility_ratio.iloc[i]
        
        if eff_weight > 1.2:
            # High efficiency
            if vol_regime < 1.2:
                final_efficiency[i] = 1.5  # Amplify in normal/low vol
            else:
                final_efficiency[i] = 1.2  # Moderate amplification in high vol
        elif eff_weight < 0.8:
            # Low efficiency
            final_efficiency[i] = 0.7  # Dampen signal
        else:
            # Mixed efficiency
            final_efficiency[i] = 1.0  # Neutral
    
    factor5 = base_momentum * final_efficiency
    
    # Combine all factors with equal weighting
    combined_factor = (factor1 + factor2 + factor3 + factor4 + factor5) / 5
    
    # Clean and return the factor
    combined_factor = combined_factor.replace([np.inf, -np.inf], 0).fillna(0)
    
    return pd.Series(combined_factor, index=df.index)
