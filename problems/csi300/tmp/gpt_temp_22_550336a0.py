import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def ATR(data, period):
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        return true_range.rolling(window=period).mean()
    
    # Multi-Timeframe Regime Classification
    # Volatility Regime
    vol_short = data['close'].rolling(window=5).std()
    vol_long = data['close'].rolling(window=20).std()
    vol_ratio = vol_short / vol_long
    
    volatility_regime = pd.Series(0, index=data.index)
    volatility_regime[vol_ratio > 1.2] = 1  # High volatility
    volatility_regime[vol_ratio < 0.8] = -1  # Low volatility
    
    # Volume Regime
    volume_sma_10 = data['volume'].rolling(window=10).mean()
    volume_ratio = data['volume'] / volume_sma_10
    
    volume_regime = pd.Series(0, index=data.index)
    volume_regime[volume_ratio > 1.5] = 1  # Volume spike
    volume_regime[volume_ratio < 0.7] = -1  # Volume drought
    
    # Overnight Momentum Regime
    overnight_gap = np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    strong_overnight_gap = (overnight_gap > 0.01).astype(int)
    
    overnight_reversal = []
    for i in range(len(data)):
        if i < 5:
            overnight_reversal.append(0)
            continue
        count = 0
        for j in range(i-4, i+1):
            if j <= 0:
                continue
            open_close_sign = np.sign(data['open'].iloc[j] - data['close'].iloc[j-1])
            close_open_sign = np.sign(data['close'].iloc[j] - data['open'].iloc[j])
            if open_close_sign != close_open_sign:
                count += 1
        overnight_reversal.append(count)
    
    overnight_reversal = pd.Series(overnight_reversal, index=data.index)
    overnight_regime = strong_overnight_gap * overnight_reversal
    
    # Fractal Divergence Structure
    # Micro-Fractal Patterns (1-3 days)
    gap_absorption = np.abs(data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    opening_range_strength = (data['high'] - data['open']) / (data['open'] - data['low']).replace(0, np.nan)
    
    range_expansion = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    momentum_persistence = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    overnight_gap_reversal = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    fractal_completion = gap_absorption * momentum_persistence
    
    micro_fractal = (gap_absorption.fillna(0) + opening_range_strength.fillna(0) + 
                    range_expansion.fillna(0) + momentum_persistence.fillna(0) + 
                    overnight_gap_reversal.fillna(0) + fractal_completion.fillna(0))
    
    # Meso-Fractal Patterns (4-10 days)
    consecutive_up = []
    for i in range(len(data)):
        if i < 4:
            consecutive_up.append(0)
            continue
        count = 0
        for j in range(i-3, i+1):
            if data['close'].iloc[j] > data['close'].iloc[j-1]:
                count += 1
        consecutive_up.append(count)
    
    consecutive_up = pd.Series(consecutive_up, index=data.index)
    atr_5 = ATR(data, 5)
    fractal_breakouts = (data['high'] - data['high'].shift(5)) / atr_5.replace(0, np.nan)
    
    volume_spike_fractals = (data['volume'] / data['volume'].shift(1) > 1.3).astype(float)
    volume_sma_5 = data['volume'].rolling(window=5).mean()
    volume_trend_consistency = data['volume'] / volume_sma_5
    
    meso_fractal = (consecutive_up.fillna(0) + fractal_breakouts.fillna(0) + 
                   volume_spike_fractals.fillna(0) + volume_trend_consistency.fillna(0))
    
    # Fractal Divergence Composite
    micro_meso_alignment = np.sign(micro_fractal) * np.sign(meso_fractal)
    overnight_anchored_fractal = micro_fractal * overnight_gap_reversal
    volume_confirmed_fractal = micro_meso_alignment * volume_trend_consistency
    
    fractal_divergence = (micro_meso_alignment.fillna(0) + overnight_anchored_fractal.fillna(0) + 
                         volume_confirmed_fractal.fillna(0))
    
    # Price-Volume-Liquidity Divergence
    # Short-term Divergence (2-5 days)
    price_momentum = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    volume_momentum = (data['volume'] - data['volume'].shift(2)) / data['volume'].shift(2)
    divergence_score = price_momentum - volume_momentum
    
    amount_volatility = np.abs(data['amount'] - data['amount'].shift(1)) / data['amount'].shift(1).replace(0, np.nan)
    amount_per_trade = data['amount'] / data['volume'].replace(0, np.nan)
    liquidity_stress = amount_volatility * amount_per_trade
    
    overnight_return = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    intraday_return = (data['close'] - data['open']) / data['open']
    overnight_intraday_gap = overnight_return - intraday_return
    
    short_term_divergence = (divergence_score.fillna(0) + liquidity_stress.fillna(0) + 
                            overnight_intraday_gap.fillna(0))
    
    # Medium-term Divergence (6-20 days)
    atr_10 = ATR(data, 10)
    price_trend = (data['close'] - data['close'].shift(10)) / atr_10.replace(0, np.nan)
    volume_trend = (data['volume'] - data['volume'].shift(10)) / data['volume'].shift(10).replace(0, np.nan)
    trend_alignment = np.sign(price_trend) * np.sign(volume_trend)
    
    close_position = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    volume_weighted_position = close_position * data['volume']
    volume_weighted_sma_10 = volume_weighted_position.rolling(window=10).mean()
    accumulation_score = volume_weighted_position - volume_weighted_sma_10
    
    medium_term_divergence = (trend_alignment.fillna(0) + accumulation_score.fillna(0))
    
    # Divergence Anchoring Composite
    multi_timeframe_divergence = short_term_divergence * medium_term_divergence
    liquidity_anchored_divergence = divergence_score * liquidity_stress
    overnight_anchored_divergence = multi_timeframe_divergence * overnight_intraday_gap
    
    price_volume_divergence = (multi_timeframe_divergence.fillna(0) + 
                              liquidity_anchored_divergence.fillna(0) + 
                              overnight_anchored_divergence.fillna(0))
    
    # Volatility Regime Integration
    # High Volatility Regime
    fractal_divergence_vol_scaling = fractal_divergence / atr_5.replace(0, np.nan)
    divergence_vol_adjustment = divergence_score * (vol_short / vol_long).replace(0, np.nan)
    high_vol_factor = fractal_divergence_vol_scaling * divergence_vol_adjustment
    
    # Low Volatility Regime
    enhanced_fractal_divergence = fractal_divergence * atr_5
    amplified_divergence = divergence_score / atr_5.replace(0, np.nan)
    low_vol_factor = enhanced_fractal_divergence * amplified_divergence
    
    # Volume Spike Regime
    volume_confirmed_fractal_regime = fractal_divergence * volume_ratio
    volume_anchored_divergence_regime = divergence_score * volume_ratio
    volume_spike_factor = volume_confirmed_fractal_regime * volume_anchored_divergence_regime
    
    # Overnight Momentum Regime
    overnight_aligned_fractal = fractal_divergence * overnight_gap_reversal
    overnight_confirmed_divergence = divergence_score * overnight_intraday_gap
    overnight_factor = overnight_aligned_fractal * overnight_confirmed_divergence
    
    # Multi-Scale Integration
    # Ultra-short-term Patterns
    opening_momentum = np.sign(data['open'] - data['close'].shift(1))
    intraday_momentum = np.sign(data['close'] - data['open'])
    
    momentum_conflict = []
    for i in range(len(data)):
        if i < 3:
            momentum_conflict.append(0)
            continue
        count = 0
        for j in range(i-2, i+1):
            if opening_momentum.iloc[j] != intraday_momentum.iloc[j]:
                count += 1
        momentum_conflict.append(count)
    
    momentum_conflict = pd.Series(momentum_conflict, index=data.index)
    
    # Short-term Confirmation
    consecutive_divergence = []
    for i in range(len(data)):
        if i < 5:
            consecutive_divergence.append(0)
            continue
        count = 0
        current_conflict = momentum_conflict.iloc[i] > 0
        for j in range(i-4, i):
            if momentum_conflict.iloc[j] > 0 == current_conflict:
                count += 1
        consecutive_divergence.append(count)
    
    consecutive_divergence = pd.Series(consecutive_divergence, index=data.index)
    
    volume_confirmation = []
    for i in range(len(data)):
        if i < 5:
            volume_confirmation.append(0)
            continue
        count = 0
        for j in range(i-4, i+1):
            if data['volume'].iloc[j] > data['volume'].iloc[j-1] and momentum_conflict.iloc[j] > 0:
                count += 1
        volume_confirmation.append(count)
    
    volume_confirmation = pd.Series(volume_confirmation, index=data.index)
    short_term_factor = consecutive_divergence * volume_confirmation
    
    # Multi-scale Core
    scale_coherence = np.sign(momentum_conflict) * np.sign(short_term_factor)
    multi_scale_divergence = momentum_conflict * short_term_factor
    multi_scale_factor = multi_scale_divergence * scale_coherence
    
    # Regime-Adaptive Final Alpha
    # Regime Detection and Weighting
    vol_weight = np.abs(volatility_regime)
    volume_weight = np.abs(volume_regime)
    overnight_weight = np.abs(overnight_regime)
    
    # Adaptive Factor Synthesis
    regime_weighted_fractal = (fractal_divergence * vol_weight.fillna(0) + 
                              fractal_divergence * volume_weight.fillna(0) + 
                              fractal_divergence * overnight_weight.fillna(0))
    
    regime_weighted_divergence = (price_volume_divergence * vol_weight.fillna(0) + 
                                 price_volume_divergence * volume_weight.fillna(0) + 
                                 price_volume_divergence * overnight_weight.fillna(0))
    
    # Final Alpha
    multi_scale_fractal_divergence_overnight_anchor = (
        regime_weighted_fractal.fillna(0) + 
        regime_weighted_divergence.fillna(0) + 
        multi_scale_factor.fillna(0) + 
        liquidity_stress.fillna(0)
    )
    
    return multi_scale_fractal_divergence_overnight_anchor
