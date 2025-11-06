import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Structure
    data['ultra_short_vol'] = (data['high'] - data['low']).rolling(window=2).sum()
    data['short_term_vol'] = (data['high'] - data['low']).rolling(window=5).sum()
    data['medium_term_vol'] = (data['high'] - data['low']).rolling(window=10).sum()
    
    # Volatility Regime Classification
    explosive_cond = (data['ultra_short_vol'] > data['short_term_vol']) & (data['short_term_vol'] > data['medium_term_vol'] / 2)
    trending_cond = (data['ultra_short_vol'] < data['short_term_vol']) & (data['short_term_vol'] > data['medium_term_vol'] / 2)
    mean_reverting_cond = (data['ultra_short_vol'] > data['short_term_vol']) & (data['short_term_vol'] < data['medium_term_vol'] / 2)
    quiet_cond = (data['ultra_short_vol'] < data['short_term_vol']) & (data['short_term_vol'] < data['medium_term_vol'] / 2)
    
    # Microstructure Efficiency Regime
    data['price_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 0.0001)
    data['volume_efficiency'] = data['volume'] / (abs(data['close'] - data['open']) + 0.0001)
    data['amount_efficiency'] = data['amount'] / (abs(data['close'] - data['open']) + 0.0001)
    
    price_driven = (data['price_efficiency'] > data['volume_efficiency']) & (data['price_efficiency'] > data['amount_efficiency'])
    volume_driven = (data['volume_efficiency'] > data['price_efficiency']) & (data['volume_efficiency'] > data['amount_efficiency'])
    amount_driven = (data['amount_efficiency'] > data['price_efficiency']) & (data['amount_efficiency'] > data['volume_efficiency'])
    
    # Regime-Specific Momentum Dynamics
    # Explosive Regime Momentum
    gap_momentum = ((data['close'] - data['open']) / (data['high'] - data['low'])) * ((data['high'] - data['low']) / ((data['high'] - data['low']).shift(1).rolling(window=4).mean()))
    volume_accel = (data['volume'] / data['volume'].shift(1)) * ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low']))
    explosive_momentum = gap_momentum * volume_accel
    
    # Trending Regime Momentum
    def trend_consistency(series):
        signs = np.sign(series.diff())
        current_sign = signs.iloc[-1]
        window_signs = signs.iloc[-6:-1] if len(signs) >= 6 else signs.iloc[:-1]
        return (window_signs == current_sign).sum() / 6
    
    trend_consist = data['close'].rolling(window=6).apply(trend_consistency, raw=False)
    trend_strength = (data['close'] - data['close'].shift(5)) / (data['high'] - data['low']).rolling(window=6).sum()
    trending_momentum = trend_consist * trend_strength
    
    # Mean-Reverting Regime Momentum
    reversion_signal = (data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    vol_compression = (data['high'] - data['low']) / ((data['high'] - data['low']).shift(1).rolling(window=4).mean())
    mean_reverting_momentum = reversion_signal * vol_compression
    
    # Quiet Regime Momentum
    micro_pressure = ((data['close'] - data['open']) / (data['high'] - data['low'])) * (data['volume'] / data['volume'].rolling(window=5).mean())
    amount_pressure = ((data['close'] - data['open']) * (data['amount'] / data['volume'])) / (data['high'] - data['low'])
    quiet_momentum = micro_pressure * amount_pressure
    
    # Core Momentum Selection
    core_momentum = pd.Series(index=data.index, dtype=float)
    core_momentum[explosive_cond] = explosive_momentum[explosive_cond]
    core_momentum[trending_cond] = trending_momentum[trending_cond]
    core_momentum[mean_reverting_cond] = mean_reverting_momentum[mean_reverting_cond]
    core_momentum[quiet_cond] = quiet_momentum[quiet_cond]
    core_momentum = core_momentum.fillna(0)
    
    # Microstructure Enhancement
    regime_enhanced = pd.Series(index=data.index, dtype=float)
    regime_enhanced[price_driven] = core_momentum[price_driven] * data['price_efficiency'][price_driven]
    regime_enhanced[volume_driven] = core_momentum[volume_driven] * data['volume_efficiency'][volume_driven]
    regime_enhanced[amount_driven] = core_momentum[amount_driven] * data['amount_efficiency'][amount_driven]
    regime_enhanced = regime_enhanced.fillna(core_momentum)
    
    # Multi-Dimensional Microstructure Pressure
    upper_rejection = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'])
    lower_support = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'])
    pressure_asymmetry = upper_rejection - lower_support
    
    volume_pressure = (data['volume'] / data['volume'].rolling(window=5).mean()) * pressure_asymmetry
    amount_pressure_val = ((data['amount'] / data['volume']) / (data['amount'] / data['volume']).rolling(window=5).mean()) * pressure_asymmetry
    pressure_combined = volume_pressure * amount_pressure_val
    
    recent_pressure = pressure_asymmetry * ((data['close'] - data['open']) / (data['high'] - data['low']))
    pressure_persistence = pressure_asymmetry.rolling(window=3).mean()
    time_weighted_pressure = recent_pressure * pressure_persistence
    
    # Volume-Amount Fractal Dynamics
    volume_momentum = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    
    def volume_trend_func(series):
        signs = np.sign(series.diff())
        current_sign = signs.iloc[-1]
        window_signs = signs.iloc[-4:-1] if len(signs) >= 4 else signs.iloc[:-1]
        return (window_signs == current_sign).sum() / 4
    
    volume_trend = data['volume'].rolling(window=4).apply(volume_trend_func, raw=False)
    volume_fractal = volume_momentum * volume_trend
    
    amount_momentum = (data['amount'] - data['amount'].shift(1)) / data['amount'].shift(1)
    amount_efficiency_val = abs(data['close'] - data['open']) / (data['amount'] / data['volume'] + 0.0001)
    amount_fractal = amount_momentum * amount_efficiency_val
    
    convergence_signal = volume_fractal * amount_fractal
    divergence_detection = np.sign(volume_fractal) != np.sign(amount_fractal)
    volume_amount_adjusted = convergence_signal * np.where(divergence_detection, -1, 1)
    
    # Regime-Adaptive Signal Integration
    pressure_enhanced = regime_enhanced * time_weighted_pressure
    volume_amount_alignment = pressure_enhanced * volume_amount_adjusted
    
    # Multi-Regime Alpha Core
    multi_regime_core = volume_amount_alignment * pressure_combined
    
    # Volatility Context Adjustment
    volatility_multiplier = pd.Series(1.0, index=data.index)
    volatility_multiplier[explosive_cond] = 1.5
    volatility_multiplier[trending_cond] = 1.2
    volatility_multiplier[mean_reverting_cond] = 0.8
    volatility_multiplier[quiet_cond] = 0.5
    
    adjusted_core = multi_regime_core * volatility_multiplier
    
    # Microstructure Efficiency Filter
    efficiency_score = (data['price_efficiency'] + data['volume_efficiency'] + data['amount_efficiency']) / 3
    filtered_alpha = adjusted_core * efficiency_score
    
    # Final Alpha Synthesis
    def trend_confirmation_func(series, alpha_sign):
        signs = np.sign(series.diff())
        window_signs = signs.iloc[-3:] if len(signs) >= 3 else signs
        return (window_signs == alpha_sign).sum() / 3
    
    trend_confirmation = pd.Series(index=data.index, dtype=float)
    for idx in data.index:
        if idx in filtered_alpha.index:
            alpha_sign = np.sign(filtered_alpha.loc[idx])
            if not np.isnan(alpha_sign):
                window_data = data['close'].loc[:idx].tail(4)
                if len(window_data) >= 3:
                    confirmation = trend_confirmation_func(window_data, alpha_sign)
                    trend_confirmation.loc[idx] = confirmation
    
    trend_confirmation = trend_confirmation.fillna(0)
    final_alpha = filtered_alpha * (1 + trend_confirmation) * volume_amount_adjusted
    
    return final_alpha
