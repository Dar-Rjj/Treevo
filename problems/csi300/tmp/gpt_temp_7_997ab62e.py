import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Asymmetric Volume-Price Framework
    # Intraday Asymmetry Components
    morning_fractal = (df['high'] - df['open']) / (df['open'] - df['low']).replace(0, np.nan)
    afternoon_fractal = (df['close'] - df['low']) / (df['high'] - df['close']).replace(0, np.nan)
    fractal_shift = morning_fractal - afternoon_fractal
    
    # Volume Distribution Asymmetry
    opening_vol_conc = df['volume'] * (df['open'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    closing_vol_conc = df['volume'] * (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    vol_dist_asymmetry = opening_vol_conc - closing_vol_conc
    
    # Multi-Timeframe Momentum Divergence
    # Short-term Momentum (3-day)
    fractal_momentum = fractal_shift - fractal_shift.shift(2)
    
    def rolling_corr_3d(vol, close):
        return vol.rolling(3).corr(close)
    
    vol_price_corr = rolling_corr_3d(df['volume'], df['close'])
    efficiency_momentum = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Medium-term Momentum (6-day)
    fractal_trend = fractal_shift - fractal_shift.shift(5)
    
    def volume_persistence_score(vol):
        result = []
        for i in range(len(vol)):
            if i >= 6:
                window = vol.iloc[i-5:i+1]
                prev_window = vol.iloc[i-6:i]
                if len(window) == 6 and len(prev_window) == 5:
                    signs = np.sign(window.values[1:] - prev_window.values)
                    score = np.sum(signs) / 5
                    result.append(score)
                else:
                    result.append(np.nan)
            else:
                result.append(np.nan)
        return pd.Series(result, index=vol.index)
    
    vol_persistence = volume_persistence_score(df['volume'])
    
    def efficiency_trend(open_p, high, low, close):
        result = []
        for i in range(len(close)):
            if i >= 5:
                window_eff = (close.iloc[i-5:i+1] - open_p.iloc[i-5:i+1]) / (high.iloc[i-5:i+1] - low.iloc[i-5:i+1]).replace(0, np.nan)
                result.append(window_eff.mean())
            else:
                result.append(np.nan)
        return pd.Series(result, index=close.index)
    
    eff_trend = efficiency_trend(df['open'], df['high'], df['low'], df['close'])
    
    # Momentum Divergence Signal
    short_term_score = fractal_momentum * vol_price_corr * efficiency_momentum
    medium_term_score = fractal_trend * vol_persistence * eff_trend
    divergence_factor = short_term_score - medium_term_score
    
    # Market Microstructure Quality Assessment
    price_impact_eff = df['amount'] / (df['volume'] * (df['high'] - df['low']).replace(0, np.nan))
    
    def trade_size_dist(amount):
        return amount / amount.rolling(2).mean().shift(1)
    
    trade_size_distribution = trade_size_dist(df['amount'])
    execution_quality = (df['close'] - df['open']) / (df['amount'] / df['volume']).replace(0, np.nan)
    
    opening_pressure = (df['open'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1)).replace(0, np.nan)
    closing_pressure = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    flow_imbalance = opening_pressure - closing_pressure
    
    # Regime-Based Asymmetric Integration
    absolute_efficiency = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    vol_acceleration = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3).replace(0, np.nan)
    
    # Volume Concentration
    vol_concentration = opening_vol_conc / (opening_vol_conc + closing_vol_conc).replace(0, np.nan)
    
    # Regime signals
    high_eff_mask = (absolute_efficiency > 0.7) & (vol_acceleration > 0)
    momentum_quality = fractal_shift * vol_concentration
    flow_alignment = flow_imbalance * vol_dist_asymmetry
    high_regime_signal = momentum_quality * flow_alignment * trade_size_distribution
    
    reversal_quality = fractal_shift * vol_concentration
    flow_contrarian = -flow_imbalance * vol_dist_asymmetry
    low_regime_signal = reversal_quality * flow_contrarian * execution_quality
    
    # Combine regime signals
    regime_signal = np.where(high_eff_mask, high_regime_signal, low_regime_signal)
    
    # Regime Transition Detection
    efficiency_change = absolute_efficiency - absolute_efficiency.shift(3)
    vol_regime_shift = vol_acceleration - vol_acceleration.shift(3)
    transition_factor = efficiency_change * vol_regime_shift
    
    # Hierarchical Alpha Synthesis
    core_asymmetry_signal = regime_signal
    divergence_enhanced = core_asymmetry_signal * divergence_factor
    quality_adjusted = divergence_enhanced * price_impact_eff
    transition_adjusted = quality_adjusted * transition_factor
    final_alpha = transition_adjusted * np.sign(core_asymmetry_signal)
    
    return pd.Series(final_alpha, index=df.index)
