import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 10:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        current = df.iloc[i]
        prev_data = df.iloc[:i+1]  # Only use current and past data
        
        # Multi-Scale Gap Microstructure
        # True range gap efficiency
        true_range = max(
            current['high'] - current['low'],
            abs(current['high'] - df.iloc[i-1]['close']),
            abs(current['low'] - df.iloc[i-1]['close'])
        )
        true_range_gap_efficiency = (current['close'] - current['open']) / true_range if true_range != 0 else 0
        
        # Gap-microstructure alignment
        gap_direction = np.sign(current['open'] - df.iloc[i-1]['close'])
        intraday_direction = np.sign(current['close'] - current['open'])
        gap_microstructure_alignment = gap_direction * intraday_direction * current['volume']
        
        # Fractal gap efficiency
        fractal_gap_efficiency = true_range_gap_efficiency * gap_microstructure_alignment
        
        # Asymmetric Path Efficiency
        if current['close'] > current['open']:
            up_day_efficiency = (current['close'] - current['open']) / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
            down_day_efficiency = 0
        elif current['close'] < current['open']:
            up_day_efficiency = 0
            down_day_efficiency = (current['close'] - current['open']) / (current['high'] - current['low']) if (current['high'] - current['low']) != 0 else 0
        else:
            up_day_efficiency = 0
            down_day_efficiency = 0
        
        # Efficiency regime asymmetry
        efficiency_regime_asymmetry = up_day_efficiency / down_day_efficiency if down_day_efficiency != 0 else 0
        
        # Price-Volume Fractal Structure
        # Micro-fractal dimension
        micro_fractal_dim = np.log(current['high'] - current['low']) / np.log(current['volume']) if (current['high'] > current['low'] and current['volume'] > 0) else 0
        
        # Meso-fractal structure
        meso_fractal = abs(current['close'] - current['open']) / np.sqrt(current['high'] - current['low']) * current['volume'] if (current['high'] > current['low']) else 0
        
        # Fractal volume efficiency
        fractal_volume_efficiency = current['amount'] / (current['volume'] * (current['high'] - current['low'])) if (current['volume'] > 0 and (current['high'] - current['low']) > 0) else 0
        
        # Multi-Timeframe Fractal Alignment
        # Short-term price-volume fractal correlation (3-day window)
        if i >= 3:
            short_window = prev_data.iloc[max(0, i-2):i+1]
            price_returns = (short_window['close'] / short_window['close'].shift(1) - 1).dropna()
            volume_returns = (short_window['volume'] / short_window['volume'].shift(1) - 1).dropna()
            if len(price_returns) >= 2 and len(volume_returns) >= 2:
                short_term_fractal = price_returns.corr(volume_returns) if not (price_returns.isna().any() or volume_returns.isna().any()) else 0
            else:
                short_term_fractal = 0
        else:
            short_term_fractal = 0
        
        # Medium-term price-volume fractal correlation (5-day window)
        if i >= 5:
            medium_window = prev_data.iloc[max(0, i-4):i+1]
            price_returns_med = (medium_window['close'] / medium_window['close'].shift(3) - 1).dropna()
            volume_returns_med = (medium_window['volume'] / medium_window['volume'].shift(3) - 1).dropna()
            if len(price_returns_med) >= 2 and len(volume_returns_med) >= 2:
                medium_term_fractal = price_returns_med.corr(volume_returns_med) if not (price_returns_med.isna().any() or volume_returns_med.isna().any()) else 0
            else:
                medium_term_fractal = 0
        else:
            medium_term_fractal = 0
        
        # Fractal scale ratio
        fractal_scale_ratio = short_term_fractal / medium_term_fractal if medium_term_fractal != 0 else 0
        
        # Volatility Regime Analysis
        if current['close'] > current['open']:
            up_day_fractal_vol = (current['high'] - current['low']) / df.iloc[i-1]['close'] if df.iloc[i-1]['close'] != 0 else 0
            down_day_fractal_vol = 0
        elif current['close'] < current['open']:
            up_day_fractal_vol = 0
            down_day_fractal_vol = (current['high'] - current['low']) / df.iloc[i-1]['close'] if df.iloc[i-1]['close'] != 0 else 0
        else:
            up_day_fractal_vol = 0
            down_day_fractal_vol = 0
        
        # Volatility regime asymmetry
        volatility_regime_asymmetry = up_day_fractal_vol / down_day_fractal_vol if down_day_fractal_vol != 0 else 0
        
        # Regime Transition Dynamics
        bull_to_bear = 0
        bear_to_bull = 0
        
        if i >= 1:
            prev_day = df.iloc[i-1]
            if current['close'] < current['open'] and prev_day['close'] > prev_day['open']:
                bull_to_bear = (current['close'] - df.iloc[i-1]['close']) * current['volume']
            elif current['close'] > current['open'] and prev_day['close'] < prev_day['open']:
                bear_to_bull = (current['close'] - df.iloc[i-1]['close']) * current['volume']
        
        # Regime transition strength
        regime_transition_strength = bull_to_bear / bear_to_bull if bear_to_bull != 0 else 0
        
        # Multi-Timeframe Momentum Alignment
        short_term_momentum = current['close'] / df.iloc[i-3]['close'] - 1 if df.iloc[i-3]['close'] != 0 else 0
        medium_term_momentum = current['close'] / df.iloc[i-10]['close'] - 1 if i >= 10 and df.iloc[i-10]['close'] != 0 else 0
        momentum_timeframe_alignment = short_term_momentum * medium_term_momentum
        
        # Fractal Quality Validation
        volume_pressure_confirmation = 1 + abs((current['close'] - current['low']) * current['volume'] + (current['high'] - current['close']) * current['volume'])
        fractal_efficiency_validation = fractal_volume_efficiency * fractal_scale_ratio
        microstructure_quality_score = volume_pressure_confirmation * fractal_efficiency_validation
        
        # Composite Alpha Generation
        # Core Asymmetric Gap Factor
        core_gap_factor = fractal_gap_efficiency * efficiency_regime_asymmetry
        
        # Multi-Scale Fractal Momentum
        fractal_momentum = fractal_scale_ratio * momentum_timeframe_alignment
        
        # Regime-Adaptive Amplification
        regime_amplification = volatility_regime_asymmetry * regime_transition_strength
        
        # Final Alpha Output
        alpha_signal = (core_gap_factor + fractal_momentum) * (1 + regime_amplification) * microstructure_quality_score
        
        result.iloc[i] = alpha_signal
    
    return result
