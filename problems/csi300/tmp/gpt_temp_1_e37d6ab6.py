import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Fractal Momentum Integration
    # Cross-Timeframe Fractal Momentum
    data['short_medium_fractal_ratio'] = (
        (data['close'] / data['close'].shift(3) - data['close'] / data['close'].shift(10)) / 
        (data['close'].shift(1) / data['close'].shift(4) - data['close'].shift(1) / data['close'].shift(11))
    )
    
    data['volatility_adjusted_fractal_momentum'] = (
        (data['close'] / data['close'].shift(5) - data['close'] / data['close'].shift(10)) * 
        (data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(3) - data['low'].shift(3))
    )
    
    data['fractal_momentum_efficiency'] = (
        data['volatility_adjusted_fractal_momentum'] / 
        ((data['high'] - data['low']) / data['close'].shift(1))
    )
    
    # Fractal Volume-Momentum Dynamics
    data['volume_fractal_regime_shift'] = (
        (data['volume'] / data['volume'].shift(3) - data['volume'] / data['volume'].shift(8)) * 
        (data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(4) - data['low'].shift(4))
    )
    
    data['volume_fractal_compression'] = (
        (data['volume'] / (data['high'] - data['low'])) / 
        (data['volume'].shift(2) / (data['high'].shift(2) - data['low'].shift(2)))
    )
    
    data['fractal_volume_momentum_synergy'] = (
        data['volatility_adjusted_fractal_momentum'] * data['volume_fractal_regime_shift']
    )
    
    # Fractal Momentum Validation
    def calculate_persistence(series, window=3):
        return series.rolling(window).apply(
            lambda x: np.mean(np.sign(x) == np.sign(x.shift(1))), raw=False
        )
    
    data['fractal_momentum_persistence'] = calculate_persistence(data['volatility_adjusted_fractal_momentum'])
    data['fractal_volume_momentum_alignment'] = (
        np.sign(data['volatility_adjusted_fractal_momentum']) * np.sign(data['volume_fractal_regime_shift'])
    )
    data['fractal_momentum_quality'] = (
        data['fractal_momentum_persistence'] * data['fractal_volume_momentum_alignment']
    )
    
    # Asymmetric Fractal Microstructure Patterns
    data['fractal_upside_rejection'] = data['high'] - np.maximum(data['open'], data['close'])
    data['fractal_downside_rejection'] = np.minimum(data['open'], data['close']) - data['low']
    data['net_fractal_rejection'] = data['fractal_upside_rejection'] - data['fractal_downside_rejection']
    data['fractal_bid_ask_pressure'] = data['net_fractal_rejection'] * (data['volume'] / data['volume'].shift(1))
    
    # Fractal Efficiency Microstructure
    data['intraday_fractal_efficiency'] = (
        np.abs(data['close'] - data['open']) / (data['high'] - data['low']) * 
        (data['volume'] / data['volume'].shift(2))
    )
    
    data['fractal_efficiency_momentum'] = (
        (data['intraday_fractal_efficiency'] / data['intraday_fractal_efficiency'].shift(2) - 1) * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    data['fractal_closing_pressure'] = (
        (data['close'] - data['open']) / (data['high'] - data['low']) * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    data['fractal_microstructure_efficiency'] = (
        data['fractal_bid_ask_pressure'] * data['intraday_fractal_efficiency']
    )
    
    # Fractal Microstructure Validation
    data['fractal_rejection_consistency'] = calculate_persistence(data['net_fractal_rejection'])
    data['fractal_efficiency_alignment'] = (
        np.sign(data['fractal_bid_ask_pressure']) * np.sign(data['intraday_fractal_efficiency'])
    )
    data['fractal_microstructure_quality'] = (
        data['fractal_rejection_consistency'] * data['fractal_efficiency_alignment']
    )
    
    # Fractal Breakout Dynamics
    data['fractal_upside_breakout'] = data['high'] / data['high'].shift(1) - 1
    data['fractal_downside_breakout'] = data['low'] / data['low'].shift(1) - 1
    data['fractal_breakout_asymmetry'] = data['fractal_upside_breakout'] - data['fractal_downside_breakout']
    
    data['volume_confirmed_fractal_breakout'] = (
        (data['volume'] / data['volume'].shift(2)) * 
        (data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(4) - data['low'].shift(4))
    )
    data.loc[~((data['high'] > data['high'].shift(1)) & (data['close'] > data['close'].shift(1))), 
             'volume_confirmed_fractal_breakout'] = 0
    
    # Fractal Range Expansion Patterns
    data['fractal_range_expansion_flow'] = (
        (data['high'] / data['low'] - data['high'].shift(1) / data['low'].shift(1)) * 
        (data['volume'] / data['volume'].shift(1))
    )
    
    data['fractal_volatility_breakout'] = (
        (data['high'] - data['high'].shift(1)) * (data['high'] - data['low'])
    )
    
    data['fractal_breakout_quality'] = (
        data['volume_confirmed_fractal_breakout'] * data['fractal_range_expansion_flow']
    )
    
    # Fractal Breakout Validation
    data['fractal_breakout_persistence'] = calculate_persistence(data['fractal_breakout_asymmetry'])
    data['fractal_range_alignment'] = (
        np.sign(data['volume_confirmed_fractal_breakout']) * np.sign(data['fractal_range_expansion_flow'])
    )
    data['fractal_breakout_quality_score'] = (
        data['fractal_breakout_persistence'] * data['fractal_range_alignment']
    )
    
    # Fractal Volatility-Regime Integration
    data['fractal_range_volatility'] = (
        (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    )
    
    data['fractal_efficiency_volatility'] = (
        np.abs(data['intraday_fractal_efficiency'] - data['intraday_fractal_efficiency'].shift(1)) * 
        (data['high'] - data['low'])
    )
    
    data['volume_fractal_volatility'] = (
        data['volume_fractal_compression'] * (data['high'] - data['low']) / 
        (data['high'].shift(1) - data['low'].shift(1))
    )
    
    # Fractal Volatility Patterns
    data['fractal_range_expansion'] = (data['fractal_range_volatility'] > 1.2).astype(float)
    data['fractal_range_contraction'] = (data['fractal_range_volatility'] < 0.8).astype(float)
    data['fractal_volatility_shift'] = data['fractal_range_volatility'] / data['fractal_range_volatility'].shift(3)
    data['fractal_mean_reversion'] = 1 - np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Fractal Volatility Enhancement
    data['fractal_volatility_adjustment'] = (
        (data['fractal_range_expansion'] - data['fractal_range_contraction']) * data['fractal_volatility_shift']
    )
    
    # Cross-Fractal Alpha Synthesis
    # Core Fractal Components
    data['momentum_fractal_component'] = (
        data['fractal_volume_momentum_synergy'] * data['fractal_momentum_quality']
    )
    
    data['microstructure_fractal_component'] = (
        data['fractal_microstructure_efficiency'] * data['fractal_microstructure_quality']
    )
    
    data['breakout_fractal_component'] = (
        data['fractal_breakout_quality'] * data['fractal_breakout_quality_score']
    )
    
    data['volatility_fractal_component'] = (
        data['fractal_volatility_adjustment'] * data['fractal_mean_reversion']
    )
    
    # Fractal Regime Weighting
    data['momentum_regime_weight'] = np.where(
        data['fractal_range_expansion'] == 1, 1.3, 1.0
    )
    
    data['microstructure_regime_weight'] = np.where(
        data['fractal_range_contraction'] == 1, 1.0, 1.3
    )
    
    data['breakout_regime_weight'] = np.where(
        data['volume_confirmed_fractal_breakout'] > 0, 1.3, 0.8
    )
    
    data['volatility_regime_weight'] = (
        data['fractal_volatility_shift'] * data['fractal_mean_reversion']
    )
    
    # Final Cross-Fractal Alpha
    data['primary_cross_fractal_factor'] = (
        data['momentum_fractal_component'] * data['momentum_regime_weight']
    )
    
    data['secondary_cross_fractal_factor'] = (
        data['microstructure_fractal_component'] * data['microstructure_regime_weight']
    )
    
    data['tertiary_cross_fractal_factor'] = (
        data['breakout_fractal_component'] * data['breakout_regime_weight']
    )
    
    data['quaternary_cross_fractal_factor'] = (
        data['volatility_fractal_component'] * data['volatility_regime_weight']
    )
    
    # Composite Cross-Fractal Alpha
    data['composite_cross_fractal_alpha'] = (
        data['primary_cross_fractal_factor'] * 
        data['secondary_cross_fractal_factor'] * 
        data['tertiary_cross_fractal_factor'] * 
        data['quaternary_cross_fractal_factor']
    )
    
    return data['composite_cross_fractal_alpha']
