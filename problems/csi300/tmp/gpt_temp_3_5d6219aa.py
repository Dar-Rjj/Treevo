import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Short-Term Reversal-Pressure Patterns (1-5 days)
    # Price Reversal Intensity
    data['price_reversal'] = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['volume_reversal'] = (data['volume'] / data['volume'].shift(1)) * (data['close'] - data['close'].shift(1))
    data['reversal_intensity'] = data['price_reversal'] * data['volume_reversal']
    
    # Gap-Pressure Asymmetry
    data['gap_magnitude'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['pressure_direction'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['amount_pressure'] = data['amount'] / (data['high'] - data['low'])
    data['gap_pressure_asymmetry'] = data['gap_magnitude'] * data['pressure_direction'] * data['amount_pressure']
    
    # Range Compression-Expansion
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['compression_expansion_signal'] = data['range_compression'] * data['close_position']
    
    # Medium-Term Regime Transition Signals (5-20 days)
    # Momentum Regime Divergence
    data['short_term_reversal'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_trend'] = data['close'] / data['close'].shift(20) - 1
    data['regime_divergence'] = data['short_term_reversal'] * data['medium_term_trend']
    
    # Volume Regime Persistence
    data['volume_regime'] = data['volume'] / data['volume'].shift(5)
    
    # Calculate volume consistency (rolling count of volume changes < 20%)
    volume_change = abs(data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    data['volume_consistency'] = volume_change.rolling(window=5).apply(
        lambda x: (x < 0.2).sum(), raw=False
    )
    data['regime_persistence'] = data['volume_regime'] * data['volume_consistency']
    
    # Efficiency Regime Transition
    data['efficiency_current'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['efficiency_lag5'] = (data['close'].shift(5) - data['open'].shift(5)) / (data['high'].shift(5) - data['low'].shift(5))
    data['efficiency_change'] = data['efficiency_current'] - data['efficiency_lag5']
    data['amount_regime'] = data['amount'] / data['amount'].shift(5)
    data['efficiency_transition'] = data['efficiency_change'] * data['amount_regime']
    
    # Long-Term Structural Pressure Cycles (15-30 days)
    # Structural Pressure Memory
    data['historical_pressure'] = (data['close'] - data['low'].shift(10)) / (data['high'].shift(10) - data['low'].shift(10))
    data['current_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['pressure_memory'] = data['historical_pressure'] * data['current_pressure']
    
    # Liquidity Regime Cycles
    data['volume_cycle_phase'] = data['volume'] / data['volume'].shift(15)
    data['amount_cycle_phase'] = data['amount'] / data['amount'].shift(15)
    data['liquidity_cycle'] = data['volume_cycle_phase'] * data['amount_cycle_phase']
    
    # Regime Persistence Patterns
    data['efficiency_lag15'] = (data['close'].shift(15) - data['open'].shift(15)) / (data['high'].shift(15) - data['low'].shift(15))
    data['price_regime_persistence'] = data['efficiency_current'] - data['efficiency_lag15']
    data['volume_regime_strength'] = (data['volume'] / data['volume'].shift(15)) * data['efficiency_current']
    data['long_term_regime_persistence'] = data['price_regime_persistence'] * data['volume_regime_strength']
    
    # Multi-Scale Regime Integration & Adaptation
    # Short-Medium Regime Alignment
    data['reversal_regime_sync'] = data['reversal_intensity'] * data['regime_divergence']
    data['gap_pressure_harmony'] = data['gap_pressure_asymmetry'] * data['efficiency_transition']
    data['compression_regime_fractal'] = data['compression_expansion_signal'] * data['regime_persistence']
    
    # Medium-Long Regime Convergence
    data['divergence_memory_link'] = data['regime_divergence'] * data['pressure_memory']
    data['transition_cycle_sync'] = data['efficiency_transition'] * data['liquidity_cycle']
    data['persistence_regime_alignment'] = data['long_term_regime_persistence'] * data['volume_regime_strength']
    
    # Volatility Regime Detection
    data['current_regime_volatility'] = (data['high'] - data['low']) / data['open']
    data['regime_volatility_base'] = (data['high'].shift(5) - data['low'].shift(5)) / data['open'].shift(5)
    data['volatility_regime'] = data['current_regime_volatility'] / data['regime_volatility_base']
    
    # Regime Confirmation System
    # Micro-Regime Patterns (1-3 days)
    data['opening_regime'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['intraday_regime'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['closing_regime'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Meso-Regime Strength (5-10 days)
    # Calculate regime consistency (rolling count of efficiency > 0.5)
    efficiency_series = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['regime_consistency'] = efficiency_series.rolling(window=10).apply(
        lambda x: (x > 0.5).sum(), raw=False
    )
    data['volume_regime_confirmation'] = (data['volume'] / data['volume'].shift(5)) * data['regime_consistency']
    data['efficiency_regime'] = efficiency_series.rolling(window=5).mean()
    
    # Macro-Regime Convergence (15-20 days)
    data['trend_regime'] = data['close'] / data['close'].shift(15) - 1
    data['liquidity_regime_strength'] = (data['volume'] / data['volume'].shift(15)) * (data['amount'] / data['amount'].shift(15))
    data['regime_convergence'] = data['trend_regime'] * data['liquidity_regime_strength']
    
    # Final Alpha Factor Generation
    # Core Reversal-Pressure Factor
    data['primary_regime_signal'] = data['reversal_regime_sync'] * data['gap_pressure_harmony'] * data['compression_regime_fractal']
    data['convergence_enhancement'] = data['divergence_memory_link'] * data['transition_cycle_sync'] * data['persistence_regime_alignment']
    data['core_factor'] = data['primary_regime_signal'] * data['convergence_enhancement']
    
    # Regime Confirmation & Volatility Adjustment
    data['regime_confirmation'] = data['volume_regime_confirmation'] * data['regime_convergence']
    data['volatility_adjustment'] = data['core_factor'] * data['volatility_regime']
    data['final_signal'] = data['volatility_adjustment'] * data['regime_confirmation']
    
    # Return the final alpha factor
    return data['final_signal']
