import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price differences and returns
    df['close_shift2'] = df['close'].shift(2)
    df['close_shift3'] = df['close'].shift(3)
    df['close_shift8'] = df['close'].shift(8)
    df['close_shift21'] = df['close'].shift(21)
    df['volume_shift3'] = df['volume'].shift(3)
    df['amount_shift5'] = df['amount'].shift(5)
    df['close_shift1'] = df['close'].shift(1)
    
    # Fractal Efficiency Analysis
    # Micro-efficiency (3-day)
    df['high_3d'] = df['high'].rolling(window=3, min_periods=3).max()
    df['low_3d'] = df['low'].rolling(window=3, min_periods=3).min()
    df['micro_efficiency'] = (df['close'] - df['close_shift2']) / (df['high_3d'] - df['low_3d'])
    
    # Meso-efficiency (8-day)
    df['high_8d'] = df['high'].rolling(window=8, min_periods=8).max()
    df['low_8d'] = df['low'].rolling(window=8, min_periods=8).min()
    df['meso_efficiency'] = (df['close'] - df['close_shift8']) / (df['high_8d'] - df['low_8d'])
    
    # Macro-efficiency (21-day)
    df['high_21d'] = df['high'].rolling(window=21, min_periods=21).max()
    df['low_21d'] = df['low'].rolling(window=21, min_periods=21).min()
    df['macro_efficiency'] = (df['close'] - df['close_shift21']) / (df['high_21d'] - df['low_21d'])
    
    # Intraday Efficiency
    df['intraday_efficiency'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Gap Efficiency
    df['gap_efficiency'] = (df['high'] - df['open']) / (abs(df['open'] - df['close_shift1']) + 1e-8)
    
    # Fractal Efficiency Coherence
    df['fractal_coherence'] = df['micro_efficiency'] * df['meso_efficiency'] * df['macro_efficiency']
    
    # Momentum-Volume Elasticity Framework
    # Price Momentum
    df['price_momentum'] = df['close'] / df['close_shift3'] - 1
    
    # Volume Momentum
    df['volume_momentum'] = df['volume'] / df['volume_shift3'] - 1
    
    # Liquidity Momentum
    df['liquidity_momentum'] = df['amount'] / df['amount_shift5'] - 1
    
    # Volume Efficiency
    df['volume_efficiency'] = df['volume'] / (df['high'] - df['low'] + 1e-8)
    
    # Volume-Efficiency Elasticity
    df['volume_efficiency_elasticity'] = (df['micro_efficiency'] - df['meso_efficiency']) * df['volume_momentum']
    
    # Core Momentum Divergence
    df['core_momentum_divergence'] = df['price_momentum'] * df['volume_momentum'] * df['liquidity_momentum']
    
    # Intraday Fractal Dynamics
    # Price Fractal Position
    df['price_fractal_position'] = (df['close'] - df['low_8d']) / (df['high_8d'] - df['low_8d'] + 1e-8)
    
    # Session Strength Asymmetry
    df['session_strength_asymmetry'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Overnight Fractal Gap
    df['overnight_fractal_gap'] = df['open'] / df['close_shift1'] - 1
    
    # Gap-Efficiency Coupling
    df['gap_efficiency_coupling'] = df['overnight_fractal_gap'] * df['micro_efficiency']
    
    # Intraday Momentum Persistence
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['intraday_sign'] = np.sign(df['intraday_return'])
    df['intraday_momentum_persistence'] = df['intraday_sign'].rolling(window=3, min_periods=3).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and x.iloc[i] != 0]), raw=False
    )
    
    # Dynamic Regime Classification
    # Volatility Regime
    df['returns'] = df['close'].pct_change()
    df['volatility_regime'] = df['returns'].rolling(window=5, min_periods=5).std()
    
    # Efficiency-Volume Correlation
    df['efficiency_volume_corr'] = df['intraday_efficiency'].rolling(window=5, min_periods=5).corr(df['volume'])
    
    # Volume Trend Persistence
    df['volume_trend'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    df['volume_trend_persistence'] = df['volume_trend'].rolling(window=3, min_periods=3).sum()
    
    # Regime Stability
    df['volatility_trend'] = (df['volatility_regime'] > df['volatility_regime'].shift(1)).astype(int)
    df['regime_stability'] = df['volatility_trend'].rolling(window=3, min_periods=3).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1]]), raw=False
    )
    
    # Regime-Adaptive Signal Construction
    # High-Intensity Regime Signals
    df['volume_momentum_resonance'] = df['fractal_coherence'] * df['liquidity_momentum']
    df['gap_strength_alignment'] = df['gap_efficiency_coupling'] * df['session_strength_asymmetry']
    df['efficiency_weighted_divergence'] = df['core_momentum_divergence'] * df['intraday_efficiency']
    
    # Low-Intensity Regime Signals
    df['coherence_based_reversal'] = df['fractal_coherence'] * df['volume_trend_persistence']
    df['efficiency_expansion'] = df['micro_efficiency'] * df['fractal_coherence']
    df['range_expansion_signal'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)
    
    # Regime Transition Dynamics
    df['volume_momentum_accumulation'] = df['volume_momentum'].rolling(window=2, min_periods=2).sum()
    df['regime_shift_momentum'] = df['volume_momentum_accumulation'] * df['fractal_coherence']
    
    # Multi-Scale Signal Fusion
    # Core Fractal Momentum
    df['core_fractal_momentum'] = df['volume_momentum_resonance'] * df['gap_strength_alignment']
    
    # Elasticity-Modulated Signals
    df['high_intensity_signal'] = df['core_fractal_momentum'] * df['volume_efficiency_elasticity']
    df['low_intensity_signal'] = df['coherence_based_reversal'] * (1 - df['volume_efficiency_elasticity'])
    df['transition_signal'] = df['regime_shift_momentum'] * df['volume_momentum_accumulation']
    
    # Volatility Adjustment
    df['volatility_adjustment'] = 1 / (df['volatility_regime'] + 1e-8)
    
    # Persistence and Quality Refinement
    # Signal Continuity
    df['core_fractal_momentum_positive'] = (df['core_fractal_momentum'] > 0).astype(int)
    df['signal_continuity'] = df['core_fractal_momentum_positive'].rolling(window=3, min_periods=3).sum()
    
    # Recent Efficiency Volatility
    df['recent_efficiency_volatility'] = df['micro_efficiency'].rolling(window=4, min_periods=4).std()
    
    # Enhanced Fractal Signal
    df['enhanced_fractal_signal'] = (
        df['high_intensity_signal'] + df['low_intensity_signal'] + df['transition_signal']
    ) * df['signal_continuity']
    
    # Quality-Calibrated Signals
    df['quality_calibrated_signals'] = df['enhanced_fractal_signal'] / (df['recent_efficiency_volatility'] + 1e-8)
    
    # Liquidity Filter
    df['liquidity_filter'] = df['volume_momentum'] * df['range_expansion_signal']
    
    # Final Alpha Synthesis
    # Unified Signal
    df['unified_signal'] = df['quality_calibrated_signals'] * df['liquidity_filter'] * df['gap_efficiency_coupling']
    
    # Regime Adaptation
    df['regime_adaptation'] = df['unified_signal'] * df['volatility_adjustment'] * df['efficiency_volume_corr']
    
    # Mean Reversion Bias
    df['mean_reversion_bias'] = -(df['price_momentum'] * (df['micro_efficiency'] / (df['meso_efficiency'] + 1e-8) - 1))
    
    # Fractal Position Scaling
    df['fractal_position_scaling'] = 1 - abs(df['price_fractal_position'] - 0.5)
    
    # Volume-Convergence Weighting
    df['volume_convergence_weighting'] = df['regime_adaptation'] * df['volume_trend_persistence']
    
    # Final Alpha
    df['final_alpha'] = df['volume_convergence_weighting'] * (1 + df['mean_reversion_bias']) * df['fractal_position_scaling']
    
    # Return the final alpha factor
    result = df['final_alpha']
    
    # Clean up intermediate columns
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result
