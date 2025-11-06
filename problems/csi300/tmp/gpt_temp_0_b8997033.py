import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Multi-Scale Asymmetric Momentum Decay
    # Micro Asymmetric Momentum
    data['micro_asym_momentum'] = ((data['high'] - data['open']) - (data['open'] - data['low'])) / (data['high'] - data['low'] + epsilon)
    
    # Meso Asymmetric Momentum
    data['high_2d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['low_2d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['meso_asym_momentum'] = ((data['high_2d'] - data['open']) - (data['open'] - data['low_2d'])) / (data['high_2d'] - data['low_2d'] + epsilon)
    
    # Macro Asymmetric Momentum
    data['high_5d'] = data['high'].rolling(window=6, min_periods=1).max()
    data['low_5d'] = data['low'].rolling(window=6, min_periods=1).min()
    data['macro_asym_momentum'] = ((data['high_5d'] - data['open']) - (data['open'] - data['low_5d'])) / (data['high_5d'] - data['low_5d'] + epsilon)
    
    # Momentum Decay Rate
    data['momentum_decay_rate'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(4) - data['close'].shift(5) + epsilon)
    
    # Asymmetric Momentum Decay
    data['asym_momentum_decay'] = (data['micro_asym_momentum'] + data['meso_asym_momentum'] + data['macro_asym_momentum']) * data['momentum_decay_rate']
    
    # Efficiency-Based Regime Detection
    # Daily Efficiency
    data['daily_efficiency'] = (data['close'] - data['open']) / np.maximum(
        data['high'] - data['low'], 
        np.maximum(np.abs(data['high'] - data['close'].shift(1)), np.abs(data['low'] - data['close'].shift(1)))
    )
    
    # Gap Efficiency
    data['gap_efficiency'] = (data['close'] - data['open']) / (np.abs(data['open'] - data['close'].shift(1)) + epsilon)
    
    # Volatility Breakout
    data['volatility_breakout'] = (data['high'] - data['low']) / (data['high'].shift(4) - data['low'].shift(4) + epsilon)
    
    # Volume Regime Change
    data['volume_regime_change'] = data['volume'] / (data['volume'].shift(4) + epsilon)
    
    # Regime Transition Signal
    data['regime_transition_signal'] = np.sign(data['daily_efficiency']) * np.sign(data['volatility_breakout'] - 1) * np.sign(data['volume_regime_change'] - 1)
    
    # Volume-Price Acceleration Dynamics
    # Micro Volume Intensity
    data['micro_volume_intensity'] = data['volume'] / (data['volume'].rolling(window=3, min_periods=1).mean() + epsilon)
    
    # Meso Volume Intensity
    data['meso_volume_intensity'] = data['volume'] / (data['volume'].rolling(window=6, min_periods=1).mean() + epsilon)
    
    # Decay Acceleration
    data['decay_acceleration'] = data['momentum_decay_rate'] - data['momentum_decay_rate'].shift(1)
    
    # Volume Acceleration
    data['volume_acceleration'] = (data['micro_volume_intensity'] - data['meso_volume_intensity']) * data['decay_acceleration']
    
    # Acceleration Confirmation
    data['acceleration_confirmation'] = np.where(
        np.sign(data['volume_acceleration']) == np.sign(data['decay_acceleration']),
        np.abs(data['volume_acceleration']),
        -np.abs(data['volume_acceleration'])
    )
    
    # Intraday Pressure Efficiency
    # Morning Pressure
    data['morning_pressure'] = (data['high'] - data['open']) * data['amount'] - (data['open'] - data['low']) * data['amount']
    
    # Afternoon Pressure
    data['afternoon_pressure'] = (data['close'] - data['low']) * data['amount'] - (data['high'] - data['close']) * data['amount']
    
    # Pressure Asymmetry
    data['pressure_asymmetry'] = data['morning_pressure'] * data['afternoon_pressure'] * np.sign(data['morning_pressure'] - data['afternoon_pressure'])
    
    # Intraday Efficiency
    data['intraday_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    
    # Pressure Efficiency Alignment
    data['pressure_efficiency_alignment'] = data['pressure_asymmetry'] * data['intraday_efficiency'] * np.sign(data['daily_efficiency'])
    
    # Adaptive Core Construction
    # High Volatility Core
    data['high_vol_core'] = data['asym_momentum_decay'] * np.abs(data['volatility_breakout'] - 1) * data['acceleration_confirmation']
    
    # Low Volatility Core
    data['low_vol_core'] = (data['daily_efficiency'] * data['gap_efficiency']) * data['volume_regime_change'] * data['pressure_efficiency_alignment']
    
    # Regime-Adaptive Core
    data['regime_adaptive_core'] = np.where(data['volatility_breakout'] > 1, data['high_vol_core'], data['low_vol_core'])
    
    # Quality and Persistence Enhancement
    # Efficiency Consistency
    data['efficiency_sign'] = np.sign(data['daily_efficiency'])
    data['efficiency_consistency'] = data['efficiency_sign'].rolling(window=6, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1
    )
    
    # Asymmetry Persistence
    data['micro_meso_sign_match'] = (np.sign(data['micro_asym_momentum']) == np.sign(data['meso_asym_momentum'])).astype(int)
    data['asymmetry_persistence'] = data['micro_meso_sign_match'].rolling(window=3, min_periods=1).mean()
    
    # Acceleration Persistence
    data['decay_accel_sign'] = np.sign(data['decay_acceleration'])
    data['acceleration_persistence'] = data['decay_accel_sign'].rolling(window=3, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1
    ) / 3
    
    # Quality Multiplier
    data['quality_multiplier'] = data['efficiency_consistency'] * data['asymmetry_persistence'] * data['acceleration_persistence']
    
    # Convergence and Divergence Dynamics
    # Momentum-Efficiency Divergence
    data['momentum_efficiency_divergence'] = data['asym_momentum_decay'] - data['daily_efficiency']
    
    # Volume-Pressure Correlation
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['volume_pressure_corr'] = data['price_efficiency'].rolling(window=5, min_periods=1).corr(data['volume'])
    
    # Convergence Signal
    data['convergence_signal'] = np.sign(data['momentum_efficiency_divergence']) * np.sign(data['volume_pressure_corr']) * data['regime_transition_signal']
    
    # Liquidity-Momentum Integration
    # Liquidity Efficiency
    data['close_open_volume'] = (data['close'] - data['open']) * data['volume']
    data['avg_close_open_volume'] = data['close_open_volume'].rolling(window=6, min_periods=1).mean()
    data['liquidity_efficiency'] = data['close_open_volume'] / (data['avg_close_open_volume'] + epsilon)
    
    # Trade Impact Efficiency
    data['trade_impact_efficiency'] = data['amount'] / (data['close_open_volume'] + epsilon)
    
    # Momentum Liquidity Factor
    data['momentum_liquidity_factor'] = data['asym_momentum_decay'] * data['liquidity_efficiency'] * data['trade_impact_efficiency']
    
    # Final Alpha Construction
    # Core Factor
    data['core_factor'] = data['regime_adaptive_core'] * data['quality_multiplier']
    
    # Dynamic Enhancement
    data['dynamic_enhancement'] = data['core_factor'] * data['convergence_signal'] * data['acceleration_confirmation']
    
    # Liquidity Enhancement
    data['liquidity_enhancement'] = data['dynamic_enhancement'] * data['momentum_liquidity_factor']
    
    # Final Alpha
    data['final_alpha'] = data['core_factor'] * data['liquidity_enhancement']
    
    return data['final_alpha']
