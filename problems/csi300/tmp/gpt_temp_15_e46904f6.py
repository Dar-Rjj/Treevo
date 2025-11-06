import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Efficiency Dynamics
    # Ultra-Short Efficiency (2-day)
    data['true_range_2d'] = data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    data['price_movement_2d'] = abs(data['close'] - data['close'].shift(2))
    data['efficiency_2d'] = data['price_movement_2d'] / data['true_range_2d']
    
    # Short-Term Efficiency (5-day)
    data['true_range_5d'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    data['price_movement_5d'] = abs(data['close'] - data['close'].shift(5))
    data['efficiency_5d'] = data['price_movement_5d'] / data['true_range_5d']
    
    # Medium-Term Efficiency (20-day)
    data['true_range_20d'] = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    data['price_movement_20d'] = abs(data['close'] - data['close'].shift(20))
    data['efficiency_20d'] = data['price_movement_20d'] / data['true_range_20d']
    
    # Efficiency Decay Patterns
    data['efficiency_decay_ultra_short'] = data['efficiency_5d'] - data['efficiency_2d']
    data['efficiency_decay_short'] = data['efficiency_20d'] - data['efficiency_5d']
    data['efficiency_decay_acceleration'] = np.sign(data['efficiency_decay_ultra_short']) * np.sign(data['efficiency_decay_short'])
    
    # Fractal Microstructure Analysis
    # Price Pattern Analysis
    data['fractal_momentum'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) - data['close'].shift(2))
    data['price_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Short-term fractal (5-day)
    data['fractal_short'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) - data['close'].shift(10))
    # Medium-term fractal (20-day)
    data['fractal_medium'] = (data['close'] - data['close'].shift(20)) / (data['close'].shift(20) - data['close'].shift(40))
    data['fractal_divergence'] = abs(data['fractal_short'] - data['fractal_medium'])
    
    # Opening Auction Dynamics
    data['opening_gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['auction_imbalance'] = (data['open'] - data['low']) - (data['high'] - data['open'])
    data['opening_efficiency_fracture'] = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1))
    
    # Closing Momentum Fracture
    data['end_of_day_momentum'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] + data['low'])/2)
    data['closing_efficiency'] = abs(data['close'] - data['open']) / abs(data['close'] - data['close'].shift(1))
    data['session_completion_bias'] = data['end_of_day_momentum'] * data['closing_efficiency']
    
    # Volume Behavior Integration
    # Volume Persistence Analysis
    data['volume_persistence'] = (
        (data['volume'] > data['volume'].shift(1)).rolling(window=3).sum()
    )
    data['volume_exhaustion'] = np.where(data['volume'] > 2 * data['volume'].shift(1), -1, 1)
    
    # Volume Asymmetry Decay
    def calc_volume_asymmetry(series):
        up_volume = series[series > 0].sum()
        down_volume = series[series < 0].sum()
        return up_volume / down_volume if down_volume != 0 else 1
    
    price_changes = data['close'].diff()
    volume_signed = data['volume'] * np.sign(price_changes)
    data['volume_asymmetry_decay'] = volume_signed.rolling(window=5).apply(calc_volume_asymmetry, raw=False)
    
    # Volume-Pressure Dynamics
    data['opening_volume_pressure'] = data['volume'] / data['volume'].shift(1)
    data['intraday_volume_persistence'] = data['volume'] / ((data['volume'].shift(2) + data['volume'].shift(1))/2)
    data['volume_pressure_convergence'] = data['opening_volume_pressure'] - data['intraday_volume_persistence']
    
    # Volume-Momentum Alignment
    data['high_vol_volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['volume_momentum_alignment'] = data['high_vol_volume_efficiency'] * data['efficiency_decay_acceleration']
    
    # Volume directional fracture
    up_volume = volume_signed.rolling(window=5).apply(lambda x: x[x > 0].sum(), raw=False)
    down_volume = volume_signed.rolling(window=5).apply(lambda x: abs(x[x < 0].sum()), raw=False)
    data['volume_directional_fracture'] = up_volume - down_volume
    
    # Market Regime Integration
    # Volatility Regime Analysis
    data['volatility_regime'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Upside and Downside Volatility
    upside_returns = np.maximum(0, data['close'].diff())
    downside_returns = np.maximum(0, -data['close'].diff())
    data['upside_volatility'] = np.sqrt((upside_returns ** 2).rolling(window=5).sum())
    data['downside_volatility'] = np.sqrt((downside_returns ** 2).rolling(window=5).sum())
    data['volatility_asymmetry_ratio'] = data['upside_volatility'] / data['downside_volatility']
    
    # Efficiency Regime Detection
    data['range_utilization_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    
    conditions = [
        (data['range_utilization_efficiency'] > 0.7) & (data['efficiency_decay_acceleration'] < -0.1),
        (data['range_utilization_efficiency'] < 0.3) & (data['efficiency_decay_acceleration'] > 0.1)
    ]
    choices = [1.2, 0.8]
    data['efficiency_regime_multiplier'] = np.select(conditions, choices, default=1.0)
    
    # Microstructure Regime
    data['transaction_efficiency'] = abs(data['close'] - data['open']) / (data['volume'] * data['open'])
    data['microstructure_friction'] = 1 - data['range_utilization_efficiency']
    data['fractal_convergence'] = 1 - data['fractal_divergence']
    
    # Composite Fractal Alpha Construction
    # Core Efficiency Signal
    data['core_efficiency_base'] = data['efficiency_decay_acceleration'] * data['efficiency_decay_ultra_short']
    data['core_efficiency_volume_enhanced'] = data['core_efficiency_base'] * data['volume_asymmetry_decay']
    data['core_efficiency_regime_adjusted'] = data['core_efficiency_volume_enhanced'] * data['efficiency_regime_multiplier']
    
    # Microstructure Integration
    data['fractal_momentum_enhanced'] = data['fractal_momentum'] * data['volume_persistence']
    data['opening_dynamics'] = data['auction_imbalance'] * data['opening_efficiency_fracture']
    data['closing_bias'] = data['session_completion_bias'] * data['closing_efficiency']
    
    # Volume Behavior Factors
    data['volume_exhaustion_factor'] = data['volume_exhaustion'] * data['volume_pressure_convergence']
    data['volume_momentum_factor'] = data['volume_momentum_alignment'] * data['volume_directional_fracture']
    data['volume_regime_factor'] = data['volume_persistence'] * data['volatility_regime']
    
    # Alpha Construction
    data['base_signal'] = data['core_efficiency_regime_adjusted'] * data['fractal_momentum_enhanced']
    data['volume_adjusted_signal'] = data['base_signal'] * data['volume_exhaustion_factor']
    data['final_alpha'] = data['volume_adjusted_signal'] * data['efficiency_regime_multiplier']
    
    # Final Alpha Classification
    conditions = [
        data['final_alpha'] <= -0.6,
        (data['final_alpha'] > -0.6) & (data['final_alpha'] <= -0.3),
        (data['final_alpha'] > -0.3) & (data['final_alpha'] <= -0.1),
        (data['final_alpha'] > -0.1) & (data['final_alpha'] <= 0.1),
        (data['final_alpha'] > 0.1) & (data['final_alpha'] <= 0.3),
        (data['final_alpha'] > 0.3) & (data['final_alpha'] <= 0.6),
        data['final_alpha'] > 0.6
    ]
    
    choices = [-1.0, -0.45, -0.2, 0.0, 0.2, 0.45, 1.0]
    
    alpha_factor = pd.Series(
        np.select(conditions, choices, default=0.0),
        index=data.index,
        name='fractal_efficiency_momentum'
    )
    
    return alpha_factor
