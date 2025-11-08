import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate regime-adaptive alpha factor combining volatility-scaled momentum acceleration,
    volume-price divergence alignment, and intraday pattern confirmation.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Volatility-Scaled Momentum Acceleration
    # Multi-Timeframe Momentum Calculation
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['mom_5d'] = data['close'] / data['close'].shift(5) - 1
    data['mom_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum Acceleration
    data['accel_3_5'] = data['mom_5d'] - data['mom_3d']
    data['accel_5_10'] = data['mom_10d'] - data['mom_5d']
    data['composite_accel'] = (data['accel_3_5'] + data['accel_5_10']) / 2
    
    # Volatility Scaling
    data['ret_1d'] = data['close'] / data['close'].shift(1) - 1
    data['vol_3d'] = data['ret_1d'].rolling(window=3).std()
    data['vol_5d'] = data['ret_1d'].rolling(window=5).std()
    data['vol_10d'] = data['ret_1d'].rolling(window=10).std()
    
    # Combined Signal
    data['vol_geomean'] = np.sqrt(data['vol_3d'] * data['vol_5d'] * data['vol_10d'])
    data['vol_scaled_accel'] = data['composite_accel'] / data['vol_geomean']
    data['vol_scaled_accel'] = data['vol_scaled_accel'] * np.sign(data['composite_accel'])
    
    # Volume-Price Divergence Alignment
    # Price Strength Components
    data['intraday_eff'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['momentum_persist'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Divergence Signals
    data['vol_short_ratio'] = data['volume'] / (data['volume'].rolling(window=5).apply(lambda x: np.exp(np.mean(np.log(x))), raw=True))
    data['vol_medium_trend'] = data['volume'] / (data['volume'].rolling(window=10).apply(lambda x: np.exp(np.mean(np.log(x))), raw=True))
    data['vol_accel'] = data['volume'] / data['volume'].shift(2)
    
    # Volume-Price Alignment
    data['core_price_strength'] = np.sqrt(data['intraday_eff'] * data['price_position'] * data['momentum_persist'])
    data['volume_confirmation'] = np.sqrt(data['vol_short_ratio'] * data['vol_medium_trend'])
    data['divergence_signal'] = data['core_price_strength'] * data['volume_confirmation']
    
    # Multi-Timeframe Integration
    data['divergence_3d'] = data['divergence_signal'].rolling(window=3).apply(lambda x: np.exp(np.mean(np.log(x))), raw=True)
    data['divergence_5d'] = data['divergence_signal'].rolling(window=5).apply(lambda x: np.exp(np.mean(np.log(x))), raw=True)
    data['combined_divergence'] = np.cbrt(data['divergence_signal'] * data['divergence_3d'] * data['divergence_5d'])
    
    # Intraday Pattern Confirmation
    # Morning Session Analysis
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['morning_buying'] = (data['high'] - data['open']) / data['open']
    data['morning_support'] = (data['open'] - data['low']) / data['open']
    
    # Afternoon Session Analysis
    data['afternoon_momentum'] = (data['close'] - data['high']) / data['high']
    data['afternoon_support'] = (data['close'] - data['low']) / data['low']
    data['closing_eff'] = (data['close'] - data['open']) / data['open']
    
    # Intraday Pattern Signals
    data['morning_dominance'] = data['morning_buying'] - data['morning_support']
    data['afternoon_consistency'] = data['afternoon_momentum'] - data['afternoon_support']
    data['day_structure'] = data['closing_eff'] * np.sign(data['morning_dominance'])
    
    # Volume-Intraday Alignment (simplified using daily data)
    data['morning_vol_intensity'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['afternoon_vol_persist'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['intraday_vol_pattern'] = np.sqrt(data['morning_vol_intensity'] * data['afternoon_vol_persist'])
    
    # Regime-Adaptive Signal Blending
    # Volatility Regime Detection
    data['short_term_vol'] = (data['high'] - data['low']) / data['close']
    data['medium_term_vol'] = data['short_term_vol'].rolling(window=5).apply(lambda x: np.exp(np.mean(np.log(x))), raw=True)
    data['vol_regime'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Momentum Regime Classification
    data['price_trend_strength'] = data['close'] / data['close'].shift(7) - 1
    data['volume_trend_strength'] = data['volume'] / (data['volume'].rolling(window=7).apply(lambda x: np.exp(np.mean(np.log(x))), raw=True))
    data['momentum_regime'] = np.sign(data['price_trend_strength']) * np.abs(data['volume_trend_strength'])
    
    # Signal Weighting
    data['high_vol_weight'] = 1 / (1 + data['vol_regime'])
    data['strong_momentum_weight'] = np.abs(data['momentum_regime'])
    data['adaptive_blend'] = data['high_vol_weight'] * data['strong_momentum_weight']
    
    # Final Alpha Factor
    # Component signals
    component1 = data['vol_scaled_accel']
    component2 = data['combined_divergence']
    component3 = data['day_structure'] * data['intraday_vol_pattern']
    
    # Apply regime-adaptive weights
    weighted_component1 = component1 * data['adaptive_blend']
    weighted_component2 = component2 * data['adaptive_blend']
    weighted_component3 = component3 * data['adaptive_blend']
    
    # Final factor using geometric mean
    data['alpha_factor'] = np.cbrt(weighted_component1 * weighted_component2 * weighted_component3)
    
    return data['alpha_factor']
