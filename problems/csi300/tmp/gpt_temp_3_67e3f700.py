import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Efficiency Components
    data['daily_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_momentum'] = data['daily_efficiency'] - data['daily_efficiency'].shift(3)
    
    # Efficiency Regime Classification
    conditions = [
        data['daily_efficiency'] > 0.7,
        (data['daily_efficiency'] >= 0.3) & (data['daily_efficiency'] <= 0.7),
        data['daily_efficiency'] < 0.3
    ]
    choices = [2, 1, 0]  # High, Normal, Low
    data['efficiency_regime'] = np.select(conditions, choices, default=1)
    
    # Volume Microstructure Dynamics
    data['volume_efficiency'] = data['volume'] / (np.abs(data['close'] - data['close'].shift(1)) + 1e-8)
    data['volume_regime_change'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).apply(lambda x: np.median(x[:-1]) if len(x) > 1 else 1.0)
    
    # Volume Concentration
    data['volume_concentration'] = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    data['volume_efficiency_trend'] = data['volume_efficiency'].rolling(window=3, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0.0)
    
    # Divergence Detection System
    data['efficiency_volume_divergence'] = (np.sign(data['efficiency_momentum']) != np.sign(data['volume_efficiency_trend'])).astype(int)
    data['divergence_strength'] = np.abs(data['efficiency_momentum']) - np.abs(data['volume_efficiency_trend'])
    
    # Price Momentum
    data['price_momentum'] = data['close'] - data['close'].shift(3)
    data['price_volume_consistency'] = (np.sign(data['price_momentum']) == np.sign(data['volume_regime_change'])).astype(int)
    data['microstructure_alignment'] = (np.sign(data['efficiency_momentum']) == np.sign(data['volume_regime_change'])).astype(int)
    
    # Intraday Pressure and Absorption Patterns
    data['opening_gap_pressure'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['closing_relative_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['pressure_asymmetry'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    # Range Rejection Analysis
    upper_rejection = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    lower_rejection = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['upper_rejection'] = np.where(data['close'] < data['open'], upper_rejection, 0)
    data['lower_rejection'] = np.where(data['close'] > data['open'], lower_rejection, 0)
    data['current_rejection'] = np.where(data['close'] < data['open'], data['upper_rejection'], data['lower_rejection'])
    data['rejection_ratio'] = data['current_rejection'] / (data['current_rejection'].rolling(window=3, min_periods=1).max() + 1e-8)
    
    # Absorption Pattern Recognition
    data['pressure_accumulation'] = (data['close'] - data['open']) * data['volume']
    data['cumulative_pressure_5d'] = data['pressure_accumulation'].rolling(window=5, min_periods=1).sum()
    data['absorption_confirmation'] = data['volume_concentration'] * data['pressure_accumulation']
    
    # Structural Break and Regime Detection
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['volatility_ratio'] = data['true_range'].rolling(window=5, min_periods=1).std() / (data['true_range'].rolling(window=20, min_periods=1).std() + 1e-8)
    data['volatility_break'] = (data['volatility_ratio'] > 2.0).astype(int)
    
    # Volume Structure Shifts
    data['volume_break_ratio'] = data['volume'] / data['volume'].rolling(window=20, min_periods=1).apply(lambda x: np.median(x[:-1]) if len(x) > 1 else 1.0)
    data['abnormal_volume'] = (data['volume_break_ratio'] > 2.5).astype(int)
    
    # Efficiency Regime Changes
    data['efficiency_volatility'] = data['daily_efficiency'].rolling(window=5, min_periods=1).std() / (data['daily_efficiency'].rolling(window=20, min_periods=1).std() + 1e-8)
    data['efficiency_break'] = (data['efficiency_volatility'] > 1.8).astype(int)
    data['regime_transition'] = data['volatility_break'].rolling(window=3, min_periods=1).max()
    
    # Amount-Based Price Discovery
    data['implied_price'] = data['amount'] / (data['volume'] + 1e-8)
    data['trading_intensity'] = np.abs(data['implied_price'] - data['close']) / (data['close'] + 1e-8)
    data['amount_price_divergence'] = np.sign(data['implied_price'] - data['close']) * data['trading_intensity']
    
    # Amount-Volume Alignment
    data['amount_efficiency'] = data['amount'] / (np.abs(data['close'] - data['close'].shift(1)) + 1e-8)
    data['amount_volume_ratio'] = data['amount'] / (data['volume'] + 1e-8)
    data['amount_price_consistency'] = (np.sign(data['amount_price_divergence']) == np.sign(data['price_momentum'])).astype(int)
    
    # Microstructure Coherence
    data['volume_amount_alignment'] = (np.sign(data['volume_regime_change']) == np.sign(data['amount_price_divergence'])).astype(int)
    data['confirmation_strength'] = data['volume_regime_change'] * data['amount_price_divergence']
    data['divergence_patterns'] = (np.sign(data['volume_regime_change']) != np.sign(data['amount_price_divergence'])).astype(int)
    
    # Primary Factor Components
    component1 = data['efficiency_volume_divergence'] * data['pressure_accumulation'] * data['volatility_ratio']
    component2 = data['amount_price_divergence'] * data['rejection_ratio'] * data['volume_concentration']
    component3 = data['microstructure_alignment'] * data['efficiency_momentum'] * data['trading_intensity']
    
    # Base Composite
    base_composite = component1 * component2 * component3
    
    # Regime-Adaptive Weighting
    volatility_conditions = [
        data['volatility_ratio'] > 1.5,
        (data['volatility_ratio'] >= 0.7) & (data['volatility_ratio'] <= 1.5),
        data['volatility_ratio'] < 0.7
    ]
    volatility_multipliers = [1.5, 1.0, 0.7]
    volatility_weight = np.select(volatility_conditions, volatility_multipliers, default=1.0)
    
    # Structural Break Enhancement
    break_conditions = [
        data['regime_transition'] == 1,
        data['volatility_break'] == 0,
        (data['volatility_break'] == 1) & (data['regime_transition'] == 0)
    ]
    break_multipliers = [1.3, 1.0, 0.8]
    break_weight = np.select(break_conditions, break_multipliers, default=1.0)
    
    # Final Alpha Calculation
    final_alpha = base_composite * volatility_weight * break_weight * data['microstructure_alignment']
    
    return final_alpha
