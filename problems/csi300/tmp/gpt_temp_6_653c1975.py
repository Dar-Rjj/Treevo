import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Efficiency-Momentum System
    # Efficiency Framework
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_efficiency'] = data['intraday_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Rolling Range Efficiency
    data['rolling_high_5'] = data['high'].rolling(window=5, min_periods=5).max()
    data['rolling_low_5'] = data['low'].rolling(window=5, min_periods=5).min()
    data['rolling_range_efficiency'] = abs(data['close'] - data['close'].shift(5)) / (data['rolling_high_5'] - data['rolling_low_5'])
    data['rolling_range_efficiency'] = data['rolling_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # True Range Efficiency
    data['vwap'] = data['amount'] / data['volume']
    data['true_range_efficiency'] = (data['high'] - data['low']) / data['vwap']
    data['true_range_efficiency'] = data['true_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Efficiency Momentum
    data['rolling_efficiency'] = data['intraday_efficiency'].rolling(window=5, min_periods=5).mean()
    data['efficiency_momentum'] = data['rolling_efficiency'] - data['rolling_efficiency'].shift(3)
    
    # Volatility-Weighted Momentum
    data['volatility'] = (data['high'] - data['low']) / data['close']
    data['short_momentum'] = (data['close'] / data['close'].shift(3) - 1) * (1 / data['volatility'])
    data['medium_momentum'] = (data['close'] / data['close'].shift(10) - 1) * (1 / data['volatility'])
    data['momentum_acceleration'] = (data['close'] - data['close'].shift(3)) - (data['close'].shift(1) - data['close'].shift(4))
    
    # Multi-Scale Convergence Analysis
    data['efficiency_momentum_alignment'] = np.sign(data['intraday_efficiency']) * np.sign(data['short_momentum'])
    data['timeframe_convergence'] = data['intraday_efficiency'] * data['rolling_efficiency']
    
    # Volume-Price-Value Divergence Detection
    # Volume Dynamics Analysis
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_intensity'] = data['volume'] / data['volume'].rolling(window=10, min_periods=10).mean()
    data['volume_efficiency'] = data['volume'] / abs(data['close'] - data['close'].shift(1))
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    data['volume_efficiency_trend'] = data['volume_efficiency'].rolling(window=3, min_periods=3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 2 if len(x) == 3 else np.nan)
    
    # Price-Value Divergence Patterns
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['value_divergence'] = abs(data['vwap'] - data['typical_price']) / data['typical_price']
    data['value_divergence'] = data['value_divergence'].replace([np.inf, -np.inf], np.nan)
    
    data['value_price_momentum'] = (data['vwap'] / data['vwap'].shift(1) - 1) / (data['close'] / data['close'].shift(1) - 1)
    data['value_price_momentum'] = data['value_price_momentum'].replace([np.inf, -np.inf], np.nan)
    
    # Amount-Volume Dynamics
    data['amount_momentum'] = data['amount'] / data['amount'].shift(1) - 1
    data['amount_volume_divergence'] = abs(data['volume_momentum'] - data['amount_momentum'])
    data['amount_efficiency'] = data['amount'] / abs(data['close'] - data['close'].shift(1))
    data['amount_efficiency'] = data['amount_efficiency'].replace([np.inf, -np.inf], np.nan)
    data['amount_value_alignment'] = np.sign(data['amount_momentum']) * np.sign(data['value_divergence'])
    
    # Multi-Scale Pressure & Structure Analysis
    # Intraday Pressure Components
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['closing_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['breakout_strength'] = (data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['shadow_analysis'] = data['upper_shadow'] / data['lower_shadow']
    data['shadow_analysis'] = data['shadow_analysis'].replace([np.inf, -np.inf], np.nan)
    
    # Multi-Timeframe Pressure
    data['intraday_pressure'] = data['upper_shadow'] / (data['high'] - data['low'])
    
    # Structure Quality Assessment
    data['pressure_quality'] = data['closing_position'] * data['breakout_strength']
    data['shadow_efficiency'] = abs(data['close'] - data['open']) / (data['upper_shadow'] + data['lower_shadow'])
    data['shadow_efficiency'] = data['shadow_efficiency'].replace([np.inf, -np.inf], np.nan)
    data['microstructure_stress'] = abs(data['vwap'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    data['microstructure_stress'] = data['microstructure_stress'].replace([np.inf, -np.inf], np.nan)
    
    # Adaptive Trend & Reversal Detection
    # Trend Strength Analysis
    def calc_trend_strength(series):
        if len(series) < 21:
            return np.nan
        signs = []
        for i in [5, 10, 20]:
            if i < len(series):
                current_sign = np.sign(series.iloc[-1] / series.iloc[-i-1] - 1)
                prev_sign = np.sign(series.iloc[-1] / series.iloc[-2] - 1)
                signs.append(1 if current_sign == prev_sign else 0)
        return sum(signs) / len(signs) if signs else 0
    
    data['trend_strength'] = data['close'].rolling(window=21, min_periods=21).apply(calc_trend_strength, raw=False)
    data['trend_consistency'] = np.sign(data['short_momentum']) * np.sign(data['medium_momentum'])
    data['trend_acceleration'] = data['momentum_acceleration'] * data['trend_strength']
    
    # Reversal Signal Generation
    data['intraday_reversal'] = data['intraday_efficiency'] * (data['open'] / data['close'].shift(1) - 1)
    data['value_based_reversal'] = (data['vwap'] / data['close'] - 1) * data['intraday_efficiency']
    data['volume_price_reversal'] = data['intraday_reversal'] * (1 + data['trend_strength'] / 3)
    
    # Adaptive Signal Weighting
    data['trend_weighted_reversal'] = data['volume_price_reversal'] * (1 - data['trend_strength'] / 3)
    data['value_enhanced_reversal'] = data['value_based_reversal'] * data['amount_volume_divergence']
    data['efficiency_weighted_signal'] = data['intraday_reversal'] * data['intraday_efficiency']
    
    # Multi-Dimensional Signal Integration
    # Core Efficiency-Momentum Convergence
    momentum_consistency = (np.sign(data['short_momentum']) == np.sign(data['medium_momentum'])).astype(float)
    data['base_convergence'] = data['timeframe_convergence'] * momentum_consistency
    
    volume_confirmation = (np.sign(data['volume_momentum']) == np.sign(data['short_momentum'])).astype(float)
    data['volume_enhanced'] = data['base_convergence'] * volume_confirmation
    
    data['value_weighted'] = data['volume_enhanced'] * (1 - data['value_divergence'])
    
    # Divergence Anomaly Detection
    divergence_patterns = [
        data['value_divergence'],
        data['amount_volume_divergence'],
        data['microstructure_stress']
    ]
    data['divergence_score'] = sum(divergence_patterns) / len(divergence_patterns)
    data['value_price_anomaly'] = data['microstructure_stress'] * data['value_divergence']
    data['amount_volume_anomaly'] = data['amount_volume_divergence'] * data['volume_efficiency_trend']
    
    # Multi-Factor Validation
    data['volume_value_support'] = data['breakout_strength'] * (1 - data['value_divergence'])
    data['efficiency_trend_alignment'] = data['timeframe_convergence'] * data['trend_acceleration']
    data['pressure_value_confirmation'] = data['pressure_quality'] * (1 - abs(data['vwap'] - data['typical_price']) / data['typical_price'])
    
    # Final Alpha Signal
    # Combine key components with appropriate weights
    alpha_signal = (
        0.25 * data['value_weighted'] +
        0.20 * data['efficiency_trend_alignment'] +
        0.15 * data['volume_value_support'] +
        0.15 * data['pressure_value_confirmation'] -
        0.10 * data['divergence_score'] -
        0.10 * data['value_price_anomaly'] -
        0.05 * data['amount_volume_anomaly']
    )
    
    return alpha_signal
