import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['price_range'] = df['high'] - df['low']
    df['close_position'] = (df['close'] - df['low']) - (df['high'] - df['close'])
    df['normalized_close_position'] = df['close_position'] / df['price_range']
    df['intraday_move'] = df['close'] - df['open']
    df['normalized_intraday_move'] = df['intraday_move'] / df['price_range']
    df['vwap'] = df['amount'] / df['volume']
    
    # Multi-Scale Order Flow Asymmetry
    for window in [3, 10, 20]:
        df[f'volume_change_{window}'] = df['volume'] - df['volume'].shift(window)
        df[f'volume_direction_{window}'] = np.sign(df[f'volume_change_{window}'])
        df[f'order_flow_asymmetry_{window}'] = df['normalized_close_position'] * df[f'volume_direction_{window}']
    
    # Fractal Volume Absorption
    df['gap_absorption'] = abs(df['open'] - df['close'].shift(1)) * df['volume'] / df['price_range']
    df['range_absorption'] = (df['volume'] / df['price_range']) * (df['price_range'] / df['price_range'].shift(5))
    df['flow_pressure'] = (df['normalized_intraday_move'] * 
                          ((df['vwap'] - df['open']) / df['price_range']) * 
                          (df['volume'] / df['volume'].shift(10)))
    
    # Fractal Efficiency Dynamics
    df['multi_scale_efficiency'] = (df['normalized_intraday_move'] * 
                                   ((df['close'] - df['close'].shift(5)) / df['price_range'].shift(5)))
    df['efficiency_memory'] = np.sign(df['normalized_intraday_move'] - 
                                     (df['close'].shift(1) - df['open'].shift(1)) / 
                                     (df['high'].shift(1) - df['low'].shift(1)))
    df['volume_efficiency'] = (df['volume'] / df['volume'].shift(5)) * df['normalized_intraday_move']
    
    # Asymmetric Fractal Rejection Framework
    df['max_oc'] = df[['open', 'close']].max(axis=1)
    df['min_oc'] = df[['open', 'close']].min(axis=1)
    df['net_asymmetric_rejection'] = (df['high'] - df['max_oc']) - (df['min_oc'] - df['low'])
    df['fractal_rejection_bias'] = ((df['high'] - df['max_oc']) / 
                                   (df['min_oc'] - df['low'] + 1e-8) * 
                                   (df['volume'] / df['volume'].shift(5)))
    df['gap_asymmetry'] = (df['open'] - df['close'].shift(1)) - (df['high'] - df['open'])
    
    # Fractal Extreme Behavior
    df['high_5d_max'] = df['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    df['low_5d_min'] = df['low'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan)
    df['price_level_breakout'] = (df['high'] / df['high_5d_max']) - (df['low'] / df['low_5d_min'])
    df['volume_spike'] = (df['volume'] / df['volume'].shift(1)) * (df['volume'].shift(1) / df['volume'].shift(2))
    df['rejection_efficiency'] = (abs(df['open'] - df['close'].shift(1)) / df['price_range'] * 
                                df['volume'] / df['volume'].shift(5) - 
                                abs(df['intraday_move']) / df['price_range'])
    
    # Fractal Transition Detection
    df['volatility_transition'] = ((df['price_range'] / df['price_range'].shift(3) > 1.5) & 
                                  (df['volume'] / df['volume'].shift(3) > 1.2)).astype(float)
    df['memory_fracture'] = (df['volume'] / df['volume'].shift(1)) * (df['volume'].shift(1) / df['volume'].shift(2))
    df['fractal_alignment'] = (np.sign(df['price_range'] - df['price_range'].shift(1)) * 
                              np.sign(df['volume'] / df['volume'].shift(1) - 1))
    
    # Volume-Velocity Fractal Integration
    df['trade_size_fractal'] = ((df['vwap'] / (df['amount'].shift(5) / df['volume'].shift(5))) * 
                               (df['close'] - df['close'].shift(1)) / df['price_range'])
    df['volume_acceleration'] = ((df['close'] - df['close'].shift(1)) / 
                                (df['close'].shift(1) - df['close'].shift(2)) * 
                                (df['volume'] / df['volume'].shift(1)))
    df['fractal_velocity'] = (df['volume'] / df['volume'].shift(5)) * df['intraday_move']
    
    # Liquidity Efficiency Fractal
    df['range_efficiency'] = df['price_range'] / (df['volume'] / df['volume'].shift(5))
    df['price_vwap_gap'] = (df['close'] - df['vwap']) / df['price_range']
    df['bid_ask_pressure'] = df['close_position'] * df['volume']
    
    # Volume-Range Coherence
    df['coherence_momentum'] = (df['volume'] * df['price_range'] / 
                               (df['volume'].shift(1) * df['price_range'].shift(1)))
    df['range_expansion'] = (df['price_range'] / df['price_range'].shift(1)) * (df['volume'] / df['volume'].shift(1))
    df['volume_distribution'] = df['volume'] / ((df['volume'].shift(3) + df['volume'].shift(2) + df['volume'].shift(1)) / 3)
    
    # Fractal Momentum Construction
    df['clean_momentum'] = df['close'] / df['close'].shift(1) - 1
    df['fractal_acceleration'] = ((df['close'] - df['close'].shift(1)) / 
                                 (df['close'].shift(1) - df['close'].shift(2)) * 
                                 (df['volume'] / df['volume'].shift(1)))
    df['breakout_asymmetry'] = (df['high'] / df['high'].shift(1) - 1) - (df['low'] / df['low'].shift(1) - 1)
    
    # Asymmetric Fractal Response
    df['upside_response'] = ((df['high'] - df['close']) / (df['high'] - df['open'] + 1e-8) * 
                            (df['volume'] / df['volume'].shift(10)))
    df['downside_response'] = ((df['close'] - df['low']) / (df['open'] - df['low'] + 1e-8) * 
                              (df['volume'] / df['volume'].shift(10)))
    df['net_asymmetry'] = (df['upside_response'] - df['downside_response']) * np.sign(df['intraday_move'])
    
    # Fractal Persistence Signals
    df['volatility_memory'] = ((df['price_range'] / df['price_range'].shift(5)) * 
                              (df['close'] - df['close'].shift(1)) / df['price_range'])
    
    # Calculate rolling efficiency persistence
    def efficiency_persistence(series):
        if len(series) < 3:
            return np.nan
        current = np.sign(series.iloc[-1] - series.iloc[-2])
        previous = np.sign(series.iloc[-2] - series.iloc[-3])
        return 1.0 if current == previous else 0.0
    
    efficiency_series = (df['close'] - df['open']) / df['price_range']
    df['efficiency_persistence'] = efficiency_series.rolling(window=3, min_periods=3).apply(
        efficiency_persistence, raw=False)
    
    # Calculate momentum consistency
    def momentum_consistency(series):
        if len(series) < 3:
            return np.nan
        current = np.sign(series.iloc[-1] - 1)
        previous = np.sign(series.iloc[-2] - 1)
        return 1.0 if current == previous else 0.0
    
    momentum_series = df['close'] / df['close'].shift(1)
    df['momentum_consistency'] = momentum_series.rolling(window=3, min_periods=3).apply(
        momentum_consistency, raw=False)
    
    # Divergence and Fractal Validation
    df['efficiency_volume_divergence'] = (np.sign(df['normalized_intraday_move'] - 
                                                 ((df['close'].shift(1) - df['open'].shift(1)) / 
                                                  (df['high'].shift(1) - df['low'].shift(1)))) * 
                                        np.sign((df['volume'] / df['volume'].shift(1)) - 
                                                (df['volume'].shift(1) / df['volume'].shift(2))))
    
    df['rejection_flow_alignment'] = (np.sign(df['net_asymmetric_rejection']) * 
                                     np.sign(df['normalized_close_position']))
    
    df['price_fracture'] = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) - df['close'].shift(2))
    
    # Range Dynamics Assessment
    df['range_expansion_signal'] = (df['price_range'] / df['price_range'].shift(1) > 1.2).astype(float)
    df['range_contraction_signal'] = (df['price_range'] / df['price_range'].shift(1) < 0.8).astype(float)
    df['mean_reversion_strength'] = 1 - abs(df['close'] - df['close'].shift(1)) / df['price_range']
    
    # Fractal Confirmation
    df['multi_scale_alignment'] = (np.sign(df['price_range'] / df['price_range'].shift(3) - 1) * 
                                  np.sign(df['price_range'] / df['price_range'].shift(10) - 1))
    df['volume_price_confirmation'] = (np.sign(df['volume'] / df['volume'].shift(3) - 1) * 
                                      np.sign(df['close'] / df['close'].shift(1) - 1))
    df['efficiency_confirmation'] = np.sign(df['normalized_intraday_move'] * 
                                          (df['volume'] / df['volume'].shift(5)) - 
                                          abs(df['intraday_move']) / df['price_range'])
    
    # Core Velocity Components
    df['order_flow_fractal_velocity'] = (df['order_flow_asymmetry_3'] * df['gap_absorption'] * df['volume_acceleration'])
    df['rejection_efficiency_velocity'] = (df['net_asymmetric_rejection'] * df['multi_scale_efficiency'] * df['fractal_rejection_bias'])
    df['volume_absorption_velocity'] = (df['clean_momentum'] * df['coherence_momentum'] * df['range_efficiency'])
    df['breakout_momentum_velocity'] = (df['breakout_asymmetry'] * df['price_level_breakout'] * df['fractal_acceleration'])
    
    # Enhanced Fractal Signals
    df['microstructure_confirmed'] = (df['order_flow_fractal_velocity'] * 
                                     df['efficiency_volume_divergence'] * 
                                     df['rejection_flow_alignment'])
    df['volume_efficiency_signal'] = (df['volume_absorption_velocity'] * 
                                     df['volume_efficiency'] * 
                                     df['efficiency_persistence'])
    df['breakout_rejection'] = (df['breakout_momentum_velocity'] * 
                               df['volume_spike'] * 
                               df['net_asymmetry'])
    df['range_enhanced'] = (df['net_asymmetry'] * 
                           (df['range_expansion_signal'] - df['range_contraction_signal']) * 
                           df['mean_reversion_strength'])
    
    # Adaptive Weighting Framework
    df['momentum_weight'] = np.where((np.sign(df['fractal_acceleration']) * np.sign(df['volume_acceleration'])) > 0, 1.5, 1.0)
    df['volatility_weight'] = np.where(df['volatility_transition'] > 0, 1.3, 1.0)
    df['efficiency_weight'] = np.where(df['efficiency_confirmation'] > 0, 1.2, 1.0)
    df['persistence_weight'] = df['momentum_consistency'] * df['efficiency_persistence']
    
    # Final Alpha Outputs
    df['alpha_1'] = (df['microstructure_confirmed'] * df['volatility_weight'] * df['volatility_memory'])
    df['alpha_2'] = (df['volume_efficiency_signal'] * df['momentum_weight'] * df['coherence_momentum'])
    df['alpha_3'] = (df['breakout_rejection'] * df['efficiency_weight'] * df['fractal_alignment'])
    df['alpha_4'] = (df['range_enhanced'] * df['persistence_weight'] * df['multi_scale_alignment'])
    
    # Combine alphas with equal weighting
    result = (df['alpha_1'] + df['alpha_2'] + df['alpha_3'] + df['alpha_4']) / 4
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
