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
    df['prev_close'] = df['close'].shift(1)
    df['prev_open'] = df['open'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_volume'] = df['volume'].shift(1)
    df['prev_amount'] = df['amount'].shift(1)
    df['prev2_high'] = df['high'].shift(2)
    df['prev2_low'] = df['low'].shift(2)
    df['prev2_close'] = df['close'].shift(2)
    df['prev5_close'] = df['close'].shift(5)
    
    # Rolling calculations
    df['rolling_max_high_2'] = df['high'].rolling(window=2, min_periods=1).max()
    df['rolling_min_low_2'] = df['low'].rolling(window=2, min_periods=1).min()
    df['rolling_sum_volume_2'] = df['volume'].rolling(window=2, min_periods=1).sum()
    
    df['rolling_max_high_5'] = df['high'].rolling(window=5, min_periods=1).max()
    df['rolling_min_low_5'] = df['low'].rolling(window=5, min_periods=1).min()
    df['rolling_sum_volume_5'] = df['volume'].rolling(window=5, min_periods=1).sum()
    
    # Directional Energy-Volatility Asymmetry
    df['opening_energy_asymmetry'] = ((df['open'] - df['prev_close']) * 
                                    (df['high'] - df['low']) * 
                                    df['volume'] / 
                                    (df['prev_high'] - df['prev_low'] + 1e-8) * 
                                    np.sign(df['close'] - df['open']))
    
    df['intraday_energy_asymmetry'] = ((df['close'] - df['open']) * 
                                     (df['high'] - df['low']) * 
                                     df['amount'] / 
                                     (df['prev_high'] - df['prev_low'] + 1e-8) * 
                                     np.sign(df['close'] - df['prev_close']))
    
    df['closing_energy_asymmetry'] = ((df['close'] - (df['high'] + df['low'])/2) * 
                                    (df['high'] - df['low']) * 
                                    df['volume'] / 
                                    (df['prev_high'] - df['prev_low'] + 1e-8) * 
                                    np.sign(df['close'] - df['open']))
    
    # Volume-Volatility Asymmetry Dynamics
    df['volume_asymmetry_energy'] = (((df['volume']/df['prev_volume'] - 1) * 
                                    (df['close'] - df['open']) * 
                                    (df['high'] - df['low']) * 
                                    np.sign(df['amount'] - df['prev_amount'])) / 
                                    (df['prev_high'] - df['prev_low'] + 1e-8))
    
    df['price_asymmetry_energy'] = (((df['close']/df['prev_close'] - 1) - 
                                   (df['open']/df['prev_open'] - 1)) * 
                                   (df['high'] - df['low']) * 
                                   df['volume'] / 
                                   (df['prev2_high'] - df['prev2_low'] + 1e-8))
    
    df['asymmetry_energy_convergence'] = (df['volume_asymmetry_energy'] * 
                                        df['price_asymmetry_energy'] * 
                                        df['amount'])
    
    # Multi-Scale Asymmetry Energy
    df['short_term_asymmetry_energy'] = (((df['close'] - df['prev2_close']) * 
                                        (df['rolling_max_high_2'] - df['rolling_min_low_2']) * 
                                        df['rolling_sum_volume_2'] / 
                                        (df['rolling_max_high_2'] - df['rolling_min_low_2'] + 1e-8)) * 
                                        np.sign(df['close'] - df['open']))
    
    df['medium_term_asymmetry_energy'] = (((df['close'] - df['prev5_close']) * 
                                         (df['rolling_max_high_5'] - df['rolling_min_low_5']) * 
                                         df['rolling_sum_volume_5'] / 
                                         (df['rolling_max_high_5'] - df['rolling_min_low_5'] + 1e-8)) * 
                                         np.sign(df['close'] - df['prev_close']))
    
    df['asymmetry_energy_divergence'] = (df['short_term_asymmetry_energy'] - 
                                       df['medium_term_asymmetry_energy'])
    
    # Fractal Range-Asymmetry Integration
    df['fractal_range_asymmetry_efficiency'] = ((np.abs(df['close'] - df['open']) * 
                                               (df['high'] - df['low']) * 
                                               df['volume'] / 
                                               (df['prev_high'] - df['prev_low'] + 1e-8)) * 
                                               np.sign(df['close'] - df['open']))
    
    df['fractal_range_expansion_asymmetry'] = (((df['high'] - df['low']) / 
                                              (df['prev_high'] - df['prev_low'] + 1e-8) * 
                                              df['volume'] / df['prev_volume']) * 
                                              np.sign(df['close'] - df['prev_close']))
    
    df['fractal_range_compression_asymmetry'] = ((1 / (df['fractal_range_expansion_asymmetry'] + 1e-8) * 
                                                df['volume'] / df['prev_volume']) * 
                                                np.sign(df['amount'] - df['prev_amount']))
    
    # Fractal Range-Asymmetry Momentum
    df['range_asymmetry_momentum_efficiency'] = (df['fractal_range_asymmetry_efficiency'] * 
                                               df['intraday_energy_asymmetry'])
    
    df['range_expansion_asymmetry_momentum'] = (df['fractal_range_expansion_asymmetry'] * 
                                              df['volume_asymmetry_energy'])
    
    df['range_compression_asymmetry_momentum'] = (df['fractal_range_compression_asymmetry'] * 
                                                df['price_asymmetry_energy'])
    
    # Fractal Range-Asymmetry Dynamics
    df['fractal_range_asymmetry_breakout'] = (df['fractal_range_expansion_asymmetry'] * 
                                            df['volume_asymmetry_energy'] * 
                                            np.sign(df['close'] - df['open']))
    
    df['fractal_range_asymmetry_consolidation'] = (df['fractal_range_compression_asymmetry'] * 
                                                 df['price_asymmetry_energy'] * 
                                                 np.sign(df['close'] - df['prev_close']))
    
    df['fractal_range_asymmetry_quality'] = (df['range_asymmetry_momentum_efficiency'] * 
                                           df['intraday_energy_asymmetry'].rolling(window=3, min_periods=1).apply(lambda x: np.sum(np.sign(x))))
    
    # Multi-Timeframe Asymmetry Energy Convergence
    df['two_day_asymmetry_energy'] = df['short_term_asymmetry_energy']
    
    df['two_day_volume_asymmetry'] = ((df['volume'] / df['volume'].shift(2)) * 
                                    df['two_day_asymmetry_energy'] * 
                                    np.sign(df['amount'] - df['amount'].shift(2)))
    
    df['short_term_asymmetry_alignment'] = (np.sign(df['intraday_energy_asymmetry']) * 
                                          np.sign(df['two_day_asymmetry_energy']) * 
                                          np.sign(df['close'] - df['open']))
    
    df['five_day_asymmetry_energy'] = df['medium_term_asymmetry_energy']
    
    df['five_day_volume_asymmetry'] = ((df['volume'] * df['intraday_energy_asymmetry']).rolling(window=5, min_periods=1).sum() / 5 * 
                                     np.sign(df['close'] - df['open']))
    
    df['medium_term_asymmetry_divergence'] = (df['two_day_asymmetry_energy'] - 
                                            df['five_day_asymmetry_energy'])
    
    df['asymmetry_time_convergence'] = (df['short_term_asymmetry_alignment'] * 
                                      np.sign(df['two_day_asymmetry_energy']) * 
                                      np.sign(df['five_day_asymmetry_energy']))
    
    df['volume_asymmetry_time_alignment'] = (df['two_day_volume_asymmetry'] * 
                                           df['five_day_volume_asymmetry'] * 
                                           np.sign(df['close'] - df['open']))
    
    df['multi_timeframe_asymmetry_momentum'] = (df['asymmetry_time_convergence'] * 
                                              df['volume_asymmetry_time_alignment'] * 
                                              (df['high'] - df['low']))
    
    # Asymmetry Breakout Detection
    df['momentum_asymmetry_breakout'] = ((df['intraday_energy_asymmetry'] > 
                                        df['intraday_energy_asymmetry'].rolling(window=20, min_periods=1).quantile(0.8)) * 
                                        np.sign(df['close'] - df['open']))
    
    df['range_asymmetry_breakout'] = ((df['fractal_range_expansion_asymmetry'] > 
                                     df['fractal_range_expansion_asymmetry'].rolling(window=20, min_periods=1).quantile(0.8)) * 
                                     np.sign(df['close'] - df['prev_close']))
    
    df['volume_asymmetry_breakout'] = ((df['volume_asymmetry_energy'] > 
                                      df['volume_asymmetry_energy'].rolling(window=20, min_periods=1).quantile(0.8)) * 
                                      np.sign(df['amount'] - df['prev_amount']))
    
    df['multi_asymmetry_breakout'] = (df['momentum_asymmetry_breakout'] * 
                                    df['range_asymmetry_breakout'] * 
                                    df['volume_asymmetry_breakout'])
    
    # Asymmetry Reversal Patterns
    df['momentum_asymmetry_reversal'] = ((df['intraday_energy_asymmetry'] < 
                                        df['intraday_energy_asymmetry'].rolling(window=20, min_periods=1).quantile(0.2)) * 
                                        np.sign(df['close'] - df['open']))
    
    df['range_asymmetry_reversal'] = ((df['fractal_range_compression_asymmetry'] > 
                                     df['fractal_range_compression_asymmetry'].rolling(window=20, min_periods=1).quantile(0.8)) * 
                                     np.sign(df['close'] - df['prev_close']))
    
    df['volume_asymmetry_reversal'] = ((df['price_asymmetry_energy'] < 
                                      df['price_asymmetry_energy'].rolling(window=20, min_periods=1).quantile(0.2)) * 
                                      np.sign(df['amount'] - df['prev_amount']))
    
    # Core Fractal Asymmetry Factors
    df['multi_scale_asymmetry_energy'] = ((df['intraday_energy_asymmetry'] + 
                                         df['short_term_asymmetry_energy'] + 
                                         df['medium_term_asymmetry_energy']) * 
                                         df['intraday_energy_asymmetry'].rolling(window=5, min_periods=1).apply(lambda x: np.sum(np.sign(x))))
    
    df['volume_asymmetry_momentum'] = (df['volume_asymmetry_energy'] * 
                                     df['price_asymmetry_energy'] * 
                                     np.sign(df['close'] - df['open']))
    
    df['range_asymmetry_momentum'] = (df['range_asymmetry_momentum_efficiency'] * 
                                    df['fractal_range_asymmetry_quality'] * 
                                    np.sign(df['close'] - df['prev_close']))
    
    # Multi-Dimensional Asymmetry Integration
    df['price_asymmetry_dimension'] = (df['multi_scale_asymmetry_energy'] * 
                                     df['intraday_energy_asymmetry'].rolling(window=5, min_periods=1).apply(lambda x: np.sum(np.sign(x))) * 
                                     df['volume'])
    
    df['volume_asymmetry_dimension'] = (df['volume_asymmetry_momentum'] * 
                                      df['volume_asymmetry_time_alignment'] * 
                                      df['amount'])
    
    df['range_asymmetry_dimension'] = (df['range_asymmetry_momentum'] * 
                                     df['fractal_range_asymmetry_breakout'] * 
                                     (df['high'] - df['low']))
    
    # Asymmetry Pattern Confirmation
    df['momentum_asymmetry_confirmation'] = (df['asymmetry_time_convergence'] * 
                                           df['multi_timeframe_asymmetry_momentum'] * 
                                           np.sign(df['close'] - df['open']))
    
    df['volume_asymmetry_confirmation'] = (np.sign(df['close'] - df['open']) * 
                                         np.sign(df['volume'] - df['prev_volume']) * 
                                         df['volume_asymmetry_breakout'] * 
                                         np.sign(df['amount'] - df['prev_amount']))
    
    df['range_asymmetry_confirmation'] = (df['range_asymmetry_momentum_efficiency'] * 
                                        df['fractal_range_asymmetry_consolidation'] * 
                                        np.sign(df['close'] - df['prev_close']))
    
    # Final Fractal Microstructure Alpha
    df['core_asymmetry_alpha'] = (df['price_asymmetry_dimension'] * 
                                df['volume_asymmetry_dimension'] * 
                                df['range_asymmetry_dimension'])
    
    df['enhanced_asymmetry_alpha'] = (df['core_asymmetry_alpha'] * 
                                    df['momentum_asymmetry_confirmation'] * 
                                    df['volume_asymmetry_confirmation'] * 
                                    df['range_asymmetry_confirmation'])
    
    result = df['enhanced_asymmetry_alpha']
    
    return result
