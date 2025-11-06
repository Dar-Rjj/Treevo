import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate entropy measures
    def calculate_entropy(series, window=5):
        if len(series) < window:
            return pd.Series(np.nan, index=series.index)
        
        entropy_values = []
        for i in range(len(series)):
            if i < window - 1:
                entropy_values.append(np.nan)
                continue
            
            window_data = series.iloc[i-window+1:i+1]
            if window_data.std() == 0:
                entropy_values.append(0)
            else:
                # Normalize and calculate entropy-like measure
                normalized = (window_data - window_data.min()) / (window_data.max() - window_data.min() + 1e-8)
                normalized = normalized[normalized > 0]
                if len(normalized) > 0:
                    entropy = -np.sum(normalized * np.log(normalized + 1e-8))
                else:
                    entropy = 0
                entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=series.index)
    
    # Calculate price and volume entropy
    price_series = (df['high'] + df['low'] + df['close']) / 3
    df['price_entropy'] = calculate_entropy(price_series, 5)
    df['volume_entropy'] = calculate_entropy(df['volume'], 5)
    
    # Fill NaN values
    df['price_entropy'] = df['price_entropy'].fillna(0)
    df['volume_entropy'] = df['volume_entropy'].fillna(0)
    
    # Entropy Intraday Divergence Patterns
    df['prev_close'] = df['close'].shift(1)
    df['entropy_opening_divergence'] = ((df['open'] - df['prev_close']) * df['volume'] / 
                                       (df['high'] - df['low'] + 1e-8) * df['price_entropy'])
    
    df['entropy_midday_divergence'] = (((df['high'] + df['low'])/2 - (df['open'] + df['close'])/2) * 
                                      df['volume'] / (df['high'] - df['low'] + 1e-8) * df['price_entropy'])
    
    df['entropy_closing_divergence'] = ((df['close'] - (df['high'] + df['low'])/2) * df['volume'] / 
                                       (df['high'] - df['low'] + 1e-8) * df['price_entropy'])
    
    # Entropy Divergence Patterns (combined)
    df['entropy_divergence_patterns'] = (df['entropy_opening_divergence'] + 
                                        df['entropy_midday_divergence'] + 
                                        df['entropy_closing_divergence'])
    
    # Entropy Multi-Timeframe Divergence
    df['entropy_short_term_divergence'] = (df['entropy_divergence_patterns'] - 
                                          df['entropy_divergence_patterns'].shift(1))
    
    df['entropy_medium_term_divergence'] = (df['entropy_divergence_patterns'].rolling(3).mean() - 
                                           df['entropy_divergence_patterns'].rolling(5).mean())
    
    df['entropy_divergence_acceleration'] = (df['entropy_short_term_divergence'] - 
                                            df['entropy_medium_term_divergence'])
    
    # Entropy Divergence Quality Assessment
    df['entropy_divergence_consistency'] = (np.sign(df['entropy_opening_divergence']) * 
                                           np.sign(df['entropy_closing_divergence']) * 
                                           df['price_entropy'])
    
    volume_avg_4d = df['volume'].rolling(4).mean()
    df['entropy_volume_confirmation'] = (df['entropy_divergence_patterns'] * 
                                        df['volume'] / (volume_avg_4d + 1e-8) * 
                                        df['volume_entropy'])
    
    df['entropy_price_efficiency'] = (np.abs(df['close'] - df['open']) / 
                                     (df['high'] - df['low'] + 1e-8) * 
                                     df['entropy_divergence_patterns'] * 
                                     df['price_entropy'])
    
    # Entropy Volume Regime Shifts
    df['entropy_volume_breakout'] = ((df['volume'] / df['volume'].shift(1) - 
                                     df['volume'].shift(1) / df['volume'].shift(2)) * 
                                    df['volume_entropy'])
    
    volume_avg_5d = df['volume'].rolling(5).mean()
    df['entropy_volume_regime_change'] = (np.sign(df['volume'] - volume_avg_4d) * 
                                         np.sign(df['volume'].shift(1) - volume_avg_5d.shift(1)) * 
                                         df['volume_entropy'])
    
    df['entropy_volume_momentum_shift'] = (((df['volume'] - df['volume'].shift(1)) / (df['volume'].shift(1) + 1e-8) - 
                                           (df['volume'].shift(1) - df['volume'].shift(2)) / (df['volume'].shift(2) + 1e-8)) * 
                                          df['volume_entropy'])
    
    # Entropy Price Regime Transitions
    df['entropy_volatility_regime'] = (((df['high'] - df['low']) / (df['close'].shift(1) + 1e-8) - 
                                       (df['high'].shift(1) - df['low'].shift(1)) / (df['close'].shift(2) + 1e-8)) * 
                                      df['price_entropy'])
    
    df['entropy_trend_regime_shift'] = (np.sign(df['close'] - df['close'].shift(1)) * 
                                       np.sign(df['close'].shift(1) - df['close'].shift(2)) * 
                                       df['price_entropy'])
    
    range_avg_5d = (df['high'] - df['low']).rolling(5).mean()
    df['entropy_range_breakout'] = ((df['high'] - df['low']) / (range_avg_5d + 1e-8) * 
                                   df['price_entropy'])
    
    # Entropy Combined Regime Detection
    df['entropy_regime_alignment'] = (df['entropy_volume_regime_change'] * 
                                     df['entropy_trend_regime_shift'] * 
                                     df['price_entropy'])
    
    df['entropy_transition_strength'] = (df['entropy_volume_breakout'] * 
                                        df['entropy_range_breakout'] * 
                                        df['volume_entropy'])
    
    df['entropy_transition_quality'] = (df['entropy_transition_strength'] * 
                                       df['entropy_price_efficiency'] * 
                                       df['price_entropy'])
    
    # Entropy Asymmetric Divergence Patterns
    bullish_condition = (df['close'] > df['open']) & (df['volume'] > df['volume'].shift(1))
    bearish_condition = (df['close'] < df['open']) & (df['volume'] > df['volume'].shift(1))
    
    df['entropy_bullish_divergence'] = np.where(bullish_condition, 
                                               df['entropy_closing_divergence'] * df['price_entropy'], 0)
    df['entropy_bearish_divergence'] = np.where(bearish_condition, 
                                               df['entropy_closing_divergence'] * df['price_entropy'], 0)
    df['entropy_divergence_asymmetry'] = (df['entropy_bullish_divergence'] - 
                                         df['entropy_bearish_divergence']) * df['price_entropy']
    
    # Entropy Volume-Weighted Asymmetry
    df['entropy_high_volume_divergence'] = (df['entropy_divergence_patterns'] * 
                                           (df['volume'] / (volume_avg_4d + 1e-8)) * 
                                           df['volume_entropy'])
    
    df['entropy_low_volume_divergence'] = (df['entropy_divergence_patterns'] * 
                                          (volume_avg_4d / (df['volume'] + 1e-8)) * 
                                          df['volume_entropy'])
    
    df['entropy_volume_asymmetry_ratio'] = (df['entropy_high_volume_divergence'] / 
                                           (df['entropy_low_volume_divergence'] + 1e-8) * 
                                           df['volume_entropy'])
    
    # Core Entropy Divergence Components
    df['primary_entropy_divergence'] = (df['entropy_divergence_asymmetry'] * 
                                       df['entropy_volume_asymmetry_ratio'] * 
                                       df['price_entropy'])
    
    df['entropy_transition_enhancement'] = (df['primary_entropy_divergence'] * 
                                           df['entropy_transition_strength'] * 
                                           df['volume_entropy'])
    
    df['entropy_quality_overlay'] = (df['entropy_transition_enhancement'] * 
                                    df['entropy_price_efficiency'] * 
                                    df['price_entropy'])
    
    # Entropy Volume-Validation Layer
    df['entropy_volume_confirmed_divergence'] = (df['primary_entropy_divergence'] * 
                                                df['entropy_volume_confirmation'] * 
                                                df['volume_entropy'])
    
    df['entropy_multi_timeframe_alignment'] = (df['entropy_volume_confirmed_divergence'] * 
                                              df['entropy_regime_alignment'] * 
                                              df['price_entropy'])
    
    df['entropy_volume_quality_integration'] = (df['entropy_multi_timeframe_alignment'] * 
                                               df['entropy_transition_quality'] * 
                                               df['volume_entropy'])
    
    # Entropy Risk Management Filters
    df['entropy_false_signal_protection'] = ((1 - np.abs(df['entropy_divergence_consistency'])) * 
                                            df['price_entropy'])
    
    df['entropy_volume_stability'] = (df['volume'] / (volume_avg_4d + 1e-8) * 
                                     df['volume_entropy'])
    
    df['entropy_consistency_score'] = (np.abs(df['entropy_divergence_consistency']) * 
                                      df['entropy_volume_stability'] * 
                                      df['price_entropy'])
    
    # Final Entropy-Enhanced Alpha Synthesis
    df['main_entropy_alpha'] = (df['entropy_volume_quality_integration'] * 
                               df['entropy_consistency_score'] * 
                               df['price_entropy'])
    
    # Entropy Risk Controls
    df['entropy_volume_buffer'] = (df['main_entropy_alpha'] * 
                                  df['entropy_volume_stability'] * 
                                  df['volume_entropy'])
    
    df['entropy_quality_floor'] = (np.maximum(df['entropy_price_efficiency'], 0.2) * 
                                  df['price_entropy'])
    
    df['entropy_consistency_check'] = (df['main_entropy_alpha'] * 
                                      df['entropy_divergence_consistency'] * 
                                      df['price_entropy'])
    
    # Final factor with risk controls
    final_factor = (df['main_entropy_alpha'] * 
                   df['entropy_volume_buffer'] * 
                   df['entropy_quality_floor'] * 
                   df['entropy_consistency_check'])
    
    # Clean up intermediate columns
    cols_to_drop = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'amount', 'volume']]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    return final_factor
