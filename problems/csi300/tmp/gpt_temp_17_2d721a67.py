import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Fractal Regime Divergence with Adaptive Efficiency alpha factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Helper function for fractal dimension approximation (Hurst exponent-like)
    def fractal_dimension(series, window=5):
        """Approximate fractal dimension using range-based method"""
        ranges = series.rolling(window).max() - series.rolling(window).min()
        log_ranges = np.log(ranges.replace(0, np.nan))
        log_window = np.log(window)
        return 2 - (log_ranges / log_window)
    
    # Calculate fractal dimensions
    data['price_fractal_dim'] = fractal_dimension(data['close'], 5)
    data['volume_fractal_dim'] = fractal_dimension(data['volume'], 5)
    
    # Volume fractal persistence (autocorrelation-like measure)
    data['volume_fractal_persistence'] = data['volume'].rolling(5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Multi-timeframe fractal coherence
    def fractal_coherence(series, windows=[3, 5, 10]):
        """Measure coherence across multiple timeframes"""
        fractals = [fractal_dimension(series, w) for w in windows]
        coherence = pd.concat(fractals, axis=1).std(axis=1)
        return 1 / (1 + coherence)
    
    data['multi_timeframe_coherence'] = fractal_coherence(data['close'])
    
    # Volume clustering intensity
    data['volume_clustering'] = data['volume'].rolling(10).apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-8)
    )
    
    # Short-term fractal divergence (3-day)
    data['price_fractal_momentum_3d'] = ((data['close'] - data['close'].shift(3)) / 
                                        (data['close'].shift(3) + 1e-8)) * data['price_fractal_dim']
    data['volume_fractal_momentum_3d'] = ((data['volume'] - data['volume'].shift(3)) / 
                                         (data['volume'].shift(3) + 1e-8)) * data['volume_fractal_dim']
    data['fractal_divergence_score'] = data['price_fractal_momentum_3d'] - data['volume_fractal_momentum_3d']
    
    # Medium-term fractal divergence (10-day)
    data['price_fractal_trend_10d'] = ((data['close'] - data['close'].shift(10)) / 
                                      (data['close'].shift(10) + 1e-8)) * data['multi_timeframe_coherence']
    data['volume_fractal_trend_10d'] = ((data['volume'] - data['volume'].shift(10)) / 
                                       (data['volume'].shift(10) + 1e-8)) * data['volume_clustering']
    data['fractal_trend_divergence'] = data['price_fractal_trend_10d'] - data['volume_fractal_trend_10d']
    
    # Fractal divergence acceleration
    data['fractal_divergence_change'] = data['fractal_divergence_score'] - data['fractal_trend_divergence']
    
    # Volume acceleration
    data['volume_acceleration'] = (data['volume'] - 2 * data['volume'].shift(1) + data['volume'].shift(2))
    data['volume_fractal_acceleration'] = data['volume_fractal_persistence'] * data['volume_acceleration']
    data['fractal_alignment'] = np.sign(data['fractal_divergence_score']) * np.sign(data['volume_fractal_acceleration'])
    
    # Fractal pressure analysis
    data['upside_fractal_pressure'] = (data['high'] - data['open']) * data['volume'] * data['price_fractal_dim']
    data['downside_fractal_pressure'] = (data['open'] - data['low']) * data['volume'] * data['price_fractal_dim']
    data['fractal_net_pressure'] = ((data['upside_fractal_pressure'] - data['downside_fractal_pressure']) / 
                                   (data['upside_fractal_pressure'] + data['downside_fractal_pressure'] + 1e-8))
    
    # Amount fractal analysis
    data['fractal_amount_efficiency'] = (data['amount'] / (data['volume'] + 1e-8)) * data['volume_fractal_dim']
    data['amount_fractal_momentum'] = ((data['amount'] - data['amount'].shift(5)) / 
                                     (data['amount'].shift(5) + 1e-8)) * data['volume_fractal_persistence']
    data['amount_volume_fractal_divergence'] = data['amount_fractal_momentum'] - data['volume_fractal_momentum_3d']
    
    # Intraday efficiency
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['fractal_pressure_efficiency'] = data['fractal_net_pressure'] * data['intraday_efficiency']
    data['amount_fractal_confirmation'] = data['amount_fractal_momentum'] * data['fractal_alignment']
    data['multi_scale_fractal_consistency'] = data['fractal_divergence_change'] * data['multi_timeframe_coherence']
    
    # Fractal regime detection
    data['price_fractal_avg_5d'] = data['price_fractal_dim'].rolling(5).mean()
    data['high_fractal_condition'] = (data['price_fractal_dim'] > data['price_fractal_avg_5d']).astype(int)
    data['low_fractal_condition'] = (data['price_fractal_dim'] <= data['price_fractal_avg_5d']).astype(int)
    
    # Calculate regime persistence
    def regime_persistence(condition_series):
        persistence = pd.Series(index=condition_series.index, dtype=float)
        current_streak = 0
        for i, val in enumerate(condition_series):
            if i == 0 or val != condition_series.iloc[i-1]:
                current_streak = 1
            else:
                current_streak += 1
            persistence.iloc[i] = current_streak
        return persistence
    
    data['fractal_regime_persistence'] = regime_persistence(data['high_fractal_condition'])
    
    # Volume regime
    data['volume_fractal_level'] = (data['volume_fractal_dim'] > data['volume_fractal_persistence']).astype(int)
    data['volume_regime_persistence'] = regime_persistence(data['volume_fractal_level'])
    data['fractal_volume_spike'] = ((data['volume'] / data['volume'].shift(1)) > 
                                   (2.0 * data['volume_fractal_dim'])).astype(int)
    
    # Regime-specific efficiency metrics
    data['fractal_intraday_efficiency'] = data['intraday_efficiency'] * data['price_fractal_dim']
    data['fractal_overnight_efficiency'] = (abs(data['open'] - data['close'].shift(1)) / 
                                           (data['high'] - data['low'] + 1e-8)) * data['multi_timeframe_coherence']
    data['total_fractal_efficiency'] = data['fractal_intraday_efficiency'] + data['fractal_overnight_efficiency']
    
    # Fractal reversal components
    data['price_fractal_reversal'] = ((data['close'] - data['close'].shift(1)) * 
                                     (data['close'].shift(1) - data['close'].shift(2)) * data['price_fractal_dim'])
    data['volume_fractal_reversal'] = ((data['volume'] - data['volume'].shift(1)) * 
                                      (data['volume'].shift(1) - data['volume'].shift(2)) * data['volume_fractal_dim'])
    data['fractal_reversal_strength'] = data['price_fractal_reversal'] * data['volume_fractal_reversal']
    
    # Fractal mean reversion
    data['price_fractal_mean_reversion'] = ((data['close'] - data['close'].shift(5)) / 
                                           data['close'].rolling(5).std()) * data['multi_timeframe_coherence']
    data['volume_fractal_mean_reversion'] = ((data['volume'] - data['volume'].shift(5)) / 
                                            data['volume'].rolling(5).std()) * data['volume_clustering']
    data['fractal_mean_reversion_score'] = data['price_fractal_mean_reversion'] * data['volume_fractal_mean_reversion']
    
    # Regime matrix construction
    def calculate_regime_factor(row):
        # High Fractal, High Volume regime
        if row['high_fractal_condition'] == 1 and row['volume_fractal_level'] == 1:
            return (0.4 * row['fractal_divergence_score'] + 
                    0.3 * row['total_fractal_efficiency'] + 
                    0.3 * row['fractal_reversal_strength'])
        
        # High Fractal, Low Volume regime
        elif row['high_fractal_condition'] == 1 and row['volume_fractal_level'] == 0:
            return (0.5 * row['fractal_trend_divergence'] + 
                    0.4 * row['fractal_overnight_efficiency'] + 
                    0.1 * row['amount_fractal_confirmation'])
        
        # Low Fractal, High Volume regime
        elif row['high_fractal_condition'] == 0 and row['volume_fractal_level'] == 1:
            return (0.3 * row['fractal_divergence_change'] + 
                    0.2 * row['fractal_intraday_efficiency'] + 
                    0.5 * row['amount_fractal_momentum'])
        
        # Low Fractal, Low Volume regime
        else:
            return (0.4 * row['fractal_divergence_change'] + 
                    0.3 * row['total_fractal_efficiency'] + 
                    0.3 * row['fractal_mean_reversion_score'])
    
    # Apply regime-based factor calculation
    data['regime_factor'] = data.apply(calculate_regime_factor, axis=1)
    
    # Apply regime persistence adjustment
    data['persistence_adjustment'] = np.tanh(data['fractal_regime_persistence'] / 10.0)
    data['adjusted_regime_factor'] = data['regime_factor'] * (1 + 0.2 * data['persistence_adjustment'])
    
    # Add breakout enhancement
    data['price_fractal_breakout'] = ((data['close'] > data['high'].shift(1)) & 
                                     (data['close'] > data['high'].shift(2)) & 
                                     (data['close'] > data['high'].shift(3))).astype(int) * data['price_fractal_dim']
    
    data['volume_fractal_surge'] = ((data['volume'] > 1.5 * data['volume'].rolling(3).mean())).astype(int) * data['volume_fractal_persistence']
    
    data['fractal_breakout_confirmation'] = data['price_fractal_breakout'] * data['volume_fractal_surge']
    
    # Final alpha with breakout enhancement
    data['final_alpha'] = (data['adjusted_regime_factor'] + 
                          0.2 * data['fractal_breakout_confirmation'] * data['fractal_divergence_score'] +
                          0.1 * data['fractal_pressure_efficiency'] * data['amount_fractal_momentum'])
    
    # Apply confidence scoring
    data['regime_confidence'] = data['fractal_regime_persistence'] * data['multi_scale_fractal_consistency']
    data['signal_confidence'] = data['fractal_alignment'] * data['fractal_breakout_confirmation']
    data['overall_confidence'] = data['regime_confidence'] * data['signal_confidence']
    
    # Final confidence-adjusted alpha
    data['confidence_adjusted_alpha'] = data['final_alpha'] * np.tanh(data['overall_confidence'])
    
    return data['confidence_adjusted_alpha']
