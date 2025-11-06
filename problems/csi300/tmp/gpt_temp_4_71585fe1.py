import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Wave-Momentum Synthesis Framework alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Basic price and volume calculations
    data['clean_wave_return'] = data['close'] / data['close'].shift(1) - 1
    data['gap_wave_momentum'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_wave_momentum'] = data['close'] / data['open'] - 1
    data['intraday_wave_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['wave_volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Wave efficiency calculations
    data['short_term_wave_efficiency'] = (data['close'] - data['close'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2))
    data['wave_efficiency_gradient'] = data['intraday_wave_efficiency'] - data['short_term_wave_efficiency']
    data['wave_efficiency_momentum'] = data['intraday_wave_efficiency'] / data['intraday_wave_efficiency'].shift(1) - 1
    
    # Volume microstructure
    data['upside_wave_volume'] = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'])
    data['downside_wave_volume'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'])
    data['wave_volume_imbalance'] = (data['upside_wave_volume'] - data['downside_wave_volume']) / data['volume']
    data['wave_volume_per_price_unit'] = data['volume'] / (data['high'] - data['low'])
    
    # Wave acceleration patterns
    data['return_wave_acceleration'] = data['clean_wave_return'] - (data['close'].shift(1) / data['close'].shift(2) - 1)
    
    # Wave efficiency-weighted momentum
    data['wave_efficiency_weighted_momentum'] = data['clean_wave_return'] * data['intraday_wave_efficiency']
    data['wave_volume_confirmed_efficiency'] = data['intraday_wave_efficiency'] * data['wave_volume_per_price_unit']
    data['wave_momentum_gradient'] = data['return_wave_acceleration'] * data['wave_efficiency_gradient']
    
    # Wave divergence signals
    data['efficiency_return_wave_divergence'] = np.sign(data['intraday_wave_efficiency']) * np.sign(data['clean_wave_return'])
    data['volume_efficiency_wave_divergence'] = np.sign(data['wave_volume_per_price_unit']) * np.sign(data['wave_efficiency_momentum'])
    data['efficiency_gradient_wave_divergence'] = np.sign(data['wave_efficiency_gradient']) * np.sign(data['return_wave_acceleration'])
    
    # Enhanced wave signals
    data['confirmed_wave_efficiency_momentum'] = data['wave_efficiency_weighted_momentum'] * data['efficiency_return_wave_divergence']
    data['volume_aligned_wave_efficiency'] = data['wave_volume_confirmed_efficiency'] * data['volume_efficiency_wave_divergence']
    data['gradient_confirmed_wave_acceleration'] = data['wave_momentum_gradient'] * data['efficiency_gradient_wave_divergence']
    
    # Wave persistence calculations
    def calculate_persistence(series, window=3):
        persistence = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window:
                window_data = series.iloc[i-window+1:i+1]
                if len(window_data) == window:
                    same_sign_count = sum(np.sign(window_data.iloc[j]) == np.sign(window_data.iloc[j-1]) 
                                        for j in range(1, len(window_data)))
                    persistence.iloc[i] = same_sign_count / (window - 1)
        return persistence
    
    # Calculate persistence measures
    data['wave_efficiency_persistence'] = calculate_persistence(data['wave_efficiency_momentum'])
    data['wave_volume_persistence'] = calculate_persistence(data['wave_volume_momentum'])
    
    # Multi-scale wave factors
    data['short_term_wave_factor'] = data['confirmed_wave_efficiency_momentum'] * data['wave_efficiency_persistence']
    data['medium_term_wave_factor'] = data['volume_aligned_wave_efficiency'] * data['wave_volume_persistence']
    data['long_term_wave_factor'] = data['gradient_confirmed_wave_acceleration'] * data['wave_efficiency_persistence']
    
    # Wave regime calculations
    data['short_term_volatility'] = data['clean_wave_return'].rolling(window=5, min_periods=3).std()
    data['medium_term_volatility'] = data['clean_wave_return'].rolling(window=10, min_periods=5).std()
    data['wave_volatility_momentum'] = data['short_term_volatility'] / data['medium_term_volatility'] - 1
    
    # Volume regime
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['wave_volume_surge'] = (data['volume'] > 1.3 * data['volume_ma_5']).astype(float)
    data['wave_volume_drought'] = (data['volume'] < 0.8 * data['volume_ma_5']).astype(float)
    data['wave_volume_regime'] = data['wave_volume_surge'] - data['wave_volume_drought']
    
    # Efficiency regime
    data['high_wave_efficiency'] = (np.abs(data['intraday_wave_efficiency']) > 0.7).astype(float)
    data['low_wave_efficiency'] = (np.abs(data['intraday_wave_efficiency']) < 0.3).astype(float)
    data['wave_efficiency_regime'] = data['high_wave_efficiency'] - data['low_wave_efficiency']
    
    # Regime persistence
    def regime_persistence(series, window=3):
        persistence = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window:
                window_data = series.iloc[i-window+1:i+1]
                if len(window_data) == window:
                    same_regime_count = sum(window_data.iloc[j] == window_data.iloc[j-1] 
                                          for j in range(1, len(window_data)))
                    persistence.iloc[i] = same_regime_count / (window - 1)
        return persistence
    
    data['wave_volume_consistency'] = regime_persistence(data['wave_volume_regime'])
    data['wave_efficiency_stability'] = regime_persistence(data['wave_efficiency_regime'])
    
    # Regime-adaptive volume
    regime_multiplier = 1 + 0.5 * data['wave_volume_regime']  # Boost for surge, reduce for drought
    data['wave_regime_adaptive_volume'] = data['wave_volume_confirmed_efficiency'] * regime_multiplier
    
    # Volume-regime divergence
    data['volume_regime_wave_divergence'] = np.sign(data['wave_volume_momentum']) * (data['wave_volume_regime'] - data['wave_volume_regime'].shift(1))
    data['regime_consistent_wave_volume'] = data['wave_regime_adaptive_volume'] * data['volume_regime_wave_divergence']
    
    # Multi-regime wave consistency
    divergence_signals = [
        data['efficiency_return_wave_divergence'],
        data['volume_efficiency_wave_divergence'], 
        data['efficiency_gradient_wave_divergence'],
        data['volume_regime_wave_divergence']
    ]
    data['multi_regime_wave_consistency'] = sum((signal > 0).astype(float) for signal in divergence_signals) / len(divergence_signals)
    
    # Regime wave factor
    data['regime_wave_factor'] = data['regime_consistent_wave_volume'] * data['multi_regime_wave_consistency']
    
    # Primary wave alpha components
    data['primary_wave_alpha'] = data['short_term_wave_factor'] * data['wave_volume_momentum']
    data['secondary_wave_alpha'] = data['medium_term_wave_factor'] * data['wave_efficiency_momentum']
    data['tertiary_wave_alpha'] = data['long_term_wave_factor'] * data['wave_volatility_momentum']
    
    # Composite wave alpha with regime-specific weights
    volatility_weight = 1 / (1 + np.abs(data['wave_volatility_momentum']))
    volume_weight = 1 + 0.3 * data['wave_volume_regime']  # Higher weight for volume surge
    efficiency_weight = 1 + 0.2 * data['wave_efficiency_regime']  # Higher weight for high efficiency
    
    data['composite_wave_alpha'] = (
        volatility_weight * data['primary_wave_alpha'] +
        volume_weight * data['secondary_wave_alpha'] +
        efficiency_weight * data['tertiary_wave_alpha'] +
        data['regime_wave_factor']
    ) / (volatility_weight + volume_weight + efficiency_weight + 1)
    
    # Wave factor persistence
    data['wave_factor_persistence'] = calculate_persistence(data['composite_wave_alpha'])
    
    # Final enhanced wave alpha
    data['persistence_weighted_wave_alpha'] = data['composite_wave_alpha'] * data['wave_factor_persistence']
    data['regime_stable_wave_alpha'] = data['persistence_weighted_wave_alpha'] * data['wave_efficiency_stability']
    data['volume_confirmed_wave_alpha'] = data['regime_stable_wave_alpha'] * data['wave_volume_consistency']
    data['efficiency_validated_wave_alpha'] = data['volume_confirmed_wave_alpha'] * data['wave_efficiency_stability']
    
    # Final alpha factor
    alpha = data['efficiency_validated_wave_alpha'].fillna(0)
    
    return alpha
