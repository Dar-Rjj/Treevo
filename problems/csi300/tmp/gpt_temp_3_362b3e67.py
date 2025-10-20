import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volume-Weighted Price Efficiency Analysis
    # Price Efficiency Calculation
    data['close_return_10d'] = data['close'].pct_change(periods=10)
    data['high_low_range_10d'] = (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()) / data['close'].shift(10)
    data['efficiency_10d'] = data['close_return_10d'] / data['high_low_range_10d'].replace(0, np.nan)
    
    data['close_return_20d'] = data['close'].pct_change(periods=20)
    data['high_low_range_20d'] = (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()) / data['close'].shift(20)
    data['efficiency_20d'] = data['close_return_20d'] / data['high_low_range_20d'].replace(0, np.nan)
    
    # Efficiency persistence
    data['efficiency_persistence'] = data['efficiency_10d'] - data['efficiency_20d']
    
    # Volume-Weighted Efficiency Enhancement
    data['volume_ratio_10d'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['volume_ratio_20d'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    data['vw_efficiency_10d'] = data['efficiency_10d'] * data['volume_ratio_10d']
    data['vw_efficiency_20d'] = data['efficiency_20d'] * data['volume_ratio_20d']
    
    # Efficiency-volume divergence
    data['efficiency_trend'] = data['efficiency_10d'].rolling(window=5).mean() - data['efficiency_20d'].rolling(window=5).mean()
    data['volume_trend'] = data['volume_ratio_10d'] - data['volume_ratio_20d']
    data['efficiency_volume_divergence'] = data['efficiency_trend'] - data['volume_trend']
    
    # Regime-Based Volume Asymmetry
    # Market Regime Classification
    data['price_momentum_20d'] = data['close'].pct_change(periods=20)
    data['volatility_10d'] = data['close'].pct_change().rolling(window=10).std()
    data['volatility_clustering'] = data['volatility_10d'] / data['volatility_10d'].rolling(window=20).mean()
    
    # Regime classification
    trending_threshold = data['price_momentum_20d'].abs().rolling(window=50).quantile(0.7)
    ranging_threshold = data['volatility_clustering'].rolling(window=50).quantile(0.3)
    
    data['is_trending'] = (data['price_momentum_20d'].abs() > trending_threshold).astype(int)
    data['is_ranging'] = (data['volatility_clustering'] < ranging_threshold).astype(int)
    
    # Regime transitions
    data['efficiency_change'] = data['efficiency_10d'].diff(5)
    data['regime_transition'] = data['is_trending'].diff(3).abs() + data['is_ranging'].diff(3).abs()
    
    # Regime-Specific Volume Patterns
    data['price_change_daily'] = data['close'].pct_change()
    data['up_volume'] = np.where(data['price_change_daily'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['price_change_daily'] < 0, data['volume'], 0)
    
    data['up_volume_5d'] = data['up_volume'].rolling(window=5).sum()
    data['down_volume_5d'] = data['down_volume'].rolling(window=5).sum()
    data['volume_asymmetry'] = (data['up_volume_5d'] - data['down_volume_5d']) / (data['up_volume_5d'] + data['down_volume_5d']).replace(0, np.nan)
    
    # Volume concentration in ranging regimes
    volume_quantile_high = data['volume'].rolling(window=20).quantile(0.7)
    volume_quantile_low = data['volume'].rolling(window=20).quantile(0.3)
    data['high_volume_days'] = (data['volume'] > volume_quantile_high).rolling(window=10).sum()
    data['low_volume_days'] = (data['volume'] < volume_quantile_low).rolling(window=10).sum()
    data['volume_concentration'] = data['high_volume_days'] / (data['high_volume_days'] + data['low_volume_days']).replace(0, np.nan)
    
    # Regime-adaptive volume momentum
    data['volume_momentum_trending'] = data['volume_asymmetry'] * data['is_trending']
    data['volume_momentum_ranging'] = data['volume_concentration'] * data['is_ranging']
    data['regime_volume_momentum'] = data['volume_momentum_trending'] + data['volume_momentum_ranging']
    
    # Multi-Scale Price-Volume Coherence
    # Short-Term Coherence Analysis
    data['price_change_5d'] = data['close'].pct_change(periods=5)
    data['volume_change_5d'] = data['volume'].pct_change(periods=5)
    data['price_volume_corr_5d'] = data['price_change_daily'].rolling(window=5).corr(data['volume'].pct_change())
    
    data['volume_autocorr_5d'] = data['volume'].pct_change().rolling(window=5).apply(lambda x: x.autocorr(), raw=False)
    data['short_term_coherence'] = data['price_volume_corr_5d'].abs() * data['volume_autocorr_5d'].abs()
    
    # Medium-Term Coherence Analysis
    data['price_change_15d'] = data['close'].pct_change(periods=15)
    data['volume_change_15d'] = data['volume'].pct_change(periods=15)
    data['price_volume_corr_15d'] = data['price_change_daily'].rolling(window=15).corr(data['volume'].pct_change())
    
    data['volume_autocorr_15d'] = data['volume'].pct_change().rolling(window=15).apply(lambda x: x.autocorr(), raw=False)
    data['medium_term_coherence'] = data['price_volume_corr_15d'].abs() * data['volume_autocorr_15d'].abs()
    
    # Coherence divergence
    data['coherence_divergence'] = data['short_term_coherence'] - data['medium_term_coherence']
    
    # Efficiency-Volume Regime Integration
    # Regime-Efficiency Synthesis
    data['regime_efficiency_trending'] = data['efficiency_10d'] * data['is_trending']
    data['regime_efficiency_ranging'] = data['efficiency_20d'] * data['is_ranging']
    
    # Regime-specific efficiency thresholds
    trending_efficiency_threshold = data['efficiency_10d'].rolling(window=50).quantile(0.6)
    ranging_efficiency_threshold = data['efficiency_20d'].rolling(window=50).quantile(0.4)
    
    data['efficiency_signal_trending'] = np.where(data['efficiency_10d'] > trending_efficiency_threshold, 1, -1) * data['is_trending']
    data['efficiency_signal_ranging'] = np.where(data['efficiency_20d'] > ranging_efficiency_threshold, 1, -1) * data['is_ranging']
    
    data['regime_efficiency_momentum'] = data['efficiency_signal_trending'] + data['efficiency_signal_ranging']
    
    # Volume-Coherence Enhancement
    data['coherence_weight'] = (data['short_term_coherence'] + data['medium_term_coherence']) / 2
    data['weighted_volume_asymmetry'] = data['volume_asymmetry'] * data['coherence_weight']
    data['weighted_volume_concentration'] = data['volume_concentration'] * data['coherence_weight']
    
    # Multi-scale coherence confirmation
    data['coherence_confirmation'] = np.where(
        (data['short_term_coherence'] > data['short_term_coherence'].rolling(window=20).mean()) & 
        (data['medium_term_coherence'] > data['medium_term_coherence'].rolling(window=20).mean()), 1, 0
    )
    
    data['coherence_adjusted_volume'] = (data['weighted_volume_asymmetry'] + data['weighted_volume_concentration']) * data['coherence_confirmation']
    
    # Adaptive Alpha Generation
    # Dynamic Factor Weighting
    regime_weight = data['is_trending'] * 0.6 + data['is_ranging'] * 0.4
    coherence_strength = data['coherence_weight'].rolling(window=10).mean()
    efficiency_persistence_score = data['efficiency_persistence'].rolling(window=10).apply(lambda x: len(x[x > 0]) / len(x) if len(x) > 0 else 0.5, raw=False)
    
    # Multi-Dimensional Signal Integration
    efficiency_component = (data['vw_efficiency_10d'] + data['vw_efficiency_20d']) / 2
    regime_component = data['regime_volume_momentum'] * regime_weight
    coherence_component = data['coherence_adjusted_volume'] * coherence_strength
    
    # Combine dimensions with persistence weighting
    combined_signal = (
        efficiency_component * 0.4 * efficiency_persistence_score +
        regime_component * 0.35 +
        coherence_component * 0.25
    )
    
    # Apply volume-coherence confirmation filters
    valid_signal_mask = (
        (data['coherence_confirmation'] == 1) &
        (data['volume'].rolling(window=5).mean() > data['volume'].rolling(window=20).mean() * 0.7) &
        (data['efficiency_10d'].abs() < 5)  # Remove extreme outliers
    )
    
    # Final predictive output
    alpha_factor = combined_signal * valid_signal_mask.astype(float)
    
    # Clean and normalize
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=50).mean()) / alpha_factor.rolling(window=50).std()
    
    return alpha_factor
