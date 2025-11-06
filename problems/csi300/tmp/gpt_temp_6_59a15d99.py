import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Cross-Association Momentum with Regime-Switching Volume Dynamics
    """
    data = df.copy()
    
    # Cross-Association Signal Generation
    # Price-Volume Association Divergence
    data['price_change'] = data['close'].diff()
    data['positive_volume'] = np.where(data['price_change'] > 0, data['volume'], 0)
    data['negative_volume'] = np.where(data['price_change'] < 0, data['volume'], 0)
    
    # Rolling sums for positive and negative volume
    data['pos_vol_sum_10d'] = data['positive_volume'].rolling(window=10, min_periods=5).sum()
    data['neg_vol_sum_10d'] = data['negative_volume'].rolling(window=10, min_periods=5).sum()
    
    # Association ratio with smoothing and avoiding division by zero
    data['association_ratio'] = (data['pos_vol_sum_10d'] + 1) / (data['neg_vol_sum_10d'] + 1)
    
    # Price-Range Efficiency Pattern
    data['daily_efficiency'] = (data['high'] - data['low']) / (np.abs(data['close'] - data['open']) + 0.0001)
    data['daily_efficiency'] = np.clip(data['daily_efficiency'], 0.1, 10)  # Bound extreme values
    
    # Efficiency trend using 5-day slope
    def calc_efficiency_slope(series):
        if len(series) < 3:
            return 0
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    data['efficiency_trend'] = data['daily_efficiency'].rolling(window=5, min_periods=3).apply(
        calc_efficiency_slope, raw=False
    )
    
    # Regime-Switching Volume Dynamics
    # Volume Regime Classification
    data['volume_median_20d'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_burst'] = data['volume'] > (2 * data['volume_median_20d'])
    data['volume_contraction'] = data['volume'] < (0.5 * data['volume_median_20d'])
    data['volume_normal'] = ~(data['volume_burst'] | data['volume_contraction'])
    
    # Calculate returns for different periods
    data['return_1d'] = data['close'].pct_change(1)
    data['return_2d'] = data['close'].pct_change(2)
    data['return_3d'] = data['close'].pct_change(3)
    data['return_5d'] = data['close'].pct_change(5)
    
    # Regime-Specific Momentum
    data['burst_momentum'] = np.where(data['volume_burst'], data['return_3d'], 0)
    data['contraction_momentum'] = np.where(data['volume_contraction'], data['return_5d'], 0)
    data['normal_momentum'] = np.where(data['volume_normal'], data['return_2d'], 0)
    
    # Combine regime momentum
    data['regime_momentum'] = data['burst_momentum'] + data['contraction_momentum'] + data['normal_momentum']
    
    # Multi-Dimensional Pattern Alignment
    # Temporal Alignment Scoring
    data['short_term_alignment'] = np.sign(data['association_ratio'] - 1) * np.sign(data['efficiency_trend'])
    
    # Medium-term alignment (regime momentum consistency)
    data['momentum_consistency'] = data['regime_momentum'].rolling(window=5, min_periods=3).std()
    data['momentum_consistency'] = 1 / (1 + np.abs(data['momentum_consistency']))  # Inverse of volatility
    
    # Cross-timeframe confirmation score
    data['confirmation_score'] = (data['short_term_alignment'] + data['momentum_consistency']) / 2
    
    # Magnitude Synchronization
    # Volume vs price movement correlation over 5 days
    def volume_price_corr(volume_series, price_series):
        if len(volume_series) < 3:
            return 0
        return np.corrcoef(volume_series, price_series)[0, 1]
    
    data['volume_price_corr'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: volume_price_corr(x, data.loc[x.index, 'return_1d']), raw=False
    )
    
    # Efficiency vs association ratio alignment
    data['efficiency_assoc_alignment'] = np.corrcoef(
        data['daily_efficiency'].rolling(window=5, min_periods=3).mean(),
        data['association_ratio'].rolling(window=5, min_periods=3).mean()
    )[0, 1]
    
    data['magnitude_sync'] = (np.nan_to_num(data['volume_price_corr']) + 
                             np.nan_to_num(data['efficiency_assoc_alignment'])) / 2
    
    # Dynamic Weighting Framework
    # Regime-Based Weight Assignment
    data['short_term_weight'] = np.where(data['volume_burst'], 0.7, 
                                       np.where(data['volume_contraction'], 0.3, 0.5))
    data['medium_term_weight'] = np.where(data['volume_burst'], 0.3, 
                                        np.where(data['volume_contraction'], 0.7, 0.5))
    
    # Signal Confidence Scoring
    data['pattern_consistency'] = ((data['short_term_alignment'] > 0).astype(int) + 
                                 (data['momentum_consistency'] > 0.5).astype(int) + 
                                 (data['magnitude_sync'] > 0).astype(int)) / 3
    
    data['magnitude_coherence'] = 1 - np.abs(data['volume_price_corr'] - data['efficiency_assoc_alignment'])
    data['temporal_persistence'] = data['confirmation_score'].rolling(window=3, min_periods=2).mean()
    
    data['signal_confidence'] = (data['pattern_consistency'] + 
                               data['magnitude_coherence'] + 
                               data['temporal_persistence']) / 3
    
    # Composite Factor Construction
    # Core momentum signal
    data['core_momentum'] = (data['association_ratio'] * 
                           data['efficiency_trend'] * 
                           data['regime_momentum'])
    
    # Dynamic adjustment with regime-specific weights
    data['weighted_alignment'] = (data['short_term_weight'] * data['short_term_alignment'] + 
                                data['medium_term_weight'] * data['momentum_consistency'])
    
    # Final factor construction
    data['factor'] = (data['core_momentum'] * 
                     (1 + data['weighted_alignment']) * 
                     (1 + data['signal_confidence']))
    
    # Normalize the factor
    data['factor_normalized'] = (data['factor'] - data['factor'].rolling(window=20, min_periods=10).mean()) / (
        data['factor'].rolling(window=20, min_periods=10).std() + 0.0001)
    
    return data['factor_normalized']
