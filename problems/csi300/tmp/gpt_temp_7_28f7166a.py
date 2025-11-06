import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['return_1d'] = data['close'] / data['prev_close'] - 1
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Gap Efficiency Analysis
    # Gap Size: |Open_t - Close_t-1| / AvgTrueRange_t-5_to_t-1
    data['gap_size'] = abs(data['open'] - data['prev_close'])
    data['avg_true_range_5d'] = data['true_range'].shift(1).rolling(window=5).mean()
    data['normalized_gap'] = data['gap_size'] / data['avg_true_range_5d']
    
    # Range Utilization: (High_t - Low_t) / |Open_t - Close_t-1|
    data['range_utilization'] = (data['high'] - data['low']) / data['gap_size'].replace(0, np.nan)
    data['range_utilization'] = data['range_utilization'].clip(upper=10)  # Cap extreme values
    
    # Efficiency Score: Gap Size × (1 - Range Utilization)
    data['gap_efficiency'] = data['normalized_gap'] * (1 - data['range_utilization'])
    
    # Intraday Absorption Strength
    # Absorption Ratio: (High_t - Low_t) / |Open_t - Close_t-1|
    data['absorption_ratio'] = data['range_utilization']
    
    # Rejection Signal: Gap Size × (1 - Absorption Ratio)
    data['rejection_signal'] = data['normalized_gap'] * (1 - data['absorption_ratio'])
    
    # Final Gap Efficiency Component
    data['gap_component'] = data['gap_efficiency'] + data['rejection_signal']
    
    # Volume-Price Acceleration Divergence
    # Price Acceleration
    data['return_t1_t'] = data['close'] / data['close'].shift(1) - 1
    data['return_t2_t1'] = data['close'].shift(1) / data['close'].shift(2) - 1
    data['return_t3_t2'] = data['close'].shift(2) / data['close'].shift(3) - 1
    data['return_t5_t3'] = data['close'].shift(3) / data['close'].shift(5) - 1
    
    data['recent_acceleration'] = data['return_t1_t'] - data['return_t2_t1']
    data['lagged_acceleration'] = data['return_t3_t2'] - data['return_t5_t3']
    data['acceleration_diff'] = data['recent_acceleration'] - data['lagged_acceleration']
    
    # Volume Confirmation
    data['volume_return_ratio'] = (data['volume'] / data['volume'].shift(1)) / (data['close'] / data['prev_close'])
    data['volume_return_ratio'] = data['volume_return_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Range Correlation
    def rolling_corr(window):
        if len(window) < 2:
            return np.nan
        volume_data = window['volume'].values
        true_range_data = window['true_range'].values
        if np.std(volume_data) == 0 or np.std(true_range_data) == 0:
            return 0
        return np.corrcoef(volume_data, true_range_data)[0, 1]
    
    # Calculate rolling correlation
    rolling_corrs = []
    for i in range(len(data)):
        if i < 5:
            rolling_corrs.append(np.nan)
            continue
        window_data = data.iloc[i-5:i+1][['volume', 'true_range']]
        rolling_corrs.append(rolling_corr(window_data))
    
    data['volume_range_corr'] = rolling_corrs
    
    # Volume-Price Mismatch
    data['volume_price_mismatch'] = data['volume_return_ratio'] * data['volume_range_corr']
    
    # Divergence Score
    data['divergence_score'] = data['acceleration_diff'] * data['volume_price_mismatch']
    
    # Amount-Based Order Flow Pressure
    # Tick Size Efficiency
    data['price_change_per_amount'] = (data['high'] - data['low']) / data['amount'].replace(0, np.nan)
    data['rolling_efficiency'] = data['price_change_per_amount'].shift(1).rolling(window=5).mean()
    data['efficiency_deviation'] = data['price_change_per_amount'] / data['rolling_efficiency'] - 1
    
    # Directional Pressure
    data['bullish_pressure'] = np.where(data['close'] > data['open'], data['amount'], 0)
    data['bearish_pressure'] = np.where(data['close'] < data['open'], data['amount'], 0)
    
    # Calculate rolling averages for normalization
    data['bullish_avg'] = data['bullish_pressure'].rolling(window=10).mean()
    data['bearish_avg'] = data['bearish_pressure'].rolling(window=10).mean()
    
    data['net_pressure'] = ((data['bullish_pressure'] - data['bearish_pressure']) / 
                           (data['bullish_avg'] + data['bearish_avg'])) * data['efficiency_deviation']
    
    # Regime-Weighted Integration
    # Volatility Regime Filter
    data['volatility_10d'] = data['return_1d'].shift(1).rolling(window=10).std()
    data['volatility_median_50d'] = data['return_1d'].shift(1).rolling(window=50).std().rolling(window=50).median()
    
    data['high_vol_regime'] = data['volatility_10d'] > data['volatility_median_50d']
    data['low_vol_regime'] = data['volatility_10d'] < data['volatility_median_50d']
    
    # Regime-specific weighting
    data['regime_weight'] = np.where(data['high_vol_regime'], 0.7, 
                                   np.where(data['low_vol_regime'], 1.3, 1.0))
    
    # Return Persistence Confidence
    def calculate_consistency(window):
        if len(window) < 2:
            return 0.5
        overnight_returns = (window['open'] / window['close'].shift(1)) - 1
        intraday_returns = (window['close'] / window['open']) - 1
        
        sign_matches = ((overnight_returns > 0) & (intraday_returns > 0)) | \
                      ((overnight_returns < 0) & (intraday_returns < 0))
        
        return sign_matches.sum() / len(sign_matches.dropna())
    
    # Calculate rolling consistency
    consistency_scores = []
    for i in range(len(data)):
        if i < 5:
            consistency_scores.append(0.5)
            continue
        window_data = data.iloc[i-5:i+1][['open', 'close']]
        consistency_scores.append(calculate_consistency(window_data))
    
    data['persistence_weight'] = consistency_scores
    
    # Final Factor Calculation
    # Base Signal: Gap Efficiency × Volume-Price Divergence × Amount Pressure
    data['base_signal'] = data['gap_component'] * data['divergence_score'] * data['net_pressure']
    
    # Regime Adjustment and Confidence Scaling
    data['final_factor'] = data['base_signal'] * data['regime_weight'] * data['persistence_weight']
    
    # Clean and return
    result = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
