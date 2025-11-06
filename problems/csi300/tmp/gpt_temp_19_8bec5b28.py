import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate log returns and log volume changes
    data['log_close'] = np.log(data['close'])
    data['log_volume'] = np.log(data['volume'])
    
    # Price acceleration: 2nd derivative of log prices
    data['price_return_1'] = data['log_close'].diff(1)
    data['price_return_2'] = data['log_close'].diff(2)
    data['price_acceleration'] = data['price_return_1'] - data['price_return_2'].shift(1)
    
    # Volume acceleration: 2nd derivative of log volumes
    data['volume_change_1'] = data['log_volume'].diff(1)
    data['volume_change_2'] = data['log_volume'].diff(2)
    data['volume_acceleration'] = data['volume_change_1'] - data['volume_change_2'].shift(1)
    
    # Acceleration asymmetry counts
    data['positive_asymmetry_day'] = ((data['price_acceleration'] > 0) & 
                                     (data['volume_acceleration'] > 0)).astype(int)
    data['negative_asymmetry_day'] = ((data['price_acceleration'] < 0) & 
                                     (data['volume_acceleration'] < 0)).astype(int)
    
    # Volatility measures
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['vol_10d_avg'] = data['daily_range'].rolling(window=10, min_periods=5).mean()
    data['vol_20d_avg'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    
    # Volatility regime
    data['high_vol_regime'] = (data['vol_10d_avg'] > data['vol_20d_avg']).astype(int)
    
    # Volatility pattern similarity (20-day lookback)
    def compute_volatility_similarity(series, window=20):
        similarities = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            current_pattern = series.iloc[i-window:i].values
            if not np.all(np.isfinite(current_pattern)):
                similarities.iloc[i] = 0
                continue
            
            similarities_list = []
            for j in range(window, i-window+1):
                historical_pattern = series.iloc[j-window:j].values
                if len(historical_pattern) == window and np.all(np.isfinite(historical_pattern)):
                    corr = np.corrcoef(current_pattern, historical_pattern)[0,1]
                    if not np.isnan(corr):
                        similarities_list.append(corr)
            
            if similarities_list:
                similarities.iloc[i] = np.mean(similarities_list)
            else:
                similarities.iloc[i] = 0
        return similarities
    
    data['vol_similarity'] = compute_volatility_similarity(data['daily_range'])
    
    # Memory weight based on volatility similarity
    data['memory_weight'] = data['vol_similarity'].rolling(window=10, min_periods=5).mean()
    
    # Rolling asymmetry counts
    lookback_window = 10
    data['positive_asymmetry_count'] = data['positive_asymmetry_day'].rolling(
        window=lookback_window, min_periods=5).sum()
    data['negative_asymmetry_count'] = data['negative_asymmetry_day'].rolling(
        window=lookback_window, min_periods=5).sum()
    
    # Volume entropy during regime shifts
    data['regime_shift'] = data['high_vol_regime'].diff().abs()
    data['volume_std_5d'] = data['volume'].rolling(window=5, min_periods=3).std()
    data['volume_mean_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_entropy'] = data['volume_std_5d'] / (data['volume_mean_5d'] + 1e-8)
    
    # Regime-adaptive factors
    high_vol_mask = data['high_vol_regime'] == 1
    low_vol_mask = data['high_vol_regime'] == 0
    
    # High volatility factor
    data['high_vol_factor'] = 0.0
    data.loc[high_vol_mask, 'high_vol_factor'] = (
        (data['positive_asymmetry_count'] - data['negative_asymmetry_count']) / 
        lookback_window * data['memory_weight']
    ).where(high_vol_mask)
    
    # Low volatility factor
    data['low_vol_factor'] = 0.0
    data.loc[low_vol_mask, 'low_vol_factor'] = (
        (data['negative_asymmetry_count'] - data['positive_asymmetry_count']) / 
        lookback_window * data['memory_weight']
    ).where(low_vol_mask)
    
    # Combine factors with volume weighting
    data['combined_factor'] = (
        data['high_vol_factor'] + data['low_vol_factor']
    ) * (1 + data['volume_entropy'])
    
    # Regime persistence adjustment
    data['regime_persistence'] = data['high_vol_regime'].rolling(window=5, min_periods=3).mean()
    data['final_factor'] = data['combined_factor'] * (1 + 0.5 * data['regime_persistence'])
    
    # Clean up and return
    result = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
