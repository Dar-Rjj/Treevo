import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Price-Volume Divergence with Microstructure Noise Filter
    """
    data = df.copy()
    
    # Calculate Realized Volatility
    data['daily_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['realized_vol'] = data['daily_range'].rolling(window=5, min_periods=3).sum()
    
    # Classify Volatility Regimes
    vol_20d_percentile_80 = data['realized_vol'].rolling(window=20, min_periods=10).quantile(0.8)
    vol_20d_percentile_20 = data['realized_vol'].rolling(window=20, min_periods=10).quantile(0.2)
    
    data['vol_regime'] = 'normal'
    data.loc[data['realized_vol'] > vol_20d_percentile_80, 'vol_regime'] = 'high'
    data.loc[data['realized_vol'] < vol_20d_percentile_20, 'vol_regime'] = 'low'
    
    # Price Direction Strength
    data['price_change_pct'] = abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['price_direction'] = np.sign(data['close'] - data['close'].shift(1)) * data['price_change_pct']
    
    # Volume Anomaly Detection
    vol_mean_20d = data['volume'].rolling(window=20, min_periods=10).mean()
    vol_std_20d = data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_zscore'] = (data['volume'] - vol_mean_20d) / vol_std_20d
    
    # Volume Change Persistence
    vol_ratio_t1 = data['volume'] / data['volume'].shift(1)
    vol_ratio_t2 = data['volume'].shift(1) / data['volume'].shift(2)
    data['volume_persistence'] = vol_ratio_t1 * vol_ratio_t2
    
    # Price-Volume Correlation Break
    data['price_change_quintile'] = pd.qcut(data['price_change_pct'], 5, labels=False, duplicates='drop')
    
    # Calculate expected volume by quintile (using rolling 60-day window)
    expected_volume = []
    for i in range(len(data)):
        if i >= 60:
            window_data = data.iloc[i-60:i]
            quintile_means = window_data.groupby('price_change_quintile')['volume'].mean()
            current_quintile = data.iloc[i]['price_change_quintile']
            if pd.notna(current_quintile) and current_quintile in quintile_means.index:
                expected_volume.append(quintile_means.loc[current_quintile])
            else:
                expected_volume.append(np.nan)
        else:
            expected_volume.append(np.nan)
    
    data['expected_volume'] = expected_volume
    data['volume_ratio'] = data['volume'] / data['expected_volume']
    
    # Direction-Volume Mismatch
    data['up_day_low_volume'] = ((data['close'] > data['close'].shift(1)) & 
                                (data['volume_zscore'] < -1)).astype(int)
    data['down_day_high_volume'] = ((data['close'] < data['close'].shift(1)) & 
                                   (data['volume_zscore'] > 1)).astype(int)
    
    # Microstructure Noise Filter
    data['spread_estimate'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    spread_median_5d = data['spread_estimate'].rolling(window=5, min_periods=3).median()
    
    # Price Reversals (simplified proxy)
    data['intraday_return_1'] = (data['close'] - data['open']) / data['open']
    data['intraday_return_2'] = (data['high'] - data['low']) / data['open']
    data['reversal_intensity'] = abs(data['intraday_return_1']) + abs(data['intraday_return_2'])
    
    # Calculate divergence components
    data['divergence_corr_break'] = data['volume_ratio'].fillna(1) - 1
    data['divergence_mismatch'] = data['down_day_high_volume'] - data['up_day_low_volume']
    
    # Noise adjustment
    high_noise = data['spread_estimate'] > spread_median_5d
    high_reversal = data['reversal_intensity'] > data['reversal_intensity'].rolling(window=10, min_periods=5).median()
    
    # Regime-Adaptive Factor Combination
    factor_values = []
    
    for i in range(len(data)):
        if i < 20:  # Need sufficient history
            factor_values.append(np.nan)
            continue
            
        current = data.iloc[i]
        regime = current['vol_regime']
        
        # Base components
        price_strength = current['price_direction'] if not pd.isna(current['price_direction']) else 0
        volume_anomaly = current['volume_zscore'] if not pd.isna(current['volume_zscore']) else 0
        corr_break = current['divergence_corr_break'] if not pd.isna(current['divergence_corr_break']) else 0
        mismatch = current['divergence_mismatch'] if not pd.isna(current['divergence_mismatch']) else 0
        persistence = current['volume_persistence'] if not pd.isna(current['volume_persistence']) else 1
        
        # Noise adjustment
        noise_multiplier = 1.0
        if high_noise.iloc[i] or high_reversal.iloc[i]:
            noise_multiplier = 0.5
        
        # Regime-specific weighting
        if regime == 'high':
            # Emphasize volume anomalies, reduce price direction, increase noise sensitivity
            factor = (0.2 * price_strength + 
                     0.5 * volume_anomaly + 
                     0.3 * corr_break + 
                     0.2 * mismatch) * noise_multiplier * 0.7
            
        elif regime == 'low':
            # Emphasize price breakouts, focus on volume persistence, relax noise filters
            factor = (0.6 * price_strength + 
                     0.2 * volume_anomaly + 
                     0.2 * corr_break + 
                     0.3 * mismatch) * persistence * 1.2
            
        else:  # normal regime
            # Balanced approach
            factor = (0.4 * price_strength + 
                     0.3 * volume_anomaly + 
                     0.3 * corr_break + 
                     0.2 * mismatch) * noise_multiplier
        
        factor_values.append(factor)
    
    result = pd.Series(factor_values, index=data.index, name='adaptive_divergence_factor')
    return result
