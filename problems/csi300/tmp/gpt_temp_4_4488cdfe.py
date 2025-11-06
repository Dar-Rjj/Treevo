import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Price-Volume Divergence factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate TR percentiles for volatility regime classification
    data['tr_percentile_10'] = data['tr'].rolling(window=20, min_periods=10).quantile(0.1)
    data['tr_percentile_50'] = data['tr'].rolling(window=20, min_periods=10).quantile(0.5)
    data['tr_percentile_90'] = data['tr'].rolling(window=20, min_periods=10).quantile(0.9)
    
    # Classify volatility regimes
    data['vol_regime'] = 1  # Normal volatility by default
    data.loc[data['tr'] < data['tr_percentile_10'], 'vol_regime'] = 0  # Low volatility
    data.loc[data['tr'] > data['tr_percentile_90'], 'vol_regime'] = 2  # High volatility
    
    # Calculate VWAP for clustering analysis
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(window=5, min_periods=3).sum() / \
                   data['volume'].rolling(window=5, min_periods=3).sum()
    
    # Low Volatility Components
    # Price clustering around VWAP
    data['price_range'] = data['high'] - data['low']
    data['vwap_distance'] = abs(data['close'] - data['vwap'])
    data['near_vwap'] = (data['vwap_distance'] < 0.2 * data['price_range']).astype(int)
    data['clustering_score'] = data['near_vwap'].rolling(window=5, min_periods=3).mean()
    
    # Micro-structure noise ratio
    data['overnight_gap'] = abs(data['open'] / data['prev_close'] - 1)
    data['intraday_range'] = data['high'] / data['low'] - 1
    data['noise_ratio'] = data['overnight_gap'] / data['intraday_range'].replace(0, np.nan)
    
    # Normal Volatility Components
    # Price-momentum consistency
    data['price_change_3d'] = data['close'] / data['close'].shift(3) - 1
    data['daily_direction'] = np.sign(data['close'] - data['open'])
    data['momentum_direction'] = np.sign(data['price_change_3d'])
    
    # Calculate daily consistency over 3 days
    consistency_list = []
    for i in range(len(data)):
        if i >= 2:
            recent_data = data.iloc[i-2:i+1]
            consistent_days = sum(recent_data['daily_direction'] == recent_data['momentum_direction'].iloc[-1])
            consistency_list.append(consistent_days / 3)
        else:
            consistency_list.append(np.nan)
    data['momentum_consistency'] = consistency_list
    
    # Volume-volatility alignment
    data['volume_percentile'] = data['volume'].rolling(window=20, min_periods=10).rank(pct=True)
    data['volatility_percentile'] = data['tr'].rolling(window=20, min_periods=10).rank(pct=True)
    
    # Calculate rank correlation over 10 days
    vol_align_list = []
    for i in range(len(data)):
        if i >= 9:
            recent_data = data.iloc[i-9:i+1]
            if len(recent_data) >= 5:
                correlation = recent_data['volume_percentile'].corr(recent_data['volatility_percentile'])
                vol_align_list.append(correlation if not np.isnan(correlation) else 0)
            else:
                vol_align_list.append(0)
        else:
            vol_align_list.append(np.nan)
    data['volume_volatility_alignment'] = vol_align_list
    
    # High Volatility Components
    # Extreme price reversion probability
    data['returns'] = data['close'].pct_change()
    data['return_std'] = data['returns'].rolling(window=20, min_periods=10).std()
    data['extreme_move'] = abs(data['returns']) > (2 * data['return_std'])
    
    # Calculate empirical reversal rate
    reversal_list = []
    for i in range(len(data)):
        if i >= 5 and data['extreme_move'].iloc[i-1]:
            current_return = data['returns'].iloc[i]
            prev_return = data['returns'].iloc[i-1]
            if prev_return != 0:
                reversal = 1 if (current_return * prev_return) < 0 else 0
                reversal_list.append(reversal)
            else:
                reversal_list.append(0)
        else:
            reversal_list.append(np.nan)
    data['reversal_signal'] = reversal_list
    data['reversal_rate'] = data['reversal_signal'].rolling(window=10, min_periods=5).mean()
    
    # Volume exhaustion signals
    data['volume_median_10d'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['volume_surge_ratio'] = data['volume'] / data['volume_median_10d']
    
    # Multi-Timeframe Divergence Signals
    # Volume trend divergence
    data['volume_short_slope'] = data['volume'].rolling(window=3, min_periods=2).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0
    )
    data['volume_medium_slope'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / len(x) if len(x) > 1 else 0
    )
    data['volume_trend_divergence'] = np.sign(data['volume_short_slope']) * \
                                     np.sign(data['volume_medium_slope']) * \
                                     abs(data['volume_short_slope'] - data['volume_medium_slope'])
    
    # Price-range efficiency divergence
    data['realized_range'] = abs(data['close'] - data['open'])
    data['total_range'] = data['high'] - data['low']
    data['range_efficiency'] = data['realized_range'] / data['total_range'].replace(0, np.nan)
    
    # Expected range from recent volatility
    data['expected_range'] = data['tr'].rolling(window=5, min_periods=3).mean()
    data['range_divergence'] = data['total_range'] / data['expected_range'].replace(0, np.nan)
    
    # Combine signals with regime weights
    for i in range(len(data)):
        if i < 20:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[i]
        regime = current_data['vol_regime']
        
        if regime == 0:  # Low volatility
            # Emphasize micro-structure and clustering
            clustering_component = current_data['clustering_score'] if not np.isnan(current_data['clustering_score']) else 0
            noise_component = -current_data['noise_ratio'] if not np.isnan(current_data['noise_ratio']) else 0
            regime_score = 0.6 * clustering_component + 0.4 * noise_component
            
        elif regime == 1:  # Normal volatility
            # Emphasize momentum and alignment
            momentum_component = current_data['momentum_consistency'] if not np.isnan(current_data['momentum_consistency']) else 0
            alignment_component = current_data['volume_volatility_alignment'] if not np.isnan(current_data['volume_volatility_alignment']) else 0
            regime_score = 0.5 * momentum_component + 0.5 * alignment_component
            
        else:  # High volatility
            # Emphasize reversion and exhaustion
            reversion_component = current_data['reversal_rate'] if not np.isnan(current_data['reversal_rate']) else 0
            exhaustion_component = -current_data['volume_surge_ratio'] if not np.isnan(current_data['volume_surge_ratio']) else 0
            regime_score = 0.6 * reversion_component + 0.4 * exhaustion_component
        
        # Apply multi-timeframe divergence adjustments
        volume_div = current_data['volume_trend_divergence'] if not np.isnan(current_data['volume_trend_divergence']) else 0
        range_div = current_data['range_divergence'] - 1 if not np.isnan(current_data['range_divergence']) else 0
        
        divergence_adjustment = 0.3 * volume_div + 0.2 * range_div
        
        # Final alpha factor
        result.iloc[i] = regime_score + divergence_adjustment
    
    # Apply regime persistence filter (smooth transitions)
    result = result.rolling(window=3, min_periods=1).mean()
    
    return result
