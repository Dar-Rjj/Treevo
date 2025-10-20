import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Dynamics
    # Ultra-short momentum
    data['ultra_short_momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Short-medium momentum divergence
    data['short_medium_momentum_divergence'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(3) - 
                                               (data['close'] - data['close'].shift(8)) / data['close'].shift(8))
    
    # Momentum consistency (5-day correlation with linear trend)
    def momentum_consistency(close_series):
        if len(close_series) < 5:
            return np.nan
        x = np.arange(1, 6)
        y = close_series.values
        return np.corrcoef(x, y)[0, 1] if not np.isnan(y).any() else np.nan
    
    data['momentum_consistency'] = data['close'].rolling(window=5).apply(momentum_consistency, raw=False)
    
    # Volume-Momentum Integration
    data['volume_momentum_asymmetry'] = ((data['volume'] / data['volume'].shift(3) - 1) - 
                                        (data['volume'].shift(3) / data['volume'].shift(6) - 1))
    
    data['volume_weighted_momentum'] = ((data['close'] - data['close'].shift(1)) * data['volume'] / 
                                       (data['high'] - data['low']))
    
    data['volume_momentum_alignment'] = (np.sign(data['close'] - data['close'].shift(1)) * 
                                        (data['volume'] / data['volume'].shift(1) - 1))
    
    # Momentum Pattern Recognition
    data['momentum_acceleration'] = ((data['close'] - data['close'].shift(1)) - 
                                    (data['close'].shift(1) - data['close'].shift(2)))
    
    def momentum_regime_detection(close_series):
        if len(close_series) < 3:
            return np.nan
        up_count = sum(close_series.iloc[i] > close_series.iloc[i-1] for i in range(1, len(close_series)))
        down_count = sum(close_series.iloc[i] < close_series.iloc[i-1] for i in range(1, len(close_series)))
        return up_count - down_count
    
    data['momentum_regime_detection'] = data['close'].rolling(window=4).apply(momentum_regime_detection, raw=False)
    
    data['momentum_divergence'] = data['ultra_short_momentum'] - data['short_medium_momentum_divergence']
    
    # Dynamic Range Efficiency System
    data['intraday_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    data['range_compression'] = ((data['high'] - data['low']) / 
                                (data['high'] - data['low']).rolling(window=3).mean().shift(1))
    
    def range_persistence_calc(hl_series, ma_series):
        if len(hl_series) < 5:
            return np.nan
        count = sum(hl_series.iloc[i] > ma_series.iloc[i-1] for i in range(1, len(hl_series)))
        return count / 5
    
    hl_ma_5 = (data['high'] - data['low']).rolling(window=5).mean().shift(1)
    data['range_persistence'] = (data['high'] - data['low']).rolling(window=6).apply(
        lambda x: range_persistence_calc(x, hl_ma_5.loc[x.index]), raw=False)
    
    # Volume-Range Synchronization
    def volume_range_correlation(volume_series, hl_series):
        if len(volume_series) < 3:
            return np.nan
        return volume_series.corr(hl_series)
    
    data['volume_range_correlation'] = pd.Series([
        volume_range_correlation(data['volume'].iloc[i-2:i+1], (data['high'] - data['low']).iloc[i-2:i+1]) 
        for i in range(len(data))], index=data.index)
    
    data['volume_range_efficiency'] = ((data['high'] - data['low']) * data['volume'] / 
                                      data['volume'].rolling(window=5).mean().shift(1))
    
    data['volume_compression'] = (data['volume'] / data['volume'].rolling(window=3).mean().shift(1) - 
                                 (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=3).mean().shift(1))
    
    # Adaptive Regime Switching Mechanism
    data['short_term_volatility'] = data['close'].rolling(window=3).std()
    data['volatility_regime'] = (data['close'].rolling(window=3).std() / 
                                data['close'].rolling(window=8).std())
    data['range_volatility'] = (data['high'] - data['low']) / data['close']
    
    data['momentum_strength'] = abs(data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_consistency_regime'] = data['close'].rolling(window=5).var()
    data['trend_detection'] = ((data['close'] - data['close'].shift(5)) / 
                              (data['high'].shift(5) - data['low'].shift(5)))
    
    # Cross-Dimensional Signal Integration
    data['range_weighted_momentum'] = data['ultra_short_momentum'] * data['intraday_range_efficiency']
    
    def momentum_range_correlation(momentum_series, range_series):
        if len(momentum_series) < 5:
            return np.nan
        return momentum_series.corr(range_series)
    
    momentum_series = data['close'].pct_change()
    range_series = (data['high'] - data['low']) / data['close']
    data['momentum_range_correlation'] = pd.Series([
        momentum_range_correlation(momentum_series.iloc[i-4:i+1], range_series.iloc[i-4:i+1]) 
        for i in range(len(data))], index=data.index)
    
    data['efficiency_enhanced_momentum'] = data['short_medium_momentum_divergence'] * data['volume_range_efficiency']
    
    # Volume-Confirmed Signals
    data['volume_momentum_range_alignment'] = data['volume_momentum_alignment'] * data['intraday_range_efficiency']
    data['volume_compression_momentum'] = data['momentum_acceleration'] * data['volume_compression']
    data['range_volume_momentum'] = data['volume_weighted_momentum'] * data['range_compression']
    
    # Regime-Contextualized Integration
    data['volatility_aware_signals'] = data['momentum_divergence'] * data['volatility_regime']
    data['range_context_modulation'] = data['momentum_consistency'] * data['range_persistence']
    
    # Advanced Divergence Detection
    data['momentum_volume_divergence'] = data['ultra_short_momentum'] - data['volume_momentum_asymmetry']
    data['range_volume_divergence'] = data['intraday_range_efficiency'] - data['volume_range_correlation']
    data['multi_scale_divergence'] = data['short_medium_momentum_divergence'] - data['momentum_regime_detection']
    
    data['range_efficiency_divergence'] = data['intraday_range_efficiency'] - data['range_persistence']
    data['volume_efficiency_divergence'] = data['volume_range_efficiency'] - data['volume_compression']
    data['momentum_efficiency_divergence'] = data['momentum_consistency'] - data['intraday_range_efficiency']
    
    # Adaptive Alpha Synthesis
    # High Volatility Alpha components
    high_vol_primary = (data['volatility_aware_signals'] * data['volume_momentum_range_alignment'] * 
                       data['range_volume_momentum'])
    high_vol_secondary = (data['momentum_volume_divergence'] * data['range_efficiency_divergence'] * 
                         data['momentum_strength'])
    
    # Low Volatility Alpha components
    low_vol_primary = (data['range_weighted_momentum'] * data['volume_compression_momentum'] * 
                      data['momentum_consistency'])
    low_vol_secondary = (data['range_context_modulation'] * data['volume_efficiency_divergence'] * 
                        data['volume_momentum_alignment'])
    
    # Dynamic regime-based blending
    volatility_threshold_high = data['volatility_regime'].quantile(0.7)
    volatility_threshold_low = data['volatility_regime'].quantile(0.3)
    
    def dynamic_weighting(vol_regime, high_vol_primary, high_vol_secondary, low_vol_primary, low_vol_secondary):
        if vol_regime > volatility_threshold_high:
            # High volatility regime
            weight_high = 0.7
            weight_low = 0.3
        elif vol_regime < volatility_threshold_low:
            # Low volatility regime
            weight_high = 0.3
            weight_low = 0.7
        else:
            # Transition regime
            weight_high = 0.5
            weight_low = 0.5
        
        high_vol_alpha = weight_high * high_vol_primary + (1 - weight_high) * high_vol_secondary
        low_vol_alpha = weight_low * low_vol_primary + (1 - weight_low) * low_vol_secondary
        
        return weight_high * high_vol_alpha + weight_low * low_vol_alpha
    
    # Final alpha calculation
    alpha_series = pd.Series([
        dynamic_weighting(
            data['volatility_regime'].iloc[i],
            high_vol_primary.iloc[i],
            high_vol_secondary.iloc[i],
            low_vol_primary.iloc[i],
            low_vol_secondary.iloc[i]
        ) if not (pd.isna(data['volatility_regime'].iloc[i]) or 
                 pd.isna(high_vol_primary.iloc[i]) or 
                 pd.isna(low_vol_primary.iloc[i])) else np.nan
        for i in range(len(data))
    ], index=data.index)
    
    # Apply efficiency-based intensity calibration
    efficiency_intensity = 1 + data['intraday_range_efficiency'].abs()
    alpha_series = alpha_series * efficiency_intensity
    
    # Apply momentum persistence enhancement
    momentum_persistence = 1 + data['momentum_consistency'].abs().fillna(0)
    alpha_series = alpha_series * momentum_persistence
    
    # Filter extreme range volatility periods
    extreme_vol_filter = (data['range_volatility'] < data['range_volatility'].quantile(0.95))
    alpha_series = alpha_series * extreme_vol_filter.astype(float)
    
    return alpha_series
