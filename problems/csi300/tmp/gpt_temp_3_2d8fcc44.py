import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    data = df.copy()
    
    # Short-term Price Behavior Analysis
    # Recent Price Efficiency
    data['close_var_5d'] = data['close'].rolling(window=5).var()
    data['abs_price_change_5d'] = data['close'].diff().abs().rolling(window=5).mean()
    data['price_efficiency'] = data['close_var_5d'] / (data['abs_price_change_5d'] + 1e-8)
    
    # Price Reversal Intensity
    data['daily_return'] = data['close'].pct_change()
    data['prev_return'] = data['daily_return'].shift(1)
    data['reversal_intensity'] = data['daily_return'] * data['prev_return']
    
    # Opening Gap Persistence
    data['open_close_return'] = (data['close'] - data['open']) / data['open']
    data['prev_close_open_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = data['open_close_return'] * data['prev_close_open_return']
    
    # Medium-term Volume Structure Analysis
    # Volume Distribution Skew
    def volume_skew_calc(volumes):
        if len(volumes) < 10:
            return np.nan
        p25, p50, p75 = np.percentile(volumes, [25, 50, 75])
        return ((p75 - p50) - (p50 - p25)) / (p50 + 1e-8)
    
    data['volume_skew'] = data['volume'].rolling(window=10).apply(volume_skew_calc, raw=True)
    
    # Volume Clustering Patterns
    def high_volume_count(volumes):
        if len(volumes) < 10:
            return np.nan
        median_vol = np.median(volumes)
        high_vol_threshold = 1.5 * median_vol
        return np.sum(volumes > high_vol_threshold)
    
    def volume_concentration(volumes):
        if len(volumes) < 10:
            return np.nan
        sorted_volumes = np.sort(volumes)[::-1]
        top3_sum = np.sum(sorted_volumes[:3])
        total_sum = np.sum(volumes)
        return top3_sum / (total_sum + 1e-8)
    
    data['high_volume_count'] = data['volume'].rolling(window=10).apply(high_volume_count, raw=True)
    data['volume_concentration'] = data['volume'].rolling(window=10).apply(volume_concentration, raw=True)
    
    # Price-Volume Asymmetry Components
    # Up-Move vs Down-Move Volume Analysis
    def up_down_volume_asymmetry(df_window):
        if len(df_window) < 15:
            return np.nan
        up_days = df_window[df_window['close'] > df_window['close'].shift(1)]
        down_days = df_window[df_window['close'] < df_window['close'].shift(1)]
        
        up_volume_intensity = up_days['volume'].sum() / (len(up_days) + 1e-8)
        down_volume_intensity = down_days['volume'].sum() / (len(down_days) + 1e-8)
        
        volume_asymmetry = (up_volume_intensity / (down_volume_intensity + 1e-8)) - 1.0
        return volume_asymmetry
    
    # Large Move Volume Analysis
    def large_move_volume_ratio(df_window):
        if len(df_window) < 15:
            return np.nan
        
        large_up_mask = df_window['daily_return'] > 0.02
        large_down_mask = df_window['daily_return'] < -0.02
        
        large_up_volume = df_window.loc[large_up_mask, 'volume'].sum()
        large_down_volume = df_window.loc[large_down_mask, 'volume'].sum()
        avg_daily_volume = df_window['volume'].mean()
        
        return (large_up_volume - large_down_volume) / (avg_daily_volume + 1e-8)
    
    # Rolling calculations for price-volume asymmetry
    volume_asymmetry_values = []
    large_move_ratio_values = []
    
    for i in range(len(data)):
        if i < 15:
            volume_asymmetry_values.append(np.nan)
            large_move_ratio_values.append(np.nan)
            continue
        
        window_data = data.iloc[i-14:i+1].copy()
        volume_asymmetry_values.append(up_down_volume_asymmetry(window_data))
        large_move_ratio_values.append(large_move_volume_ratio(window_data))
    
    data['volume_asymmetry_ratio'] = volume_asymmetry_values
    data['large_move_volume_ratio'] = large_move_ratio_values
    
    # Market Regime Detection
    # Volatility Regime Classification
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_20d'] = data['daily_range'].rolling(window=20).std()
    volatility_median = data['volatility_20d'].median()
    
    def classify_volatility_regime(volatility):
        if pd.isna(volatility):
            return 'normal'
        if volatility > 1.5 * volatility_median:
            return 'high'
        elif volatility < 0.7 * volatility_median:
            return 'low'
        else:
            return 'normal'
    
    data['volatility_regime'] = data['volatility_20d'].apply(classify_volatility_regime)
    
    # Trend Regime Classification
    def compute_price_slope(prices):
        if len(prices) < 10:
            return np.nan
        x = np.arange(len(prices))
        slope, _, _, _, _ = stats.linregress(x, prices)
        return slope
    
    data['price_slope'] = data['close'].rolling(window=10).apply(compute_price_slope, raw=True)
    
    def classify_trend_regime(row):
        if pd.isna(row['price_slope']) or pd.isna(row['close']):
            return 'moderate'
        slope_threshold_strong = 0.002 * row['close']
        slope_threshold_weak = 0.0005 * row['close']
        
        if abs(row['price_slope']) > slope_threshold_strong:
            return 'strong'
        elif abs(row['price_slope']) < slope_threshold_weak:
            return 'weak'
        else:
            return 'moderate'
    
    data['trend_regime'] = data.apply(classify_trend_regime, axis=1)
    
    # Generate Adaptive Alpha Signal
    # Core Asymmetry Score
    data['core_score'] = (data['price_efficiency'] * data['reversal_intensity']) * data['gap_persistence']
    data['core_score'] = data['core_score'] * data['volume_skew'] * data['volume_concentration']
    
    # Asymmetry Multiplier
    data['asymmetry_multiplier'] = data['volume_asymmetry_ratio'] * data['large_move_volume_ratio']
    data['asymmetry_multiplier'] = np.power(data['asymmetry_multiplier'], 3)  # Cubic transformation
    
    # Regime-Based Adjustment
    def apply_regime_adjustments(row):
        signal = row['core_score'] * row['asymmetry_multiplier']
        
        # Volatility regime adjustments
        if row['volatility_regime'] == 'high':
            signal *= 0.7  # Reduce by 30%
        elif row['volatility_regime'] == 'low':
            signal *= 1.2  # Amplify by 20%
        
        # Trend regime adjustments
        if row['trend_regime'] == 'strong':
            # Momentum enhancement - amplify positive signals, dampen negative ones
            if signal > 0:
                signal *= 1.3
            else:
                signal *= 0.8
        elif row['trend_regime'] == 'weak':
            # Mean reversion enhancement - amplify negative signals, dampen positive ones
            if signal < 0:
                signal *= 1.3
            else:
                signal *= 0.8
        
        return signal
    
    data['raw_signal'] = data.apply(apply_regime_adjustments, axis=1)
    
    # Final scaling with logistic function
    def logistic_scaling(x):
        return 1 / (1 + np.exp(-x / (np.std(data['raw_signal'].dropna()) + 1e-8)))
    
    data['alpha_signal'] = data['raw_signal'].apply(logistic_scaling)
    
    return data['alpha_signal']
