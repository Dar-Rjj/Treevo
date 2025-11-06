import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Efficiency with Regime-Adaptive Momentum factor
    """
    data = df.copy()
    
    # Fractal Market Structure Analysis
    # Price Fractal Dimension
    daily_range = (data['high'] - data['low']) / data['close']
    range_efficiency_5d = daily_range / daily_range.rolling(window=5).mean()
    
    # Hurst Exponent Approximation using Rescaled Range Analysis
    def hurst_approximation(series, window=20):
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative_deviations = deviations.cumsum()
            range_val = cumulative_deviations.max() - cumulative_deviations.min()
            std_val = window_data.std()
            
            if std_val > 0:
                rs_ratio = range_val / std_val
                hurst = np.log(rs_ratio) / np.log(window)
            else:
                hurst = 0.5
            hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=series.index)
    
    hurst_exp = hurst_approximation(data['close'], window=20)
    
    # Fractal Consistency Score
    fractal_stability_5d = 1 / (daily_range.rolling(window=5).std() + 1e-6)
    fractal_persistence_20d = hurst_exp.rolling(window=20).std()
    fractal_consistency = fractal_stability_5d / (fractal_persistence_20d + 1e-6)
    
    # Volume Fractal Pattern Recognition
    volume_median_10d = data['volume'].rolling(window=10).median()
    volume_spike = (data['volume'] > 2 * volume_median_10d).astype(int)
    volume_drought = (data['volume'] < 0.5 * volume_median_10d).astype(int)
    
    # Volume-Price Fractal Correlation
    compression_pattern = ((data['volume'] > volume_median_10d) & 
                          (daily_range < daily_range.rolling(window=10).mean())).astype(int)
    expansion_pattern = ((data['volume'] < volume_median_10d) & 
                        (daily_range > daily_range.rolling(window=10).mean())).astype(int)
    
    volume_range_divergence = (compression_pattern - expansion_pattern) * daily_range
    
    # Multi-scale Momentum Calculation
    micro_momentum = data['close'].pct_change(1)
    short_momentum = (data['close'] / data['close'].shift(3) - 1)
    medium_momentum = (data['close'] / data['close'].shift(8) - 1)
    long_momentum = (data['close'] / data['close'].shift(21) - 1)
    
    # Regime Identification
    price_ma_20 = data['close'].rolling(window=20).mean()
    trending_regime = ((data['close'] > price_ma_20) & 
                      (micro_momentum.rolling(window=5).std() < micro_momentum.rolling(window=20).std()) &
                      (hurst_exp < 0.6)).astype(int)
    
    mean_reversion_regime = ((hurst_exp > 0.7) & 
                            (micro_momentum.rolling(window=5).std() > micro_momentum.rolling(window=10).std()) &
                            (volume_spike.rolling(window=5).sum() > 2)).astype(int)
    
    transition_regime = ((hurst_exp.diff().abs() > 0.1) | 
                        (volume_spike.diff().abs() > 0) |
                        ((micro_momentum * short_momentum) < 0)).astype(int)
    
    # Regime-Dependent Momentum Weighting
    regime_momentum = (
        trending_regime * (0.4 * micro_momentum + 0.3 * short_momentum + 0.2 * medium_momentum + 0.1 * long_momentum) +
        mean_reversion_regime * (-0.4 * micro_momentum - 0.3 * short_momentum + 0.2 * medium_momentum + 0.1 * long_momentum) +
        transition_regime * (0.2 * micro_momentum + 0.2 * short_momentum + 0.3 * medium_momentum + 0.3 * long_momentum)
    )
    
    # Price Efficiency Assessment
    def price_efficiency_ratio(series, window=10):
        efficiency_ratios = []
        for i in range(len(series)):
            if i < window:
                efficiency_ratios.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            net_move = abs(window_data.iloc[-1] - window_data.iloc[0])
            total_move = abs(window_data.diff()).sum()
            
            if total_move > 0:
                efficiency = net_move / total_move
            else:
                efficiency = 1.0
            efficiency_ratios.append(efficiency)
        
        return pd.Series(efficiency_ratios, index=series.index)
    
    price_efficiency = price_efficiency_ratio(data['close'], window=10)
    
    # Market Inefficiency Score
    over_reaction = (micro_momentum.abs() > 2 * micro_momentum.rolling(window=20).std()).astype(int)
    under_reaction = ((micro_momentum.abs() < 0.5 * micro_momentum.rolling(window=20).std()) & 
                     (volume_spike == 1)).astype(int)
    inefficiency_score = over_reaction - under_reaction
    
    # Volume Efficiency Analysis
    volume_price_correlation = data['volume'].rolling(window=10).corr(data['close'])
    emotional_trading = ((volume_spike == 1) & (price_efficiency < 0.7)).astype(int)
    institutional_flow = ((volume_drought == 1) & (price_efficiency > 0.8)).astype(int)
    
    # Volume-weighted price impact
    volume_weighted_impact = (micro_momentum * data['volume']) / data['volume'].rolling(window=10).mean()
    
    # Efficiency Regime Classification
    high_efficiency = (price_efficiency > 0.8).astype(int)
    low_efficiency = (price_efficiency < 0.5).astype(int)
    mixed_efficiency = ((price_efficiency >= 0.5) & (price_efficiency <= 0.8)).astype(int)
    
    # Adaptive Signal Combination
    # Fractal-Regime Signal Weighting
    fractal_weight = np.where(hurst_exp < 0.6, 1.2, 
                             np.where(hurst_exp > 0.7, 0.8, 1.0))
    
    # Efficiency-Confidence Adjustment
    efficiency_confidence = (
        high_efficiency * 1.5 +
        mixed_efficiency * 1.0 +
        low_efficiency * 0.7
    )
    
    # Composite Factor Construction
    base_fractal_efficiency = (
        fractal_consistency * 0.3 +
        (1 - hurst_exp) * 0.2 +
        price_efficiency * 0.2 +
        (1 - abs(volume_range_divergence)) * 0.1 +
        volume_price_correlation * 0.2
    )
    
    # Apply regime-adaptive momentum overlay
    regime_adjusted_momentum = regime_momentum * fractal_weight
    
    # Final composite factor
    composite_factor = (
        base_fractal_efficiency * 0.6 +
        regime_adjusted_momentum * 0.3 +
        efficiency_confidence * 0.1
    )
    
    # Normalize the final factor
    composite_factor_normalized = (
        composite_factor - composite_factor.rolling(window=63).mean()
    ) / (composite_factor.rolling(window=63).std() + 1e-6)
    
    return composite_factor_normalized
