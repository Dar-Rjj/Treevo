import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function to identify gap days
    def is_gap_day(row):
        return abs(row['open'] / data['close'].shift(1).loc[row.name] - 1) > 0.001
    
    # Calculate gap days
    gap_mask = data.apply(is_gap_day, axis=1)
    
    # Multi-Timeframe Gap Analysis
    def calculate_gap_range(window):
        gap_days_in_window = gap_mask.rolling(window=window).sum()
        gap_ranges = []
        for i in range(len(data)):
            if i < window:
                gap_ranges.append(np.nan)
                continue
            window_data = data.iloc[i-window+1:i+1]
            gap_indices = gap_mask.iloc[i-window+1:i+1]
            if gap_indices.sum() == 0:
                gap_ranges.append(np.nan)
            else:
                gap_values = []
                for j in range(i-window+1, i+1):
                    if gap_mask.iloc[j]:
                        gap_val = abs(data['open'].iloc[j] / data['close'].iloc[j-1] - 1)
                        gap_values.append(gap_val)
                gap_ranges.append(np.mean(gap_values) if gap_values else np.nan)
        return pd.Series(gap_ranges, index=data.index)
    
    micro_gap_range = calculate_gap_range(3)
    meso_gap_range = calculate_gap_range(8)
    macro_gap_range = calculate_gap_range(21)
    
    # Gap Fractal Compression
    gap_micro_meso = micro_gap_range / meso_gap_range
    gap_meso_macro = meso_gap_range / macro_gap_range
    gap_fractal_regime = gap_micro_meso * gap_meso_macro
    
    # Volume-Price Gap Divergence
    def calculate_gap_volume_bias():
        up_gap_volume = np.where((gap_mask) & (data['close'] > data['open']), data['volume'], 0)
        down_gap_volume = np.where((gap_mask) & (data['close'] < data['open']), data['volume'], 0)
        
        up_gap_volume_series = pd.Series(up_gap_volume, index=data.index).rolling(21).sum()
        down_gap_volume_series = pd.Series(down_gap_volume, index=data.index).rolling(21).sum()
        
        gap_volume_bias = (up_gap_volume_series - down_gap_volume_series) / (up_gap_volume_series + down_gap_volume_series + 1e-8)
        return gap_volume_bias
    
    gap_volume_bias = calculate_gap_volume_bias()
    
    # Multi-Scale Gap Momentum
    def calculate_gap_momentum(period):
        returns = data['close'].pct_change(period)
        return pd.Series(np.where(gap_mask, returns, np.nan), index=data.index)
    
    short_term_gap_momentum = calculate_gap_momentum(5)
    medium_term_gap_momentum = calculate_gap_momentum(13)
    
    gap_volume_price_divergence = gap_volume_bias * (short_term_gap_momentum + medium_term_gap_momentum)
    
    # Gap Efficiency Dynamics
    def calculate_gap_efficiency(window):
        efficiency_values = []
        for i in range(len(data)):
            if i < window:
                efficiency_values.append(np.nan)
                continue
            window_data = data.iloc[i-window+1:i+1]
            gap_indices = gap_mask.iloc[i-window+1:i+1]
            if gap_indices.sum() == 0:
                efficiency_values.append(np.nan)
            else:
                efficiencies = []
                for j in range(i-window+1, i+1):
                    if gap_mask.iloc[j]:
                        high_low_range = data['high'].iloc[j] - data['low'].iloc[j]
                        if high_low_range > 0:
                            eff = abs(data['close'].iloc[j] - data['open'].iloc[j]) / high_low_range
                            efficiencies.append(eff)
                efficiency_values.append(np.mean(efficiencies) if efficiencies else np.nan)
        return pd.Series(efficiency_values, index=data.index)
    
    gap_efficiency_3d = calculate_gap_efficiency(3)
    gap_efficiency_7d = calculate_gap_efficiency(7)
    multi_scale_gap_efficiency = gap_efficiency_3d / gap_efficiency_7d
    
    # Gap Volume Timing
    gap_volume_intensity = pd.Series(np.where(gap_mask, data['volume'] / data['volume'].shift(1), np.nan), index=data.index)
    
    def calculate_gap_volume_persistence():
        persistence = []
        current_streak = 0
        for i in range(len(data)):
            if not gap_mask.iloc[i]:
                persistence.append(0)
                current_streak = 0
            else:
                if i == 0:
                    current_streak = 1
                else:
                    prev_dir = 1 if data['close'].iloc[i-1] > data['open'].iloc[i-1] else -1
                    curr_dir = 1 if data['close'].iloc[i] > data['open'].iloc[i] else -1
                    if prev_dir == curr_dir and gap_mask.iloc[i-1]:
                        current_streak += 1
                    else:
                        current_streak = 1
                persistence.append(current_streak)
        return pd.Series(persistence, index=data.index)
    
    gap_volume_persistence = calculate_gap_volume_persistence()
    gap_efficiency_pressure = multi_scale_gap_efficiency * gap_volume_intensity * gap_volume_persistence
    
    # Pressure Asymmetry with Gap Enhancement
    def calculate_gap_pressure_asymmetry():
        morning_pressure = np.where(gap_mask, 
                                   (data['high'] - data['open']) / (abs(data['open'] / data['close'].shift(1) - 1) + 1e-8), 
                                   np.nan)
        gap_fill_pressure = np.where(gap_mask,
                                    (data['close'] - data['open']) / (abs(data['open'] / data['close'].shift(1) - 1) + 1e-8),
                                    np.nan)
        pressure_asymmetry = morning_pressure - gap_fill_pressure
        return pd.Series(pressure_asymmetry, index=data.index)
    
    gap_pressure_asymmetry = calculate_gap_pressure_asymmetry()
    
    gap_pressure_2d = gap_pressure_asymmetry - gap_pressure_asymmetry.shift(2)
    gap_pressure_5d = gap_pressure_asymmetry - gap_pressure_asymmetry.shift(5)
    gap_pressure_convergence = gap_pressure_2d - gap_pressure_5d
    
    overnight_gap = data['open'] / data['close'].shift(1) - 1
    enhanced_gap_pressure = gap_pressure_convergence * overnight_gap * gap_volume_persistence * gap_fractal_regime
    
    # Volume-Cluster Enhanced Gap Correlation
    gap_turnover = np.where(gap_mask, data['volume'] * data['close'], np.nan)
    gap_turnover_median = pd.Series(gap_turnover, index=data.index).rolling(8).median()
    gap_turnover_cluster = gap_turnover > 2.5 * gap_turnover_median
    
    def calculate_gap_cluster_duration():
        duration = []
        current_cluster = 0
        for i in range(len(data)):
            if gap_turnover_cluster[i]:
                current_cluster += 1
            else:
                current_cluster = 0
            duration.append(current_cluster)
        return pd.Series(duration, index=data.index)
    
    gap_cluster_duration = calculate_gap_cluster_duration()
    
    def calculate_gap_correlation(short_period, long_period):
        short_returns = data['close'].pct_change(short_period)
        long_returns = data['close'].pct_change(long_period)
        correlation = short_returns.rolling(10).corr(long_returns)
        return pd.Series(np.where(gap_mask, correlation, np.nan), index=data.index)
    
    short_term_gap_corr = calculate_gap_correlation(2, 4)
    medium_term_gap_corr = calculate_gap_correlation(3, 6)
    gap_correlation_divergence = short_term_gap_corr - medium_term_gap_corr
    
    gap_turnover_momentum = pd.Series(gap_turnover, index=data.index).pct_change(3)
    volume_enhanced_gap_corr = gap_correlation_divergence * gap_turnover_momentum * gap_cluster_duration * gap_volume_bias
    
    # Adaptive Gap-Fractal Integration
    expanding_regime = (gap_micro_meso > 1.1) & (gap_meso_macro > 1.1)
    contracting_regime = (gap_micro_meso < 0.9) & (gap_meso_macro < 0.9)
    transition_regime = ~expanding_regime & ~contracting_regime
    
    # Initialize component scores
    divergence_component = gap_volume_price_divergence.fillna(0)
    efficiency_component = gap_efficiency_pressure.fillna(0)
    pressure_component = enhanced_gap_pressure.fillna(0)
    correlation_component = volume_enhanced_gap_corr.fillna(0)
    
    # Apply phase-optimized weights
    final_alpha = pd.Series(0.0, index=data.index)
    
    # Expanding regime weights
    final_alpha[expanding_regime] = (
        0.4 * divergence_component[expanding_regime] +
        0.3 * efficiency_component[expanding_regime] +
        0.2 * pressure_component[expanding_regime] +
        0.1 * correlation_component[expanding_regime]
    )
    
    # Contracting regime weights
    final_alpha[contracting_regime] = (
        0.1 * divergence_component[contracting_regime] +
        0.2 * efficiency_component[contracting_regime] +
        0.4 * pressure_component[contracting_regime] +
        0.3 * correlation_component[contracting_regime]
    )
    
    # Transition regime weights
    final_alpha[transition_regime] = (
        0.25 * divergence_component[transition_regime] +
        0.25 * efficiency_component[transition_regime] +
        0.25 * pressure_component[transition_regime] +
        0.25 * correlation_component[transition_regime]
    )
    
    # Volume Cluster Validation
    volume_multiplier = np.where(gap_turnover_cluster, 1.6, 1.0)
    volume_multiplier = np.where(gap_volume_persistence > 3, volume_multiplier * 1.3, volume_multiplier)
    
    # Apply final adjustments
    final_alpha = final_alpha * volume_multiplier
    final_alpha = final_alpha * gap_pressure_convergence.fillna(1)
    final_alpha = final_alpha * gap_volume_bias.fillna(1)
    
    return final_alpha
