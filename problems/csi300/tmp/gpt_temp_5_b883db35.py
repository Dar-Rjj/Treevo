import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel interpretable alpha factors combining price, volume, and range information.
    """
    # Price-Volume Convergence Factor
    # Price Trend Component
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    mom_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_convergence = mom_5d * mom_10d
    
    # Volume Confirmation
    volume_trend_5d = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    
    # Volume persistence
    volume_changes = df['volume'].pct_change()
    same_sign_count = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            recent_changes = volume_changes.iloc[i-4:i+1]
            if len(recent_changes) > 0:
                current_sign = np.sign(volume_trend_5d.iloc[i]) if not pd.isna(volume_trend_5d.iloc[i]) else 0
                if current_sign != 0:
                    same_sign_count.iloc[i] = np.sum(np.sign(recent_changes) == current_sign)
                else:
                    same_sign_count.iloc[i] = 0
            else:
                same_sign_count.iloc[i] = 0
        else:
            same_sign_count.iloc[i] = 0
    
    volume_persistence = np.sign(volume_trend_5d) * same_sign_count
    volume_price_alignment = volume_trend_5d * mom_5d
    
    # Volatility Adjustment
    price_vol_10d = df['close'].rolling(window=10).std()
    volume_vol_10d = df['volume'].rolling(window=10).std()
    
    # Final Price-Volume Convergence Factor
    pv_convergence = (momentum_convergence * volume_price_alignment) / (price_vol_10d * volume_vol_10d + 1e-6)
    
    # Range Efficiency Momentum Factor
    # Price Efficiency Component
    net_movement_10d = abs(df['close'] - df['close'].shift(10))
    
    total_movement_10d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 10:
            total_movement_10d.iloc[i] = sum(abs(df['close'].iloc[j] - df['close'].iloc[j-1]) for j in range(i-9, i+1))
        else:
            total_movement_10d.iloc[i] = np.nan
    
    market_efficiency_ratio = net_movement_10d / (total_movement_10d + 1e-6)
    
    # Range Momentum Component
    normalized_range_5d_avg = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            ranges = [(df['high'].iloc[j] - df['low'].iloc[j]) / df['close'].iloc[j-1] if j > 0 else np.nan 
                     for j in range(i-4, i+1)]
            valid_ranges = [r for r in ranges if not pd.isna(r)]
            normalized_range_5d_avg.iloc[i] = np.mean(valid_ranges) if valid_ranges else np.nan
        else:
            normalized_range_5d_avg.iloc[i] = np.nan
    
    current_range = (df['high'] - df['low']) / df['close'].shift(1)
    range_expansion = current_range / normalized_range_5d_avg
    range_momentum = range_expansion * mom_5d
    
    # Combined Range Efficiency Factor
    range_efficiency_momentum = market_efficiency_ratio * range_momentum
    
    # Gap-Recovery Momentum Factor
    # Opening Gap Analysis
    morning_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    intraday_recovery = (df['close'] - df['open']) / df['open']
    gap_recovery_ratio = intraday_recovery / (abs(morning_gap) + 1e-6)
    
    # Volume Confirmation
    volume_5d_avg = df['volume'].rolling(window=5).mean().shift(1)
    gap_day_volume_intensity = df['volume'] / volume_5d_avg
    recovery_volume_pattern = np.sign(intraday_recovery) * gap_day_volume_intensity
    volume_weighted_recovery = gap_recovery_ratio * recovery_volume_pattern
    
    # Multi-day Persistence
    significant_gaps = abs(morning_gap) > morning_gap.rolling(window=20).std()
    consecutive_gap_days = pd.Series(index=df.index, dtype=float)
    recovery_consistency = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 3:
            # Count significant gaps in past 3 days
            gap_days = sum(significant_gaps.iloc[i-3:i+1]) if i >= 3 else 0
            consecutive_gap_days.iloc[i] = gap_days
            
            # Average recovery ratio for past gap days
            gap_indices = []
            for j in range(max(0, i-3), i+1):
                if significant_gaps.iloc[j]:
                    gap_indices.append(j)
            
            if gap_indices:
                recovery_ratios = [gap_recovery_ratio.iloc[idx] for idx in gap_indices if not pd.isna(gap_recovery_ratio.iloc[idx])]
                recovery_consistency.iloc[i] = np.mean(recovery_ratios) if recovery_ratios else 0
            else:
                recovery_consistency.iloc[i] = 0
        else:
            consecutive_gap_days.iloc[i] = 0
            recovery_consistency.iloc[i] = 0
    
    # Final Gap-Recovery Factor
    gap_recovery_momentum = volume_weighted_recovery * recovery_consistency
    
    # Asymmetric Volume-Price Factor
    # Directional Volume Analysis
    up_volume_days = pd.Series(index=df.index, dtype=float)
    down_volume_days = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 5:
            up_count = 0
            down_count = 0
            for j in range(i-4, i+1):
                if j > 0 and df['close'].iloc[j] > df['close'].iloc[j-1]:
                    up_count += 1
                elif j > 0 and df['close'].iloc[j] < df['close'].iloc[j-1]:
                    down_count += 1
            up_volume_days.iloc[i] = up_count
            down_volume_days.iloc[i] = down_count
        else:
            up_volume_days.iloc[i] = 0
            down_volume_days.iloc[i] = 0
    
    volume_direction_bias = (up_volume_days - down_volume_days) / 5
    
    # Magnitude-Weighted Signals
    up_day_avg_return = pd.Series(index=df.index, dtype=float)
    down_day_avg_return = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 5:
            up_returns = []
            down_returns = []
            for j in range(i-4, i+1):
                if j > 0:
                    daily_return = (df['close'].iloc[j] - df['close'].iloc[j-1]) / df['close'].iloc[j-1]
                    if df['close'].iloc[j] > df['close'].iloc[j-1]:
                        up_returns.append(daily_return)
                    elif df['close'].iloc[j] < df['close'].iloc[j-1]:
                        down_returns.append(daily_return)
            
            up_day_avg_return.iloc[i] = np.mean(up_returns) if up_returns else 0
            down_day_avg_return.iloc[i] = np.mean(down_returns) if down_returns else 0
        else:
            up_day_avg_return.iloc[i] = 0
            down_day_avg_return.iloc[i] = 0
    
    return_asymmetry = up_day_avg_return - down_day_avg_return
    
    # Combined Asymmetric Factor
    volume_direction_component = volume_direction_bias * mom_5d
    magnitude_component = return_asymmetry * volume_direction_bias
    asymmetric_volume_price = volume_direction_component * magnitude_component
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'pv_convergence': pv_convergence,
        'range_efficiency': range_efficiency_momentum,
        'gap_recovery': gap_recovery_momentum,
        'asymmetric_volume': asymmetric_volume_price
    })
    
    # Remove NaN values and standardize
    factors_clean = factors.fillna(0)
    final_factor = factors_clean.mean(axis=1)
    
    # Final standardization
    final_factor = (final_factor - final_factor.mean()) / (final_factor.std() + 1e-6)
    
    return final_factor
