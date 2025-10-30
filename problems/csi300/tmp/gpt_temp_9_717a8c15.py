import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Range Efficiency
    df['range_efficiency'] = abs(df['close'] - df['prev_close']) / df['true_range']
    
    # Range Efficiency Acceleration
    df['range_eff_diff1'] = df['range_efficiency'] - df['range_efficiency'].shift(1)
    df['range_eff_diff2'] = df['range_efficiency'].shift(1) - df['range_efficiency'].shift(2)
    df['range_efficiency_acceleration'] = df['range_eff_diff1'] - df['range_eff_diff2']
    
    # Close Price Acceleration
    df['close_diff1'] = df['close'] - df['close'].shift(1)
    df['close_diff2'] = df['close'].shift(1) - df['close'].shift(2)
    df['close_price_acceleration'] = df['close_diff1'] - df['close_diff2']
    
    # Local Extreme for Reversal
    df['rolling_high_5'] = df['high'].rolling(window=5, min_periods=5).max()
    df['rolling_low_5'] = df['low'].rolling(window=5, min_periods=5).min()
    df['close_vs_5d'] = df['close'] > df['close'].shift(5)
    df['local_extreme'] = np.where(df['close_vs_5d'], df['rolling_low_5'], df['rolling_high_5'])
    df['reversal'] = (df['close'] - df['local_extreme']) / df['local_extreme']
    
    # Liquidity Adjustment
    df['liquidity_raw'] = df['volume'] / (df['high'] - df['low'])
    df['liquidity_ma'] = df['liquidity_raw'].rolling(window=5, min_periods=5).mean()
    df['liquidity'] = df['liquidity_raw'] - df['liquidity_ma']
    df['adjusted_reversal'] = df['reversal'] * df['liquidity']
    
    # Volume Efficiency Analysis
    df['price_move_per_volume'] = (df['close'] - df['prev_close']) / df['volume']
    df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=20).mean()
    df['volume_spike'] = df['volume'] > (2 * df['volume_ma_20'].shift(1))
    
    # Volume-Weighted Momentum
    df['short_term_return'] = df['close'] / df['close'].shift(3) - 1
    df['long_term_return'] = df['close'] / df['close'].shift(8) - 1
    df['volume_binary'] = (df['volume'] > df['volume_ma_20'].shift(1)).astype(int)
    df['weighted_momentum'] = (0.7 * df['short_term_return'] + 0.3 * df['long_term_return']) * df['volume_binary']
    
    # Advanced Divergence Detection
    df['divergence_strength'] = np.sign(df['range_efficiency_acceleration']) * np.sign(df['close_price_acceleration'])
    
    # Volume-Efficiency Phase Analysis
    df['price_acceleration_rolling'] = df['close_price_acceleration'].rolling(window=5, min_periods=5).apply(lambda x: x.mean() if len(x) == 5 else np.nan)
    df['volume_efficiency_rolling'] = df['price_move_per_volume'].rolling(window=5, min_periods=5).apply(lambda x: x.mean() if len(x) == 5 else np.nan)
    
    rolling_corr = []
    for i in range(len(df)):
        if i >= 4:
            window_price = df['close_price_acceleration'].iloc[i-4:i+1]
            window_volume = df['price_move_per_volume'].iloc[i-4:i+1]
            if not window_price.isna().any() and not window_volume.isna().any():
                corr = np.corrcoef(window_price, window_volume)[0,1]
                rolling_corr.append(corr if not np.isnan(corr) else 0)
            else:
                rolling_corr.append(0)
        else:
            rolling_corr.append(0)
    
    df['rolling_correlation'] = rolling_corr
    df['phase_shift'] = np.sign(df['rolling_correlation']) * (1 - abs(df['rolling_correlation']))
    
    # Combine Acceleration Components
    df['weighted_acceleration'] = 0.6 * df['range_efficiency_acceleration'] + 0.4 * df['close_price_acceleration']
    
    # Acceleration Persistence
    df['accel_sign'] = np.sign(df['weighted_acceleration'])
    persistence = []
    current_streak = 0
    current_sign = 0
    
    for i in range(len(df)):
        if i == 0 or np.isnan(df['weighted_acceleration'].iloc[i]):
            persistence.append(0)
            current_streak = 0
            current_sign = 0
        else:
            current_sign_val = df['accel_sign'].iloc[i]
            if current_sign_val == current_sign and current_sign_val != 0:
                current_streak = min(current_streak + 1, 5)
            else:
                current_streak = 1 if current_sign_val != 0 else 0
                current_sign = current_sign_val
            persistence.append(current_streak)
    
    df['acceleration_persistence'] = persistence
    
    # Apply Volume-Weighted Confirmation
    df['volume_adjusted'] = df['weighted_acceleration'] * df['weighted_momentum']
    df['spike_adjusted'] = np.where(df['volume_spike'], df['volume_adjusted'] * 1.5, df['volume_adjusted'])
    
    # Incorporate Reversal and Divergence
    df['reversal_component'] = df['adjusted_reversal'] * df['divergence_strength']
    df['phase_adjustment'] = df['reversal_component'] * df['phase_shift']
    
    # Final Alpha Construction
    df['composite_score'] = df['spike_adjusted'] + df['phase_adjustment']
    df['alpha_factor'] = df['composite_score'] * np.minimum(df['acceleration_persistence'], 5)
    
    return df['alpha_factor']
