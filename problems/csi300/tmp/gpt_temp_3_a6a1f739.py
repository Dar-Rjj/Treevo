import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Weighted Price Acceleration
    # Calculate Price Acceleration Component
    price_return_5d = data['close'] / data['close'].shift(5) - 1
    price_return_10d = data['close'] / data['close'].shift(10) - 1
    price_acceleration = price_return_5d - price_return_10d
    
    # Apply Volatility Weighting
    daily_range = (data['high'] - data['low']) / data['close']
    hist_volatility = daily_range.rolling(window=20, min_periods=10).mean()
    volatility_weighted_price_acc = price_acceleration / (hist_volatility + 1e-8)
    
    # Liquidity-Confirmed Volume Acceleration
    # Calculate Volume Acceleration Component
    volume_change_5d = data['volume'] / data['volume'].shift(5) - 1
    volume_change_10d = data['volume'] / data['volume'].shift(10) - 1
    volume_acceleration = volume_change_5d - volume_change_10d
    
    # Apply Liquidity Efficiency Confirmation
    volume_to_amount_t = data['volume'] / (data['amount'] + 1e-8)
    volume_to_amount_t_5 = data['volume'].shift(5) / (data['amount'].shift(5) + 1e-8)
    liquidity_change_ratio = volume_to_amount_t / (volume_to_amount_t_5 + 1e-8)
    liquidity_confirmed_vol_acc = volume_acceleration * liquidity_change_ratio
    
    # Divergence Signal Generation
    # Compute Acceleration Divergence
    acceleration_ratio = volatility_weighted_price_acc / (liquidity_confirmed_vol_acc + 1e-8)
    log_acceleration_ratio = np.log(np.abs(acceleration_ratio) + 1e-8) * np.sign(acceleration_ratio)
    
    # Assess Direction Alignment
    price_direction = np.sign(price_return_5d)
    volume_direction = np.sign(volume_change_5d)
    alignment_factor = np.where(price_direction == volume_direction, 1, -1)
    divergence_signal = log_acceleration_ratio * alignment_factor
    
    # Apply Signal Enhancement
    smoothed_signal = divergence_signal.rolling(window=3, min_periods=2).mean()
    
    # Filter by Volatility Threshold
    signal_abs = np.abs(smoothed_signal)
    signal_median = signal_abs.rolling(window=20, min_periods=10).median()
    volatility_filter = signal_abs > signal_median
    
    # Require consistent direction for 2 consecutive days
    direction_consistency = smoothed_signal.rolling(window=2).apply(
        lambda x: 1 if len(x) == 2 and x.iloc[0] * x.iloc[1] > 0 else 0, raw=False
    )
    
    enhanced_signal = smoothed_signal * volatility_filter * direction_consistency
    
    # Final Alpha Output
    # Apply Cross-Sectional Ranking
    ranked_signal = enhanced_signal.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False
    )
    
    # Scale to [-1, 1] range
    final_alpha = (ranked_signal * 2 - 1)
    
    return final_alpha
