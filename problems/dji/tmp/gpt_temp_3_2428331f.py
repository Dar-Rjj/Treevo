import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Base Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Compute High-Low Momentum Component
    high_low_momentum = (df['high'] - df['low']) / df['open']
    
    # Combine Intraday and High-Low Momentum
    combined_momentum = (intraday_return + high_low_momentum) / 2
    
    # Apply Adjustments for Trading Activity
    volume_ratio = df['volume'] / df['volume'].shift(1)
    amount_ratio = df['amount'] / df['amount'].shift(1)
    
    # Volume Impact
    combined_momentum *= volume_ratio
    
    # Amount Impact
    combined_momentum *= amount_ratio * ((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1)))
    
    # Apply Volume Shock Filter
    volume_shock_threshold = 1.5
    volume_change_ratio = df['volume'] / df['volume'].shift(1, freq='D')
    combined_momentum = combined_momentum[volume_change_ratio > volume_shock_threshold]
    
    # Smoothing
    smoothed_combined_momentum = combined_momentum.ewm(span=5).mean()
    
    # Calculate Intraday Price Range
    intraday_range = df['high'] - df['low']
    
    # Measure Close Position in Range
    close_position_in_range = (df['close'] - df['low']) / (df['high'] - df['low'])
    close_position_in_range[intraday_range == 0] = 0
    
    # Calculate Volume-Weighted Average Price (VWAP)
    vwap = (df['open'] * df['volume'] + df['high'] * df['volume'] + df['low'] * df['volume'] + df['close'] * df['volume']) / (4 * df['volume'])
    
    # Calculate Intraday Momentum
    high_low_diff = df['high'] - df['low']
    open_close_return = (df['close'] - df['open']) / df['open']
    intraday_momentum = 0.5 * high_low_diff + 0.5 * open_close_return
    
    # Apply Volume and Amount Shock Filter
    volume_shock_threshold = 1.5
    amount_shock_threshold = 1.5
    volume_change_ratio = df['volume'] / df['volume'].shift(1, freq='D')
    amount_change_ratio = df['amount'] / df['amount'].shift(1, freq='D')
    filtered_intraday_momentum = intraday_momentum[(volume_change_ratio > volume_shock_threshold) & (amount_change_ratio > amount_shock_threshold)]
    
    # Combine Factors
    intraday_range_std = intraday_range.rolling(window=5).std()
    close_position_in_range_std = close_position_in_range.rolling(window=5).std()
    vwap_std = vwap.rolling(window=5).std()
    filtered_intraday_momentum_std = filtered_intraday_momentum.rolling(window=5).std()
    
    weights = 1 / pd.DataFrame({
        'intraday_range': intraday_range_std,
        'close_position_in_range': close_position_in_range_std,
        'vwap': vwap_std,
        'filtered_intraday_momentum': filtered_intraday_momentum_std
    }).sum(axis=1)
    
    final_alpha_factor = (
        smoothed_combined_momentum * weights +
        intraday_range * weights +
        close_position_in_range * weights +
        vwap * weights +
        filtered_intraday_momentum * weights
    )
    
    return final_alpha_factor
