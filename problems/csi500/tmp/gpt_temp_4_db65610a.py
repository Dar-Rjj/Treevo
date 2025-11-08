import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price features
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['range'] = df['high'] - df['low']
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Volatility-Regime Adaptive Framework
    df['atr_5'] = df['true_range'].rolling(window=5).mean()
    df['range_vol_5'] = df['range'].rolling(window=5).std()
    
    # Classify volatility regimes
    high_vol_threshold = df['atr_5'].rolling(window=20).quantile(0.7)
    low_vol_threshold = df['atr_5'].rolling(window=20).quantile(0.3)
    df['vol_regime'] = np.where(df['atr_5'] > high_vol_threshold, 2, 
                               np.where(df['atr_5'] < low_vol_threshold, 0, 1))
    
    # Dual-Period Acceleration Core
    df['mom_3'] = df['close'] / df['close'].shift(3) - 1
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['accel_3'] = df['mom_3'] - df['mom_3'].shift(3)
    df['accel_10'] = df['mom_10'] - df['mom_10'].shift(3)
    
    # Volume acceleration (2nd derivative)
    df['volume_ma_3'] = df['volume'].rolling(window=3).mean()
    df['volume_1st_deriv'] = df['volume_ma_3'] - df['volume_ma_3'].shift(3)
    df['volume_2nd_deriv'] = df['volume_1st_deriv'] - df['volume_1st_deriv'].shift(3)
    
    # Volume-Liquidity Depth Analysis
    df['volume_slope_5'] = (df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    ))
    df['amount_per_trade'] = df['amount'] / df['volume']
    df['price_impact'] = df['true_range'] / df['volume']
    
    # Price-Level Anchoring
    df['recent_high_10'] = df['high'].rolling(window=10).max()
    df['recent_low_10'] = df['low'].rolling(window=10).min()
    df['dist_from_high'] = (df['close'] - df['recent_high_10']) / df['recent_high_10']
    df['dist_from_low'] = (df['close'] - df['recent_low_10']) / df['recent_low_10']
    
    # Days since momentum direction change
    df['mom_direction'] = np.sign(df['mom_3'])
    df['mom_direction_change'] = (df['mom_direction'] != df['mom_direction'].shift(1)).astype(int)
    df['days_since_mom_change'] = df['mom_direction_change'].rolling(window=10, min_periods=1).apply(
        lambda x: len(x) - 1 - x[::-1].argmax() if x.any() else len(x), raw=False
    )
    
    # Dynamic Signal Blending
    # High-volatility component: volatility-scaled momentum
    high_vol_signal = (df['accel_3'] + df['accel_10']) / df['atr_5']
    
    # Low-volatility component: overnight returns
    df['overnight_ret'] = (df['open'] - df['prev_close']) / df['prev_close']
    low_vol_signal = df['overnight_ret'].rolling(window=3).mean()
    
    # Price Acceleration with Volume Confirmation
    price_accel_signal = (df['accel_3'] * 0.6 + df['accel_10'] * 0.4) * df['volume_2nd_deriv']
    
    # Combine signals based on volatility regime
    df['final_signal'] = np.where(
        df['vol_regime'] == 2,  # High volatility
        high_vol_signal * 0.7 + price_accel_signal * 0.3,
        np.where(
            df['vol_regime'] == 0,  # Low volatility
            low_vol_signal * 0.6 + price_accel_signal * 0.4,
            price_accel_signal * 0.5 + high_vol_signal * 0.3 + low_vol_signal * 0.2  # Normal volatility
        )
    )
    
    # Apply liquidity filters
    liquidity_score = (df['volume_slope_5'] * 0.4 + 
                      (1 / df['price_impact']) * 0.3 + 
                      df['amount_per_trade'] * 0.3)
    
    # Final factor with price anchoring adjustment
    final_factor = (df['final_signal'] * liquidity_score * 
                   (1 - abs(df['dist_from_high'] + df['dist_from_low']) / 2) * 
                   (1 / (1 + df['days_since_mom_change'] * 0.1)))
    
    return final_factor
