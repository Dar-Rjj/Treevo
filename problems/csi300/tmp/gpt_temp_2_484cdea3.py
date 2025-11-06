import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-scaled momentum reversal, intraday extreme reversal,
    price-volume efficiency, and adaptive breakout quality.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Volatility-Scaled Momentum Reversal
    # Multi-Timeframe Momentum Blend
    data['ret_1d'] = data['close'].pct_change()
    data['momentum_5d'] = data['close'].pct_change(5)  # Short-term reversal
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1  # Medium-term momentum
    data['momentum_blend'] = data['momentum_10d'] - data['momentum_5d']
    
    # Dynamic Volatility Scaling
    data['range_vol'] = (data['high'] - data['low']) / data['close']
    data['range_vol_ma'] = data['range_vol'].rolling(window=10, min_periods=5).mean()
    data['close_vol'] = data['ret_1d'].abs()
    data['close_vol_ma'] = data['close_vol'].rolling(window=10, min_periods=5).mean()
    data['combined_vol'] = (data['range_vol_ma'] + data['close_vol_ma']) / 2 + 0.0001
    
    # Volume-Weighted Adjustment
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_trend'] = data['volume'] / data['volume_ma_5']
    
    # Volume-Price Correlation (5-day rolling)
    vol_price_corr = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            corr = window_data['volume'].corr(window_data['ret_1d'].abs())
            vol_price_corr.append(corr if not np.isnan(corr) else 0)
        else:
            vol_price_corr.append(0)
    data['vol_price_corr'] = vol_price_corr
    
    # Volatility-Scaled Momentum Factor
    vol_momentum_factor = data['momentum_blend'] / data['combined_vol'] * data['volume_trend']
    vol_momentum_factor = vol_momentum_factor * np.sign(data['vol_price_corr'])
    
    # Intraday Extreme Reversal with Volume Confirmation
    epsilon = 0.0001
    data['upper_shadow'] = (data['high'] - data['close']) / (data['high'] - data['low'] + epsilon)
    data['lower_shadow'] = (data['close'] - data['low']) / (data['high'] - data['low'] + epsilon)
    
    data['upper_extreme'] = (data['upper_shadow'] > 0.7).astype(int)
    data['lower_extreme'] = (data['lower_shadow'] > 0.7).astype(int)
    data['shadow_diff'] = data['lower_shadow'] - data['upper_shadow']
    
    # Volume Confirmation
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_surge'] = data['volume'] / data['volume_ma_10']
    data['volume_surge_flag'] = (data['volume_surge'] > 1.5).astype(int)
    
    # Intraday Extreme Factor
    intraday_factor = data['shadow_diff'] * data['volume_surge_flag'] * data['volume_surge']
    intraday_factor = intraday_factor / (data['combined_vol'] + epsilon)
    
    # Price-Volume Efficiency Ratio
    data['daily_efficiency'] = (data['close'] - data['close'].shift(1)).abs() / data['volume']
    data['cum_ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['cum_vol_5d'] = data['volume'].rolling(window=5, min_periods=3).sum()
    data['multi_day_efficiency'] = data['cum_ret_5d'].abs() / data['cum_vol_5d']
    
    data['efficiency_ma_5'] = data['daily_efficiency'].rolling(window=5, min_periods=3).mean()
    data['efficiency_ratio'] = data['daily_efficiency'] / (data['efficiency_ma_5'] + epsilon)
    
    # Volume Distribution Analysis
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['avg_trade_size_ma'] = data['avg_trade_size'].rolling(window=10, min_periods=5).mean()
    data['trade_size_ratio'] = data['avg_trade_size'] / (data['avg_trade_size_ma'] + epsilon)
    
    # Efficiency Factor
    efficiency_factor = data['efficiency_ratio'] * data['multi_day_efficiency']
    efficiency_factor = efficiency_factor * np.tanh(data['trade_size_ratio'])
    
    # Adaptive Breakout Quality Score
    # Breakout Event Identification
    data['high_5'] = data['high'].rolling(window=5, min_periods=3).max().shift(1)
    data['low_5'] = data['low'].rolling(window=5, min_periods=3).min().shift(1)
    data['high_10'] = data['high'].rolling(window=10, min_periods=5).max().shift(1)
    data['low_10'] = data['low'].rolling(window=10, min_periods=5).min().shift(1)
    
    data['breakout_5_up'] = (data['close'] > data['high_5']).astype(int)
    data['breakout_5_down'] = (data['close'] < data['low_5']).astype(int)
    data['breakout_10_up'] = (data['close'] > data['high_10']).astype(int)
    data['breakout_10_down'] = (data['close'] < data['low_10']).astype(int)
    
    data['breakout_mag_5_up'] = (data['close'] - data['high_5']) / data['high_5']
    data['breakout_mag_5_down'] = (data['low_5'] - data['close']) / data['low_5']
    data['breakout_mag_10_up'] = (data['close'] - data['high_10']) / data['high_10']
    data['breakout_mag_10_down'] = (data['low_10'] - data['close']) / data['low_10']
    
    # Volume Confirmation for Breakouts
    data['volume_breakout_flag'] = (data['volume'] > 1.2 * data['volume_ma_10']).astype(int)
    
    # Breakout Quality Score
    breakout_signal = (data['breakout_5_up'] * data['breakout_mag_5_up'] + 
                      data['breakout_10_up'] * data['breakout_mag_10_up'] -
                      data['breakout_5_down'] * data['breakout_mag_5_down'] - 
                      data['breakout_10_down'] * data['breakout_mag_10_down'])
    
    breakout_factor = breakout_signal * data['volume_breakout_flag'] * data['volume_surge']
    breakout_factor = breakout_factor / (data['combined_vol'] + epsilon)
    
    # Combine all factors with equal weights
    final_factor = (vol_momentum_factor.fillna(0) + 
                   intraday_factor.fillna(0) + 
                   efficiency_factor.fillna(0) + 
                   breakout_factor.fillna(0)) / 4
    
    return final_factor
