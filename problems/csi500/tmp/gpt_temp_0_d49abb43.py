import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns and volatility
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Volatility Asymmetry
    # Upside volatility (close > open periods)
    upside_mask = df['close'] > df['open']
    downside_mask = df['close'] < df['open']
    
    # Calculate rolling volatility for upside and downside periods separately
    df['upside_vol'] = np.where(upside_mask, df['true_range'], 0)
    df['downside_vol'] = np.where(downside_mask, df['true_range'], 0)
    
    # 10-day rolling average volatility for each regime
    df['upside_vol_ma'] = df['upside_vol'].rolling(window=10, min_periods=5).mean()
    df['downside_vol_ma'] = df['downside_vol'].rolling(window=10, min_periods=5).mean()
    
    # Volatility asymmetry ratio
    df['vol_asymmetry'] = df['upside_vol_ma'] / (df['downside_vol_ma'] + 1e-8)
    
    # Breakout Detection
    # 20-day high/low
    df['20d_high'] = df['high'].rolling(window=20, min_periods=10).max()
    df['20d_low'] = df['low'].rolling(window=20, min_periods=10).min()
    
    # Breakout signals
    high_breakout = (df['close'] > df['20d_high'].shift(1)).astype(int)
    low_breakout = (df['close'] < df['20d_low'].shift(1)).astype(int)
    
    # Breakout magnitude
    df['high_breakout_mag'] = np.where(
        high_breakout == 1,
        (df['close'] - df['20d_high'].shift(1)) / df['20d_high'].shift(1),
        0
    )
    df['low_breakout_mag'] = np.where(
        low_breakout == 1,
        (df['close'] - df['20d_low'].shift(1)) / df['20d_low'].shift(1),
        0
    )
    
    # Net breakout magnitude
    df['net_breakout'] = df['high_breakout_mag'] + df['low_breakout_mag']
    
    # Volume Confirmation
    # 5-day volume trend (slope of linear regression)
    def volume_trend(series):
        if len(series) < 3:
            return 0
        x = np.arange(len(series))
        slope = np.polyfit(x, series.values, 1)[0]
        return slope / (series.mean() + 1e-8)
    
    df['volume_trend'] = df['volume'].rolling(window=5, min_periods=3).apply(
        volume_trend, raw=False
    )
    
    # Combine components into final factor
    # Higher volatility asymmetry + positive breakouts + rising volume = bullish signal
    df['factor'] = (
        df['vol_asymmetry'] * 
        df['net_breakout'] * 
        df['volume_trend']
    )
    
    # Normalize the factor
    df['factor_normalized'] = (
        df['factor'] - df['factor'].rolling(window=20, min_periods=10).mean()
    ) / (df['factor'].rolling(window=20, min_periods=10).std() + 1e-8)
    
    return df['factor_normalized']
