import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Volatility Momentum Regime
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    
    # True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr_20d'] = df['true_range'].rolling(window=20).mean()
    
    # Volume z-score
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
    
    # Price position relative to MA
    df['ma_50d'] = df['close'].rolling(window=50).mean()
    df['price_position'] = (df['close'] - df['ma_50d']) / df['ma_50d']
    
    # Volume-Volatility Momentum Regime factor
    momentum_avg = (df['momentum_5d'] + df['momentum_10d']) / 2
    factor1 = momentum_avg / df['atr_20d'] * df['volume_zscore'] * df['price_position']
    
    # Range Breakout Efficiency
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['avg_range_20d'] = df['daily_range'].rolling(window=20).mean()
    df['range_ratio'] = df['daily_range'] / df['avg_range_20d']
    
    df['intraday_return'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['breakout_efficiency'] = df['intraday_return'] * (df['range_ratio'] > 1)
    
    df['volume_percentile'] = df['volume'].rolling(window=20).apply(lambda x: (x[-1] > x).mean())
    factor2 = df['breakout_efficiency'] * df['volume_percentile'].rolling(window=5).sum()
    
    # Liquidity-Driven Reversal
    df['volume_median_20d'] = df['volume'].rolling(window=20).median()
    df['volume_ratio'] = df['volume'] / df['volume_median_20d']
    
    df['return_1d'] = df['close'].pct_change(1)
    
    df['amount_percentile'] = df['amount'].rolling(window=20).apply(lambda x: (x[-1] > x).mean())
    factor3 = -df['return_1d'] * df['volume_ratio'] * df['amount_percentile']
    
    # Gap Acceleration Reversion
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_std_20d'] = df['overnight_gap'].rolling(window=20).std()
    df['gap_zscore'] = df['overnight_gap'] / df['gap_std_20d']
    
    df['volume_ma_5d'] = df['volume'].rolling(window=5).mean()
    df['volume_acceleration'] = df['volume_ma_5d'].diff(5)
    
    df['gap_percentile'] = df['overnight_gap'].rolling(window=20).apply(lambda x: (x[-1] > x).mean())
    factor4 = -df['overnight_gap'] * df['volume_acceleration'] * df['gap_percentile']
    
    # Pressure-Volume Trend
    df['williams_r'] = (df['high'].rolling(window=14).max() - df['close']) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) * -100
    df['market_pressure'] = 100 - df['williams_r']
    
    df['volume_ratio_5_20'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
    
    df['high_10d'] = df['high'].rolling(window=10).max()
    df['low_10d'] = df['low'].rolling(window=10).min()
    df['channel_position'] = (df['close'] - df['low_10d']) / (df['high_10d'] - df['low_10d']).replace(0, np.nan)
    
    factor5 = df['market_pressure'] * df['volume_ratio_5_20'] * df['channel_position']
    
    # Combine factors with equal weights
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + factor4.fillna(0) + factor5.fillna(0)) / 5
    
    return combined_factor
