import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum with Volatility Adjustment
    intraday_momentum = (df['high'] - df['low']) / (df['high'] + df['low'])
    returns = df['close'].pct_change()
    vol_20d = returns.rolling(window=20).std()
    factor1 = intraday_momentum / vol_20d.replace(0, np.nan)
    
    # Volume-Scaled Price Reversal
    price_reversal = df['close'].pct_change()
    volume_change = df['volume'].pct_change()
    factor2 = price_reversal * volume_change
    
    # Amplitude-Weighted Trend Strength
    price_amplitude = (df['high'] - df['low']) / df['close']
    ma_10 = df['close'].rolling(window=10).mean()
    trend_strength = df['close'] - ma_10
    factor3 = price_amplitude * trend_strength
    
    # Volume-Price Divergence Factor
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_slope = volume_ma_5.diff()
    price_ma_5 = df['close'].rolling(window=5).mean()
    price_slope = price_ma_5.diff()
    factor4 = volume_slope * price_slope
    
    # Dynamic Support Resistance Breakout
    support_20d = df['low'].rolling(window=20).min()
    breakout_strength = (df['close'] - support_20d) / support_20d.replace(0, np.nan)
    factor5 = breakout_strength * df['volume']
    
    # Momentum Acceleration with Volume Confirmation
    returns_2d = df['close'].pct_change(periods=2)
    returns_1d = df['close'].pct_change()
    momentum_accel = returns_2d - returns_1d
    volume_roc = df['volume'].pct_change()
    factor6 = momentum_accel * volume_roc
    
    # Volatility-Regime Adjusted Mean Reversion
    hl2 = (df['high'] + df['low']) / 2
    mean_reversion = (df['close'] - hl2) / (df['high'] - df['low']).replace(0, np.nan)
    price_range = (df['high'] - df['low']).rolling(window=10).mean()
    factor7 = mean_reversion * price_range
    
    # Volume-Weighted Price Efficiency
    price_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    volume_autocorr = df['volume'].rolling(window=5).apply(lambda x: x.autocorr(), raw=False)
    factor8 = price_efficiency * volume_autocorr
    
    # Combine factors with equal weights
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
                      factor4.fillna(0) + factor5.fillna(0) + factor6.fillna(0) + 
                      factor7.fillna(0) + factor8.fillna(0)) / 8
    
    return combined_factor
