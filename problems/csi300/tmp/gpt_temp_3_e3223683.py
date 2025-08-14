import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Raw Momentum
    n = 10
    raw_momentum = df['close'].pct_change(periods=n)
    
    # Weight by Cumulative Volume
    cumulative_volume = df['volume'].rolling(window=n).sum()
    volume_weighted_momentum = raw_momentum / cumulative_volume
    
    # Simple Moving Average Cross Over
    sma_5 = df['close'].rolling(window=5).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    sma_cross_over = (sma_5 - sma_20) / sma_20
    
    # Exponential Weighted Moving Average Growth
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_30 = df['close'].ewm(span=30, adjust=False).mean()
    ema_growth = (ema_10 - ema_30) / ema_30
    
    # Incorporate ADMI Factor
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    
    atr = df['tr'].rolling(window=14).mean()
    
    df['+dm'] = np.where((df['high'] > df['high'].shift(1)) & 
                         (df['high'] - df['high'].shift(1) > df['low'].shift(1) - df['low']), 
                         df['high'] - df['high'].shift(1), 0)
    df['-dm'] = np.where((df['low'] < df['low'].shift(1)) & 
                         (df['low'].shift(1) - df['low'] > df['high'] - df['high'].shift(1)), 
                         df['low'].shift(1) - df['low'], 0)
    
    +di = 100 * (df['+dm'].rolling(window=14).sum() / atr)
    -di = 100 * (df['-dm'].rolling(window=14).sum() / atr)
    
    admi = (+di - -di) / (+di + -di)
    
    adjusted_momentum = volume_weighted_momentum * admi + volume_weighted_momentum
    
    # Measure Enhanced Volatility
    volatility_atr = df['tr'].rolling(window=20).mean()
    volume_weighted_price_change = (df['close'].diff().abs() * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    volume_change_ratio = df['volume'] / df['volume'].shift(1)
    
    # Combine All Factors
    composite_alpha_factor = (
        adjusted_momentum +
        sma_cross_over +
        ema_growth +
        (volume_weighted_price_change / volatility_atr) +
        volume_change_ratio
    )
    
    return composite_alpha_factor
