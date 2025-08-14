import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Daily Price Movement
    daily_price_movement = df['close'] - df['open']
    
    # Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Volume-Weighted Price Movement
    volume_weighted_price_movement = (daily_price_movement * df['volume']) / intraday_volatility
    
    # Close Price Momentum (N=10)
    close_price_momentum = df['close'].diff(10)
    
    # Directional Days
    up_days = (df['close'] > df['open']).astype(int)
    down_days = (df['open'] > df['close']).astype(int)
    directional_day_count = (up_days - down_days).rolling(window=10).sum()
    
    # Volume Weighted Directional Counts
    volume_weighted_directional_counts = directional_day_count * df['volume']
    
    # Combined Momentum and Volume-Weighted Directional
    combined_momentum_volume = close_price_momentum + volume_weighted_directional_counts
    
    # Adjusted Daily Price Movement
    adjusted_daily_price_movement = combined_momentum_volume + volume_weighted_price_movement
    
    # Volume-Weighted Average Prices
    vwap = (df[['high', 'low', 'close']].median(axis=1) * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    
    # Current Day's Volume-Weighted Price
    current_day_vw_price = (df[['high', 'low', 'close']].median(axis=1) * df['volume'])
    
    # VWPTI
    vwpti = (current_day_vw_price - vwap) / vwap
    
    # High-Low Range Momentum
    high_low_range_momentum = (df['high'] - df['low']).diff(10)
    
    # Simple Moving Averages (SMA)
    sma_5 = df['close'].rolling(window=5).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    
    # Exponential Moving Averages (EMA)
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Price Differentials
    daily_price_change = df['close'].diff()
    high_low_spread = df['high'] - df['low']
    open_close_spread = df['open'] - df['close']
    
    # Volume-related Factors
    volume_differentials = df['volume'].diff()
    volume_price_corr = df['volume'].rolling(window=10).corr(df['close'])
    
    # Momentum Indicators
    roc_5 = df['close'].pct_change(5)
    roc_20 = df['close'].pct_change(20)
    rsi_14 = 100 - (100 / (1 + df['close'].diff().rolling(window=14).apply(lambda x: x[x > 0].sum()) / df['close'].diff().rolling(window=14).apply(lambda x: -x[x < 0].sum())))
    
    # Volatility
    true_range = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    atr_14 = true_range.rolling(window=14).mean()
    
    # Money Flow
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_money_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_money_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    mfi_14 = 100 - (100 / (1 + positive_money_flow.rolling(window=14).sum() / negative_money_flow.rolling(window=14).sum()))
    
    # Final Alpha Factor
    alpha_factor = (
        adjusted_daily_price_movement +
        vwpti +
        high_low_range_momentum +
        (sma_5 - sma_20) +
        (ema_5 - ema_20) +
        rsi_14 +
        atr_14 +
        mfi_14
    )
    
    return alpha_factor
