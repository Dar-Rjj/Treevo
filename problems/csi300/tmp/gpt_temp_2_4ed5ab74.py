import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Reversal Signal
    df['5_day_return'] = df['close'].pct_change(5)
    df['20_day_low'] = df['low'].rolling(window=20, min_periods=1).min()
    df['yesterday_low'] = df['low'].shift(1)
    
    price_reversal = (
        (df['5_day_return'] < 0).astype(int) +  # Recent decline
        (df['close'] < df['20_day_low']).astype(int) +  # Oversold
        (df['close'] > df['yesterday_low']).astype(int)  # Bounce detection
    )
    
    # Volume Pattern Analysis
    df['20_day_avg_volume'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_spike'] = (df['volume'] > df['20_day_avg_volume']).astype(int)
    
    # Volume trend (5-day slope)
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0
    )
    df['volume_trend'] = np.where(volume_trend > 0, 1, 0)
    
    # Volume-Price Divergence
    price_down = df['close'] < df['close'].shift(1)
    volume_decreasing = df['volume'] < df['volume'].shift(1)
    df['volume_price_divergence'] = (price_down & volume_decreasing).astype(int)
    
    volume_pattern = df['volume_spike'] + df['volume_trend'] + df['volume_price_divergence']
    
    # Volatility Regime
    df['20_day_range'] = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close'].rolling(window=20).mean()
    df['60_day_range'] = (df['high'].rolling(window=60).max() - df['low'].rolling(window=60).min()) / df['close'].rolling(window=60).mean()
    df['5_day_range'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['close'].rolling(window=5).mean()
    
    df['high_vol_period'] = (df['20_day_range'] > df['60_day_range']).astype(int)
    df['vol_compression'] = (df['5_day_range'] < df['20_day_range']).astype(int)
    
    df['yesterday_range'] = (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    df['today_range'] = (df['high'] - df['low']) / df['close']
    df['vol_breakout'] = (df['today_range'] > 2 * df['yesterday_range']).astype(int)
    
    volatility_regime = df['high_vol_period'] + df['vol_compression'] + df['vol_breakout']
    
    # Combine all components
    factor = price_reversal + volume_pattern + volatility_regime
    
    return factor
