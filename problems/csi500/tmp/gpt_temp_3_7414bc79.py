import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple heuristics including:
    - High-Low Range Breakout with Volume Confirmation
    - Price Reversal After Extreme Volume
    - Intraday Momentum Persistence
    - Volume-Weighted Price Acceleration
    - Overnight Gap Mean Reversion
    """
    
    # High-Low Range Breakout with Volume Confirmation
    df['range_today'] = df['high'] - df['low']
    df['range_yesterday'] = df['range_today'].shift(1)
    df['range_expansion'] = df['range_today'] / df['range_yesterday'] - 1
    
    # Volume confirmation
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_signal'] = df['volume'] / df['volume_ma_5']
    
    # Combine range expansion with volume signal
    breakout_factor = df['range_expansion'] * df['volume_signal']
    
    # Price Reversal After Extreme Volume
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
    df['extreme_volume'] = (df['volume_zscore'] > 2).astype(int)
    
    # Calculate future 3-day returns for reversal signal
    df['future_3d_return'] = df['close'].shift(-3) / df['close'] - 1
    
    # Negative relationship: high volume → negative future returns
    reversal_factor = -df['extreme_volume'] * df['volume_zscore'].clip(lower=0)
    
    # Intraday Momentum Persistence
    # Note: Since we don't have intraday data, we'll approximate with OHLC
    df['morning_strength'] = (df['high'] - df['open']) / df['open']
    df['afternoon_persistence'] = (df['close'] - df['low']) / df['low']
    
    momentum_factor = df['morning_strength'] * df['afternoon_persistence']
    
    # Volume-Weighted Price Acceleration
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['price_acceleration'] = df['return_5d'] - df['return_10d']
    
    # Volume validation using percentiles
    df['volume_percentile'] = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(int)
    )
    
    acceleration_factor = df['price_acceleration'] * df['volume_percentile']
    
    # Overnight Gap Mean Reversion
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Mean reversion: negative relationship between gap and intraday return
    gap_factor = -df['overnight_gap'] * df['intraday_return']
    
    # Combine all factors with equal weights
    combined_factor = (
        breakout_factor.fillna(0) * 0.2 +
        reversal_factor.fillna(0) * 0.2 +
        momentum_factor.fillna(0) * 0.2 +
        acceleration_factor.fillna(0) * 0.2 +
        gap_factor.fillna(0) * 0.2
    )
    
    # Normalize the final factor
    final_factor = (combined_factor - combined_factor.rolling(window=20).mean()) / combined_factor.rolling(window=20).std()
    
    return final_factor
