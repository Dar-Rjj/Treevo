import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators:
    - High-Low Range Persistence Momentum
    - Volume-Adjusted Price Reversal Strength
    - Opening Gap Fill Probability
    - Intraday Volatility Compression Breakout
    - Relative Volume-Weighted Momentum Divergence
    """
    
    # High-Low Range Persistence Momentum
    df['high_low_range'] = df['high'] - df['low']
    df['range_change'] = df['high_low_range'] - df['high_low_range'].shift(1)
    
    # Track consecutive same-direction range changes
    df['range_direction'] = np.sign(df['range_change'])
    df['range_persistence'] = 0
    for i in range(1, len(df)):
        if df['range_direction'].iloc[i] == df['range_direction'].iloc[i-1]:
            df.loc[df.index[i], 'range_persistence'] = df['range_persistence'].iloc[i-1] + 1
        else:
            df.loc[df.index[i], 'range_persistence'] = 1
    
    # Price momentum (5-day return)
    df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
    range_momentum = df['range_persistence'] * df['price_momentum']
    
    # Volume-Adjusted Price Reversal Strength
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['price_reversal'] = ((df['price_change'] * df['price_change'].shift(1)) < 0).astype(int)
    
    # Volume ratio to 20-day average
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
    
    # Reversal strength score
    reversal_strength = df['price_reversal'] * abs(df['price_change']) * df['volume_ratio']
    
    # Opening Gap Fill Probability
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Historical gap fill analysis (using rolling window)
    df['gap_fill_signal'] = 0
    for i in range(1, len(df)):
        if abs(df['opening_gap'].iloc[i]) > 0.005:  # 0.5% threshold
            lookback = min(10, i)
            historical_data = df['opening_gap'].iloc[max(0, i-lookback):i]
            if len(historical_data) > 0:
                fill_rate = (abs(historical_data[historical_data.abs() > 0.005]) < 
                           abs(historical_data[historical_data.abs() > 0.005]).shift(1)).mean()
                df.loc[df.index[i], 'gap_fill_signal'] = fill_rate if not pd.isna(fill_rate) else 0
    
    gap_fill_prob = df['gap_fill_signal'] * df['volume_ratio']
    
    # Intraday Volatility Compression Breakout
    df['daily_range'] = df['high'] - df['low']
    df['range_std_10'] = df['daily_range'].rolling(window=10, min_periods=1).std()
    
    # Volatility compression indicator
    df['vol_compression'] = df['daily_range'] / (df['range_std_10'] + 1e-8)
    
    # Breakout detection
    df['range_expansion'] = (df['daily_range'] > df['daily_range'].rolling(window=5, min_periods=1).mean()).astype(int)
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(window=20, min_periods=1).mean() * 1.2).astype(int)
    
    breakout_quality = df['range_expansion'] * df['volume_surge'] * df['vol_compression']
    
    # Relative Volume-Weighted Momentum Divergence
    # Short-term momentum (3-day)
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    # Medium-term momentum (10-day)
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Volume-weighted momentum
    df['vw_momentum_3d'] = df['momentum_3d'] * df['volume_ratio']
    df['vw_momentum_10d'] = df['momentum_10d'] * df['volume_ratio']
    
    # Divergence detection
    momentum_divergence = (df['momentum_3d'] - df['vw_momentum_3d']) + (df['momentum_10d'] - df['vw_momentum_10d'])
    
    # Combine all factors with weights
    factor = (
        0.3 * range_momentum +
        0.25 * reversal_strength +
        0.2 * gap_fill_prob +
        0.15 * breakout_quality +
        0.1 * momentum_divergence
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20, min_periods=1).mean()) / (factor.rolling(window=20, min_periods=1).std() + 1e-8)
    
    return factor
