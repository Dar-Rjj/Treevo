import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor combining multiple technical indicators across different timeframes.
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Divergence
    df['mom_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['mom_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['vol_accel_3d'] = (df['volume'] - df['volume'].shift(3)) / df['volume'].shift(3)
    df['vol_accel_5d'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    df['momentum_div'] = (df['mom_3d'] * df['vol_accel_3d']) + (df['mom_5d'] * df['vol_accel_5d'])
    
    # Recent Price-Volume Interaction
    df['cum_price_change_3d'] = df['close'] - df['close'].shift(3)
    
    # Calculate cumulative volume divergence
    vol_div = []
    for i in range(len(df)):
        if i < 2:
            vol_div.append(np.nan)
            continue
        cum_sum = 0
        for j in range(i-2, i+1):
            if j > 0:
                price_change = df['close'].iloc[j] - df['close'].iloc[j-1]
                cum_sum += df['volume'].iloc[j] * price_change
        vol_div.append(cum_sum)
    
    df['cum_vol_div_3d'] = vol_div
    df['price_vol_interaction'] = df['cum_price_change_3d'] * df['cum_vol_div_3d']
    
    # Volatility-Adjusted Breakout
    df['prev_4d_high'] = df['high'].rolling(window=4, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    df['breakout_strength'] = (df['close'] - df['prev_4d_high']) / df['prev_4d_high']
    df['recent_volatility'] = (df['high'] - df['low']) / df['close']
    df['volatility_breakout'] = df['breakout_strength'] / df['recent_volatility']
    
    # Gap Momentum with Volume Confirmation
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['intraday_momentum'] = (df['close'] - df['open']) / df['open']
    df['volume_intensity'] = df['volume'] / df['volume'].shift(1)
    df['gap_momentum'] = df['opening_gap'] * df['intraday_momentum'] * df['volume_intensity']
    
    # Amount-Based Price Efficiency
    df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['amount_per_move'] = df['amount'] / abs(df['close'] - df['close'].shift(1))
    df['efficiency_signal'] = df['price_efficiency'] * df['amount_per_move']
    
    # Combine all signals with equal weights
    factors = ['momentum_div', 'price_vol_interaction', 'volatility_breakout', 
               'gap_momentum', 'efficiency_signal']
    
    # Standardize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        standardized = (df[factor] - df[factor].mean()) / df[factor].std()
        combined_factor += standardized
    
    return combined_factor
