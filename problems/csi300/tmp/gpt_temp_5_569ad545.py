import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration with Volume Confirmation
    # Calculate Price Momentum Acceleration
    momentum_3d = data['close'] / data['close'].shift(3) - 1
    momentum_3d_prev = data['close'].shift(1) / data['close'].shift(4) - 1
    momentum_acceleration = momentum_3d - momentum_3d_prev
    
    # Apply Volume Confirmation
    vol_5d_avg = data['volume'].shift(1).rolling(window=5).mean()
    volume_momentum = data['volume'] / vol_5d_avg
    factor1 = momentum_acceleration * volume_momentum
    
    # Volatility-Adjusted Return Momentum
    # Calculate Volatility-Regime Returns
    tr = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': abs(data['high'] - data['close'].shift(1)),
        'lc': abs(data['low'] - data['close'].shift(1))
    }).max(axis=1)
    vol_5d = tr.rolling(window=5).mean()
    adj_returns = (data['close'] / data['close'].shift(1) - 1) / vol_5d
    
    # Compute Momentum of Adjusted Returns
    adj_momentum_3d = adj_returns.rolling(window=3).sum()
    adj_momentum_3d_prev = adj_returns.shift(3).rolling(window=3).sum()
    factor2 = adj_momentum_3d - adj_momentum_3d_prev
    
    # Opening Gap Persistence Factor
    # Calculate Gap Strength
    gap_pct = (data['open'] / data['close'].shift(1) - 1)
    gap_vol_10d = gap_pct.shift(1).rolling(window=10).std()
    gap_strength = gap_pct / gap_vol_10d
    
    # Assess Gap Persistence Pattern
    gap_signs = gap_pct.rolling(window=3).apply(
        lambda x: sum(np.sign(x.iloc[-1]) == np.sign(x.iloc[:-1])) if len(x) == 3 else np.nan
    )
    factor3 = gap_strength * gap_signs
    
    # Intraday Momentum Persistence
    # Measure Intraday Strength
    intraday_return = data['close'] / data['open'] - 1
    intraday_3d_avg = intraday_return.shift(1).rolling(window=3).mean()
    intraday_strength = intraday_return / intraday_3d_avg
    
    # Evaluate Persistence Signal
    intraday_signs = intraday_return.rolling(window=3).apply(
        lambda x: sum(np.sign(x.iloc[-1]) == np.sign(x.iloc[:-1])) if len(x) == 3 else np.nan
    )
    factor4 = intraday_strength * intraday_signs
    
    # Range-Expansion Trend Factor
    # Assess Price Range Behavior
    range_pct = (data['high'] - data['low']) / data['low']
    range_5d_avg = range_pct.shift(1).rolling(window=5).mean()
    range_expansion = range_pct / range_5d_avg
    
    # Combine with Price Trend
    sma_5d = data['close'].rolling(window=5).mean()
    trend_direction = np.where(data['close'] > sma_5d, 1, 
                              np.where(data['close'] < sma_5d, -1, 0))
    factor5 = range_expansion * trend_direction
    
    # Volume-Price Convergence Detector
    # Calculate Independent Momentum
    price_momentum = data['close'] / data['close'].shift(5) - 1
    volume_momentum_2 = data['volume'] / data['volume'].shift(1).rolling(window=5).mean()
    
    # Detect Convergence Pattern
    alignment_multiplier = np.where(
        np.sign(price_momentum) == np.sign(volume_momentum_2), 1, -1
    )
    factor6 = price_momentum * volume_momentum_2 * alignment_multiplier
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'f1': factor1,
        'f2': factor2,
        'f3': factor3,
        'f4': factor4,
        'f5': factor5,
        'f6': factor6
    })
    
    # Z-score normalize each factor and take simple average
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
