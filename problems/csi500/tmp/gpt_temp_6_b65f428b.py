import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate multi-timeframe volume-confirmed momentum factors
    """
    df = data.copy()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price returns
    df['ret_1d'] = df['close'] / df['close'].shift(1) - 1
    df['ret_3d'] = df['close'] / df['close'].shift(3) - 1
    df['ret_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Calculate volume changes
    df['vol_chg_1d'] = df['volume'] / df['volume'].shift(1) - 1
    df['vol_chg_3d'] = df['volume'] / df['volume'].shift(3) - 1
    df['vol_chg_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['vol_chg_10d'] = df['volume'] / df['volume'].shift(10) - 1
    
    # Factor 1: Accelerating Momentum with Volume Support
    momentum_acceleration = df['ret_10d'] - df['ret_3d']
    direction_alignment = np.sign(df['ret_3d']) * np.sign(df['vol_chg_5d'])
    factor1 = momentum_acceleration * direction_alignment
    
    # Factor 2: Persistent Gap with Volume Amplification
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate gap persistence (3-day window)
    gap_persistence = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        if i >= 2:
            gaps = [df['opening_gap'].iloc[i-2], df['opening_gap'].iloc[i-1], df['opening_gap'].iloc[i]]
            if all(gap > 0 for gap in gaps) or all(gap < 0 for gap in gaps):
                gap_persistence.iloc[i] = 3
            elif (gaps[1] > 0 and gaps[2] > 0) or (gaps[1] < 0 and gaps[2] < 0):
                gap_persistence.iloc[i] = 2
            elif gaps[2] != 0:
                gap_persistence.iloc[i] = 1
    
    volume_trend_3d = df['vol_chg_3d']
    factor2 = df['opening_gap'] * gap_persistence * df['volume'] * volume_trend_3d
    
    # Factor 3: Range Efficiency with Volume Confirmation
    df['intraday_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Calculate range persistence (5-day average)
    range_persistence = df['intraday_range'].rolling(window=5).mean()
    
    df['volume_efficiency'] = df['volume'] / (df['high'] - df['low'])
    
    # Calculate volume efficiency persistence (3-day increasing count)
    vol_eff_persistence = pd.Series(0, index=df.index)
    for i in range(2, len(df)):
        if i >= 2:
            eff_today = df['volume_efficiency'].iloc[i]
            eff_yest = df['volume_efficiency'].iloc[i-1] if i-1 >= 0 else np.nan
            eff_2days = df['volume_efficiency'].iloc[i-2] if i-2 >= 0 else np.nan
            
            count = 0
            if not np.isnan(eff_2days) and eff_yest > eff_2days:
                count += 1
            if not np.isnan(eff_yest) and eff_today > eff_yest:
                count += 1
            if count > 0:  # At least one increase
                count += 1  # Base persistence
            vol_eff_persistence.iloc[i] = count
    
    factor3 = df['intraday_range'] * df['volume_efficiency'] * vol_eff_persistence
    
    # Factor 4: Volume-Confirmed Reversal Detection
    price_divergence = df['ret_10d'] - df['ret_3d']
    df['intraday_momentum'] = (df['close'] - df['open']) / df['open']
    gap_reversal = np.sign(df['opening_gap']) * np.sign(df['intraday_momentum'])
    
    volume_spike = df['vol_chg_10d']
    volume_price_divergence = np.sign(price_divergence) * np.sign(volume_spike)
    
    factor4 = price_divergence * volume_price_divergence * gap_reversal
    
    # Factor 5: Multi-Timeframe Volume-Weighted Momentum
    # Momentum consistency (count of positive momentum across timeframes)
    momentum_consistency = (
        (df['ret_1d'] > 0).astype(int) + 
        (df['ret_3d'] > 0).astype(int) + 
        (df['ret_10d'] > 0).astype(int)
    )
    
    # Volume persistence (5-day increasing count)
    vol_persistence = pd.Series(0, index=df.index)
    for i in range(4, len(df)):
        if i >= 4:
            vol_today = df['volume'].iloc[i]
            vol_1d = df['volume'].iloc[i-1] if i-1 >= 0 else np.nan
            vol_2d = df['volume'].iloc[i-2] if i-2 >= 0 else np.nan
            vol_3d = df['volume'].iloc[i-3] if i-3 >= 0 else np.nan
            vol_4d = df['volume'].iloc[i-4] if i-4 >= 0 else np.nan
            
            vols = [vol_4d, vol_3d, vol_2d, vol_1d, vol_today]
            valid_vols = [v for v in vols if not np.isnan(v)]
            
            if len(valid_vols) >= 2:
                increasing_count = sum(1 for j in range(1, len(valid_vols)) 
                                   if valid_vols[j] > valid_vols[j-1])
                vol_persistence.iloc[i] = increasing_count
    
    # Volume momentum profile average
    vol_momentum_profile = (df['vol_chg_1d'] + df['vol_chg_3d'] + df['vol_chg_10d']) / 3
    
    factor5 = momentum_consistency * vol_persistence * vol_momentum_profile
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5
    })
    
    # Z-score normalize each factor and take average
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    result = normalized_factors.mean(axis=1)
    
    return result
