import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Price-Volume Divergence Momentum
    # Calculate Price Momentum
    mom_5 = df['close'] / df['close'].shift(5) - 1
    mom_10 = df['close'] / df['close'].shift(10) - 1
    
    # Calculate Volume Divergence
    vol_trend_5 = df['volume'] / df['volume'].shift(5) - 1
    vol_trend_10 = df['volume'] / df['volume'].shift(10) - 1
    
    # Compute divergence ratio
    div_short = mom_5 / (vol_trend_5 + 1e-8)
    div_long = mom_10 / (vol_trend_10 + 1e-8)
    
    factor1 = div_short * div_long
    
    # High-Low Range Efficiency
    # Calculate Daily Range
    daily_range = df['high'] - df['low']
    
    # Calculate Price Movement
    abs_return = abs(df['close'] / df['close'].shift(1) - 1)
    
    # Compute Efficiency Ratio
    efficiency = abs_return / (daily_range / df['close'].shift(1) + 1e-8)
    eff_5 = efficiency.rolling(5).mean()
    eff_10 = efficiency.rolling(10).mean()
    
    factor2 = eff_5 * eff_10
    
    # Volume-Weighted Acceleration
    # Calculate Price Acceleration
    mom_3 = df['close'] / df['close'].shift(3) - 1
    mom_6 = df['close'] / df['close'].shift(6) - 1
    acceleration_raw = mom_6 - mom_3
    acceleration = acceleration_raw.rolling(5).mean()
    
    # Weight by Volume Profile
    vol_perc_10 = df['volume'].rolling(10).apply(lambda x: np.percentile(x, 50))
    vol_perc_20 = df['volume'].rolling(20).apply(lambda x: np.percentile(x, 50))
    vol_strength_10 = df['volume'] / (vol_perc_10 + 1e-8)
    vol_strength_20 = df['volume'] / (vol_perc_20 + 1e-8)
    vol_strength = vol_strength_10 * vol_strength_20
    
    factor3 = acceleration * vol_strength * np.sign(mom_3)
    
    # Open-Close Relative Strength
    # Calculate Intraday Strength
    intraday_strength = (df['close'] - df['open']) / (df['open'] + 1e-8)
    intraday_avg = intraday_strength.rolling(5).mean()
    
    # Compare with Market Regime
    daily_vol = (df['high'] - df['low']) / (df['open'] + 1e-8)
    vol_adjusted_strength = intraday_strength / (daily_vol + 1e-8)
    vol_mom = vol_adjusted_strength / vol_adjusted_strength.shift(3) - 1
    
    factor4 = intraday_avg * vol_mom
    
    # Amount-Based Return Persistence
    # Calculate Return Stream
    ret_1d = df['close'] / df['close'].shift(1) - 1
    ret_3d = df['close'] / df['close'].shift(3) - 1
    
    # Weight by Transaction Size
    avg_tx_size = df['amount'] / (df['volume'] + 1e-8)
    
    def rolling_corr(x):
        if len(x) < 5:
            return np.nan
        returns = x[:, 0]
        sizes = x[:, 1]
        if np.std(returns) == 0 or np.std(sizes) == 0:
            return 0
        return np.corrcoef(returns, sizes)[0, 1]
    
    combined_data = np.column_stack([ret_1d, avg_tx_size])
    size_corr = pd.Series([rolling_corr(combined_data[i-4:i+1]) if i >= 4 else np.nan 
                          for i in range(len(combined_data))], index=df.index)
    
    factor5 = (abs(ret_1d) + abs(ret_3d)) * size_corr * np.sign(ret_1d)
    
    # Combine all factors with equal weights
    final_factor = (factor1 + factor2 + factor3 + factor4 + factor5) / 5
    
    return final_factor
