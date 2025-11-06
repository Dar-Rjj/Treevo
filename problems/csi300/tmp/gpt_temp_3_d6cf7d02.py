import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price-volume divergence, range efficiency,
    volume-confirmed reversal, amount flow, and volatility-adaptive volume signals.
    """
    # Price-Volume Divergence Momentum
    df = df.copy()
    
    # Multi-timeframe Momentum
    df['mom_5'] = df['close'] / df['close'].shift(5) - 1
    df['mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['mom_ratio'] = df['mom_5'] / df['mom_10'].replace(0, np.nan)
    
    # Volume Confirmation
    df['vol_trend'] = df['volume'] / df['volume'].shift(5)
    df['vol_mom'] = (df['volume'] / df['volume'].shift(5)) / (df['volume'].shift(5) / df['volume'].shift(10)).replace(0, np.nan)
    
    # Volume persistence (count of days with volume > previous day)
    vol_persistence = []
    for i in range(len(df)):
        if i < 5:
            vol_persistence.append(np.nan)
        else:
            window = df['volume'].iloc[i-5:i+1]
            count = (window > window.shift(1)).sum()
            vol_persistence.append(count)
    df['vol_persistence'] = vol_persistence
    
    # Divergence Signal
    df['divergence_strength'] = np.abs(df['mom_5']) / np.abs(df['vol_trend']).replace(0, np.nan)
    
    # Range Efficiency Momentum
    # Price Efficiency
    df['daily_eff'] = np.abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    
    # 3-day efficiency
    df['eff_3d'] = np.abs(df['close'] - df['close'].shift(3)) / (
        (df['high'] - df['low']).rolling(window=3).sum().replace(0, np.nan)
    )
    df['eff_trend'] = df['daily_eff'] / df['daily_eff'].shift(3)
    
    # Volatility Scaling
    df['range_5d'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['close'].shift(5)
    df['eff_vol'] = df['daily_eff'].rolling(window=5).std()
    
    # Volume-Confirmed Extreme Reversal
    df['mom_3'] = df['close'] / df['close'].shift(3) - 1
    df['mom_3_z'] = (df['mom_3'] - df['mom_3'].rolling(window=5).mean()) / df['mom_3'].rolling(window=5).std()
    df['vol_spike'] = df['volume'] / df['volume'].rolling(window=10).median()
    df['range_expansion'] = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=5).mean()
    
    # Amount Flow Regime Detection
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['directional_flow'] = df['amount'] * np.sign(df['price_change'])
    df['net_flow_3d'] = df['directional_flow'].rolling(window=3).sum()
    df['flow_mom'] = df['net_flow_3d'] / df['net_flow_3d'].shift(3).replace(0, np.nan)
    df['flow_vol'] = df['net_flow_3d'].rolling(window=5).std()
    
    # Flow persistence (count of consecutive same sign flows)
    flow_signs = np.sign(df['net_flow_3d'])
    flow_persistence = []
    for i in range(len(df)):
        if i < 5:
            flow_persistence.append(np.nan)
        else:
            window = flow_signs.iloc[i-5:i+1]
            if len(window) < 2:
                flow_persistence.append(0)
            else:
                count = 1
                for j in range(len(window)-1, 0, -1):
                    if window.iloc[j] == window.iloc[j-1] and window.iloc[j] != 0:
                        count += 1
                    else:
                        break
                flow_persistence.append(count)
    df['flow_persistence'] = flow_persistence
    
    # Volatility-Adaptive Volume Clustering
    df['range_vol'] = (df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()) / df['close'].shift(10)
    df['vol_ratio'] = df['close'].rolling(window=5).std() / df['close'].shift(5).rolling(window=5).std().replace(0, np.nan)
    df['vol_trend'] = df['range_vol'] / df['range_vol'].shift(5)
    
    df['vol_regime'] = df['volume'] / df['volume'].rolling(window=10).mean()
    
    # Volume cluster persistence
    vol_threshold = df['volume'].rolling(window=10).mean()
    vol_cluster = []
    for i in range(len(df)):
        if i < 5:
            vol_cluster.append(np.nan)
        else:
            window = df['volume'].iloc[i-5:i+1]
            threshold = vol_threshold.iloc[i]
            count = (window > threshold).sum()
            vol_cluster.append(count)
    df['vol_cluster'] = vol_cluster
    
    # Combined Alpha Factor
    # 1. Price-Volume Divergence Component
    divergence_component = (
        df['mom_5'] * (1 - df['vol_trend']) * df['divergence_strength'] * df['vol_persistence'] / 5
    )
    
    # 2. Range Efficiency Component
    efficiency_component = (
        df['mom_5'] * df['daily_eff'] * (1 / df['range_5d']) * np.sign(df['eff_trend'])
    )
    
    # 3. Volume-Confirmed Reversal Component
    reversal_component = (
        -df['mom_3_z'] * df['vol_spike'] * (1 / df['range_expansion']) * np.sign(df['net_flow_3d'])
    )
    
    # 4. Amount Flow Component
    flow_component = (
        df['net_flow_3d'] * df['flow_persistence'] * np.sign(df['flow_mom']) / df['flow_vol'].replace(0, np.nan)
    )
    
    # 5. Volatility-Adaptive Component
    volatility_component = (
        df['mom_5'] * df['vol_regime'] * df['vol_cluster'] / 5 * (1 / df['range_vol'])
    )
    
    # Final combined alpha factor
    alpha = (
        0.3 * divergence_component +
        0.25 * efficiency_component +
        0.2 * reversal_component +
        0.15 * flow_component +
        0.1 * volatility_component
    )
    
    return alpha
