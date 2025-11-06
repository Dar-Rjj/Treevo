import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price change and signed volume
    df = df.copy()
    df['price_change'] = df['close'] - df['open']
    df['signed_volume'] = np.where(df['price_change'] > 0, df['volume'], 
                                  np.where(df['price_change'] < 0, -df['volume'], 0))
    
    # Order flow autocorrelation imbalance
    df['buy_flow'] = np.where(df['signed_volume'] > 0, df['signed_volume'], 0)
    df['sell_flow'] = np.where(df['signed_volume'] < 0, -df['signed_volume'], 0)
    
    # Calculate autocorrelations for buy and sell flows
    df['buy_flow_autocorr'] = df['buy_flow'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 3 else np.nan, raw=False
    )
    df['sell_flow_autocorr'] = df['sell_flow'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 3 else np.nan, raw=False
    )
    
    # Flow persistence difference
    df['flow_persistence_diff'] = df['buy_flow_autocorr'] - df['sell_flow_autocorr']
    
    # Asymmetric price impact dynamics
    df['high_low_range'] = df['high'] - df['low']
    df['price_impact_buy'] = np.where(df['signed_volume'] > 0, 
                                     df['price_change'] / (df['buy_flow'] + 1e-10), 0)
    df['price_impact_sell'] = np.where(df['signed_volume'] < 0, 
                                      df['price_change'] / (df['sell_flow'] + 1e-10), 0)
    
    # Rolling impact asymmetry
    df['impact_asymmetry'] = (
        df['price_impact_buy'].rolling(window=5, min_periods=3).mean() - 
        df['price_impact_sell'].rolling(window=5, min_periods=3).mean()
    )
    
    # Liquidity restoration asymmetry using volume patterns
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    df['restoration_speed'] = df['high_low_range'].rolling(window=3).std() / (df['volume_ratio'] + 1e-10)
    
    # Calculate restoration asymmetry (lower restoration speed suggests better liquidity recovery)
    df['restoration_asymmetry'] = -df['restoration_speed'].rolling(window=5, min_periods=3).mean()
    
    # Generate composite alpha signal
    # Normalize components
    df['flow_persistence_norm'] = (df['flow_persistence_diff'] - 
                                  df['flow_persistence_diff'].rolling(window=20).mean()) / \
                                 (df['flow_persistence_diff'].rolling(window=20).std() + 1e-10)
    
    df['impact_asymmetry_norm'] = (df['impact_asymmetry'] - 
                                  df['impact_asymmetry'].rolling(window=20).mean()) / \
                                 (df['impact_asymmetry'].rolling(window=20).std() + 1e-10)
    
    df['restoration_asymmetry_norm'] = (df['restoration_asymmetry'] - 
                                       df['restoration_asymmetry'].rolling(window=20).mean()) / \
                                      (df['restoration_asymmetry'].rolling(window=20).std() + 1e-10)
    
    # Combine components with weights
    alpha_signal = (
        0.4 * df['flow_persistence_norm'] +
        0.35 * df['impact_asymmetry_norm'] + 
        0.25 * df['restoration_asymmetry_norm']
    )
    
    return alpha_signal
