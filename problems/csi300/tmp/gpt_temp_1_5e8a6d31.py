import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Mid-Price
    df['mid_price'] = (df['high'] + df['low']) / 2
    
    # Calculate Mid-Price Returns
    df['mid_price_ret_5'] = (df['mid_price'] - df['mid_price'].shift(5)) / df['mid_price'].shift(5)
    df['mid_price_ret_10'] = (df['mid_price'] - df['mid_price'].shift(10)) / df['mid_price'].shift(10)
    
    # Calculate Directional Volume Flow
    df['directional_movement'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['directional_volume_flow'] = df['directional_movement'] * df['volume']
    
    # Calculate Synchronization Correlations
    df['sync_corr_5'] = df['mid_price_ret_5'].rolling(window=5).corr(df['directional_volume_flow'].rolling(window=5).mean())
    df['sync_corr_10'] = df['mid_price_ret_10'].rolling(window=10).corr(df['directional_volume_flow'].rolling(window=10).mean())
    
    # Calculate Net Synchronization
    df['net_sync'] = np.sign(df['sync_corr_5']) + np.sign(df['sync_corr_10'])
    
    # Calculate Volatility Asymmetry
    returns = df['close'].pct_change()
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    df['upside_vol'] = positive_returns.rolling(window=30, min_periods=1).std()
    df['downside_vol'] = negative_returns.rolling(window=30, min_periods=1).std()
    df['vol_asymmetry'] = (df['upside_vol'] / (df['downside_vol'] + 1e-8)) - 1
    
    # Regime Classification
    conditions = [
        df['vol_asymmetry'] > 0.2,
        df['vol_asymmetry'] < -0.2
    ]
    choices = [1.5, 0.7]  # Bull, Bear weights
    df['regime_weight'] = np.select(conditions, choices, default=1.0)  # Neutral weight = 1.0
    
    # Calculate Efficiency Score
    df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Calculate Breakout Efficiency
    df['prev_high_max'] = df['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    df['breakout_efficiency'] = np.maximum(0, df['high'] - df['prev_high_max']) / (df['high'] - df['low'] + 1e-8)
    
    # Calculate Efficiency Score
    df['efficiency_score'] = np.cbrt(df['price_efficiency'] * df['breakout_efficiency'])
    
    # Calculate Momentum Strength
    df['momentum_strength'] = np.cbrt((np.abs(df['mid_price_ret_5']) + np.abs(df['mid_price_ret_10'])) / 2)
    
    # Calculate Regime-Weighted Synchronization
    df['regime_weighted_sync'] = df['net_sync'] * df['regime_weight']
    
    # Final Alpha
    alpha = np.tanh(df['regime_weighted_sync'] * df['efficiency_score'] * df['momentum_strength'])
    
    return alpha
