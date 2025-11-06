import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining momentum persistence, regime-adaptive signals,
    multi-timeframe confirmation, and extreme move analysis.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns for momentum calculations
    df['ret_1'] = df['close'].pct_change()
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)
    df['ret_20'] = df['close'].pct_change(20)
    
    # Multi-Timeframe Momentum Alignment
    df['mom_short'] = df['close'] / df['close'].shift(5) - 1
    df['mom_medium'] = df['close'] / df['close'].shift(10) - 1
    df['mom_long'] = df['close'] / df['close'].shift(20) - 1
    
    # Momentum Strength Assessment
    for window in [5, 10, 20]:
        df[f'mom_consistency_{window}'] = df[f'ret_{window}'].rolling(5).apply(
            lambda x: (x > 0).sum(), raw=True
        )
    
    df['mom_acceleration'] = (df['close'] / df['close'].shift(5)) / (df['close'].shift(5) / df['close'].shift(10))
    
    df['cross_timeframe_alignment'] = (
        (np.sign(df['mom_short']) == np.sign(df['mom_medium'])) & 
        (np.sign(df['mom_medium']) == np.sign(df['mom_long']))
    ).astype(int)
    
    # Volume-Confirmed Momentum
    df['volume_trend'] = df['volume'] / df['volume'].shift(5)
    df['volume_momentum_alignment'] = (np.sign(df['volume_trend']) == np.sign(df['mom_short'])).astype(int)
    
    df['volume_persistence'] = df['volume'].rolling(5).apply(
        lambda x: (x > pd.Series(x).shift(1)).sum(), raw=True
    )
    
    df['momentum_quality'] = df['mom_short'] * df['volume_trend']
    
    # Volatility Regime Classification
    df['vol_10d'] = df['close'].rolling(10).std()
    df['vol_30d'] = df['close'].rolling(30).std()
    df['vol_regime'] = (df['vol_10d'] > df['vol_30d'].rolling(30).median()).astype(int)
    
    df['vol_transition'] = df['close'].rolling(5).std() / df['close'].shift(5).rolling(5).std()
    
    # Volume Regime Analysis
    df['volume_30d_median'] = df['volume'].rolling(30).median()
    df['volume_regime'] = (df['volume'] > df['volume_30d_median']).astype(int)
    
    df['volume_clustering'] = df['volume'].rolling(5).apply(
        lambda x: (x > x.mean()).sum(), raw=True
    )
    
    # Regime persistence calculation
    df['volume_regime_persistence'] = 0
    for i in range(1, len(df)):
        if df['volume_regime'].iloc[i] == df['volume_regime'].iloc[i-1]:
            df['volume_regime_persistence'].iloc[i] = df['volume_regime_persistence'].iloc[i-1] + 1
    
    # Multi-Timeframe Confirmation System
    df['resistance_10d'] = df['high'].rolling(10).max().shift(1)
    df['support_10d'] = df['low'].rolling(10).min().shift(1)
    df['breakout_up'] = (df['close'] > df['resistance_10d']).astype(int)
    df['breakout_down'] = (df['close'] < df['support_10d']).astype(int)
    
    df['breakout_strength'] = (df['close'] - df['resistance_10d']) / (df['resistance_10d'] - df['support_10d'])
    df['breakout_strength'] = df['breakout_strength'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume-Price Divergence
    df['volume_momentum'] = df['volume'] / df['volume'].shift(5)
    df['divergence_strength'] = df['mom_short'] / df['volume_momentum']
    df['divergence_strength'] = df['divergence_strength'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Smart Money Flow
    df['smart_money_flow'] = (df['amount'] * np.sign(df['close'] - df['close'].shift(1))) / df['volume']
    df['smart_money_flow'] = df['smart_money_flow'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Extreme Move Framework
    df['ret_std_20d'] = df['ret_1'].rolling(20).std()
    df['price_extreme'] = (abs(df['ret_1']) > 2 * df['ret_std_20d'].shift(1)).astype(int)
    
    df['volume_20d_median'] = df['volume'].rolling(20).median()
    df['volume_extreme'] = (df['volume'] > 2 * df['volume_20d_median']).astype(int)
    
    df['combined_extreme'] = (df['price_extreme'] & df['volume_extreme']).astype(int)
    
    # Pre-extreme analysis
    df['pre_extreme_trend'] = df['ret_1'].rolling(5).mean().shift(1)
    df['volume_pattern'] = df['volume'] / df['volume'].rolling(5).mean().shift(1)
    
    # Composite factor calculation
    for i in range(30, len(df)):
        if pd.isna(df.iloc[i][['mom_short', 'mom_medium', 'mom_long']]).any():
            continue
            
        # Momentum component (40% weight)
        momentum_score = (
            0.4 * df['mom_short'].iloc[i] +
            0.3 * df['mom_medium'].iloc[i] +
            0.3 * df['mom_long'].iloc[i]
        ) * df['cross_timeframe_alignment'].iloc[i]
        
        # Regime component (25% weight)
        regime_score = 0
        if df['vol_regime'].iloc[i] == 1 and df['volume_regime'].iloc[i] == 1:
            regime_score = df['momentum_quality'].iloc[i]  # Momentum continuation
        elif df['vol_regime'].iloc[i] == 0 and df['volume_extreme'].iloc[i] == 1:
            regime_score = -df['mom_short'].iloc[i]  # Mean reversion
        elif df['vol_transition'].iloc[i] > 1.2 and df['volume_momentum_alignment'].iloc[i] == 1:
            regime_score = df['mom_acceleration'].iloc[i]  # Trend initiation
        
        # Confirmation component (25% weight)
        confirmation_score = (
            df['breakout_strength'].iloc[i] * 0.4 +
            (1 / (1 + abs(df['divergence_strength'].iloc[i]))) * 0.3 +
            df['smart_money_flow'].iloc[i] * 0.3
        )
        
        # Extreme move component (10% weight)
        extreme_score = 0
        if df['combined_extreme'].iloc[i] == 1:
            if np.sign(df['pre_extreme_trend'].iloc[i]) != np.sign(df['ret_1'].iloc[i]):
                extreme_score = -df['ret_1'].iloc[i] * 2  # Counter-trend extreme
            else:
                extreme_score = df['ret_1'].iloc[i] * 1.5  # Trend-confirming extreme
        
        # Final composite score
        result.iloc[i] = (
            0.4 * momentum_score +
            0.25 * regime_score +
            0.25 * confirmation_score +
            0.1 * extreme_score
        )
    
    return result
