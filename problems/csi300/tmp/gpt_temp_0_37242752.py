import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Alignment
    # Calculate 3/10/20-day returns from Close
    ret_3 = df['close'].pct_change(3)
    ret_10 = df['close'].pct_change(10)
    ret_20 = df['close'].pct_change(20)
    
    # Count matching positive/negative directions
    momentum_signs = pd.DataFrame({
        'ret_3': np.sign(ret_3),
        'ret_10': np.sign(ret_10),
        'ret_20': np.sign(ret_20)
    })
    
    # Sum aligned momentum magnitudes
    aligned_momentum = np.zeros(len(df))
    for i in range(len(df)):
        signs = momentum_signs.iloc[i]
        if not signs.isna().any():
            if signs.nunique() == 1:  # All same direction
                aligned_momentum[i] = ret_3.iloc[i] + ret_10.iloc[i] + ret_20.iloc[i]
            elif signs.value_counts().max() == 2:  # Two out of three agree
                majority_sign = signs.value_counts().index[0]
                aligned_momentum[i] = sum([ret_3.iloc[i], ret_10.iloc[i], ret_20.iloc[i]][j] 
                                        for j in range(3) if signs.iloc[j] == majority_sign)
    
    # Volatility Regime Adjustment
    # Compute True Range (High, Low, previous Close)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compare current TR to 20-day average
    tr_20_avg = true_range.rolling(window=20, min_periods=10).mean()
    tr_ratio = true_range / tr_20_avg
    
    # Apply regime weights
    regime_weights = np.ones(len(df))
    high_vol_mask = tr_ratio > 1.2
    low_vol_mask = tr_ratio < 0.8
    regime_weights[high_vol_mask] = 0.6  # Emphasize short-term
    regime_weights[low_vol_mask] = 1.4   # Emphasize long-term
    
    # Microstructure Confirmation
    # Intraday efficiency (|Close-Open|/True Range)
    intraday_efficiency = abs(df['close'] - df['open']) / true_range
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume intensity (Volume/10-day average Volume)
    vol_10_avg = df['volume'].rolling(window=10, min_periods=5).mean()
    volume_intensity = df['volume'] / vol_10_avg
    volume_intensity = volume_intensity.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Liquidity cost ((High-Low)/Close × |return|/Volume)
    daily_return = df['close'].pct_change()
    liquidity_cost = ((df['high'] - df['low']) / df['close']) * (abs(daily_return) / (df['volume'] + 1e-8))
    liquidity_cost = liquidity_cost.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Signal Integration
    # Multiply momentum score by microstructure factors
    microstructure_score = intraday_efficiency * volume_intensity * (1 - liquidity_cost)
    
    # Apply regime-specific scaling and output composite alpha factor
    alpha_factor = aligned_momentum * microstructure_score * regime_weights
    
    return pd.Series(alpha_factor, index=df.index)
