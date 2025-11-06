import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate true range
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Gap-Momentum Divergence Component
    df['fast_gap_momentum'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['slow_momentum'] = df['close'] / df['close'].shift(10) - 1
    df['gap_momentum_divergence'] = df['fast_gap_momentum'] - df['slow_momentum']
    
    # Volume Regime Component
    df['volume_stability'] = df['volume'].rolling(window=10).std() / df['volume'].rolling(window=10).mean()
    
    # Volume surge using rolling mean of last 5 days including current
    df['volume_surge'] = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Volume consistency - count of days where volume > mean of previous 5 days
    def volume_consistency_calc(series):
        if len(series) < 5:
            return np.nan
        current_vol = series.iloc[-1]
        window = series.iloc[-5:]
        count = 0
        for i in range(len(window)):
            if i >= 4:  # Need at least 5 elements for rolling mean
                vol_val = window.iloc[i]
                mean_prev = window.iloc[max(0, i-4):i+1].mean()
                if vol_val > mean_prev:
                    count += 1
        return count / 5
    
    df['volume_consistency'] = df['volume'].rolling(window=5).apply(
        volume_consistency_calc, raw=False
    )
    
    # Volatility Regime Component
    df['volatility_clustering'] = df['true_range'] / df['true_range'].rolling(window=9).mean().shift(1)
    
    df['realization_efficiency'] = (
        abs(df['close'] - df['open']) / 
        (abs(df['open'] - df['close'].shift(1)) + 0.0001)
    )
    
    # Alpha Integration
    df['volume_regime_adjustment'] = (
        df['volume_stability'] * df['volume_surge'] * df['volume_consistency']
    )
    
    df['volatility_adjustment'] = (
        df['volatility_clustering'] * df['realization_efficiency']
    )
    
    # Final alpha
    alpha = df['gap_momentum_divergence'] * df['volume_regime_adjustment'] / df['volatility_adjustment']
    
    return alpha
