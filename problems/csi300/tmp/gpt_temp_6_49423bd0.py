import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate volatility-regime adaptive alpha factor combining momentum, efficiency, and range signals.
    """
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    
    # Volatility-Normalized Momentum Factors
    # 5-day momentum components
    mom_5_raw = df['close'] / df['close'].shift(5) - 1
    vol_5 = daily_returns.rolling(window=5).std()
    mom_5_vol_adj = mom_5_raw / vol_5.replace(0, np.nan)
    
    # 20-day momentum components
    mom_20_raw = df['close'] / df['close'].shift(20) - 1
    vol_20 = daily_returns.rolling(window=20).std()
    mom_20_vol_adj = mom_20_raw / vol_20.replace(0, np.nan)
    
    # Momentum persistence
    mom_acceleration = mom_5_vol_adj - mom_20_vol_adj
    
    # Calculate rolling correlation between 5-day and 20-day raw returns
    mom_stability = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window_5 = mom_5_raw.iloc[i-19:i+1]
        window_20 = mom_20_raw.iloc[i-19:i+1]
        if len(window_5.dropna()) >= 10 and len(window_20.dropna()) >= 10:
            mom_stability.iloc[i] = window_5.corr(window_20)
    
    # Volume-Price Efficiency Correlation Factors
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    efficiency_momentum = daily_efficiency / daily_efficiency.shift(5) - 1
    
    # Calculate efficiency-volume correlation
    eff_vol_corr = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window_eff = daily_efficiency.iloc[i-19:i+1]
        window_vol = df['volume'].iloc[i-19:i+1]
        if len(window_eff.dropna()) >= 10 and len(window_vol.dropna()) >= 10:
            eff_vol_corr.iloc[i] = window_eff.corr(window_vol)
    
    volume_confirmed_efficiency = efficiency_momentum * eff_vol_corr
    
    # Calculate efficiency-amount correlation
    eff_amt_corr = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window_eff = daily_efficiency.iloc[i-19:i+1]
        window_amt = df['amount'].iloc[i-19:i+1]
        if len(window_eff.dropna()) >= 10 and len(window_amt.dropna()) >= 10:
            eff_amt_corr.iloc[i] = window_eff.corr(window_amt)
    
    amount_weighted_efficiency = efficiency_momentum * eff_amt_corr
    
    # Volatility Regime Detection
    vol_60 = daily_returns.rolling(window=60).std()
    regime_high = vol_20 > (1.5 * vol_60)
    regime_low = vol_20 < (0.67 * vol_60)
    regime_normal = ~regime_high & ~regime_low
    
    # Regime-adaptive momentum
    regime_momentum = pd.Series(index=df.index, dtype=float)
    regime_momentum[regime_high] = mom_5_vol_adj[regime_high]
    regime_momentum[regime_low] = mom_20_vol_adj[regime_low]
    regime_momentum[regime_normal] = (mom_5_vol_adj[regime_normal] + mom_20_vol_adj[regime_normal]) / 2
    
    # Regime-adaptive efficiency
    regime_efficiency = pd.Series(index=df.index, dtype=float)
    regime_efficiency[regime_high] = amount_weighted_efficiency[regime_high]
    regime_efficiency[regime_low] = volume_confirmed_efficiency[regime_low]
    regime_efficiency[regime_normal] = (amount_weighted_efficiency[regime_normal] + volume_confirmed_efficiency[regime_normal]) / 2
    
    # Range-Based Volatility Signals
    normalized_range = (df['high'] - df['low']) / df['close']
    range_momentum_5 = normalized_range / normalized_range.shift(5) - 1
    range_momentum_20 = normalized_range / normalized_range.shift(20) - 1
    
    # Range-efficiency correlation
    range_eff_corr = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window_range = normalized_range.iloc[i-19:i+1]
        window_eff = daily_efficiency.iloc[i-19:i+1]
        if len(window_range.dropna()) >= 10 and len(window_eff.dropna()) >= 10:
            range_eff_corr.iloc[i] = window_range.corr(window_eff)
    
    range_efficiency_signal = range_momentum_5 * range_eff_corr
    
    # Composite Alpha Generation with regime-based weighting
    alpha_composite = pd.Series(index=df.index, dtype=float)
    
    # High volatility regime weights
    high_vol_mask = regime_high
    alpha_composite[high_vol_mask] = (
        0.4 * regime_momentum[high_vol_mask] +
        0.35 * range_efficiency_signal[high_vol_mask] +
        0.25 * regime_efficiency[high_vol_mask]
    )
    
    # Low volatility regime weights
    low_vol_mask = regime_low
    alpha_composite[low_vol_mask] = (
        0.35 * regime_momentum[low_vol_mask] +
        0.20 * range_efficiency_signal[low_vol_mask] +
        0.45 * regime_efficiency[low_vol_mask]
    )
    
    # Normal volatility regime weights
    normal_vol_mask = regime_normal
    alpha_composite[normal_vol_mask] = (
        (regime_momentum[normal_vol_mask] + 
         range_efficiency_signal[normal_vol_mask] + 
         regime_efficiency[normal_vol_mask]) / 3
    )
    
    # Signal enhancement with momentum persistence stability
    alpha_enhanced = alpha_composite * mom_stability.replace(np.nan, 1)
    
    # Adjust by regime confidence (distance from thresholds)
    regime_confidence = pd.Series(1.0, index=df.index)
    high_vol_confidence = (vol_20 / (1.5 * vol_60) - 1).clip(lower=0)
    low_vol_confidence = (1 - vol_20 / (0.67 * vol_60)).clip(lower=0)
    regime_confidence[regime_high] = 1 + high_vol_confidence[regime_high]
    regime_confidence[regime_low] = 1 + low_vol_confidence[regime_low]
    
    final_alpha = alpha_enhanced * regime_confidence
    
    return final_alpha
