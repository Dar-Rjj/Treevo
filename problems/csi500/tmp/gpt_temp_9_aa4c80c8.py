import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Assessment
    vol_10d = df['close'].rolling(window=10).std()
    vol_30d = df['close'].rolling(window=30).std()
    vol_ratio = vol_10d / vol_30d
    high_vol_regime = vol_ratio > 1.2
    
    # Microstructure Momentum Components
    price_momentum = df['close'] / df['close'].shift(1) - 1
    volume_momentum = df['volume'] / df['volume'].shift(1) - 1
    daily_range = df['high'] - df['low']
    vol_momentum = daily_range / daily_range.shift(1) - 1
    price_acceleration = (df['close'] / df['close'].shift(5) - 1) - (df['close'] / df['close'].shift(10) - 1)
    
    # Microstructure Divergence Analysis
    mid_range = (df['high'] + df['low']) / 2
    close_to_mid_dev = df['close'] / mid_range - 1
    
    # Efficiency asymmetry calculation with zero division protection
    high_open_diff = df['high'] - df['open']
    open_low_diff = df['open'] - df['low']
    efficiency_asymmetry = np.where(
        (high_open_diff != 0) & (open_low_diff != 0),
        (df['close'] - df['open']) / high_open_diff - (df['open'] - df['close']) / open_low_diff,
        0
    )
    
    intraday_rejection = (df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_rejection = intraday_rejection.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Price-volume divergence correlation
    price_vol_divergence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window_prices = price_momentum.iloc[i-4:i+1]
        window_volumes = volume_momentum.iloc[i-4:i+1]
        if len(window_prices.dropna()) >= 3 and len(window_volumes.dropna()) >= 3:
            price_vol_divergence.iloc[i] = window_prices.corr(window_volumes)
        else:
            price_vol_divergence.iloc[i] = 0
    price_vol_divergence = price_vol_divergence.fillna(0)
    
    # Liquidity Dynamics Integration
    volume_velocity = df['volume'] / df['volume'].shift(5) - 1
    volume_spike = df['volume'] / df['volume'].rolling(window=5, min_periods=1).median()
    signed_volume = df['volume'] * np.sign(df['close'] - df['open'])
    
    # Regime-Based Factor Synthesis
    primary_factor = pd.Series(index=df.index, dtype=float)
    secondary_factor = pd.Series(index=df.index, dtype=float)
    
    # High Volatility Regime Processing
    high_vol_mask = high_vol_regime.fillna(False)
    momentum_divergence = price_momentum * close_to_mid_dev
    volatility_efficiency = efficiency_asymmetry * vol_ratio
    
    primary_factor[high_vol_mask] = (
        momentum_divergence[high_vol_mask] * 
        volatility_efficiency[high_vol_mask] * 
        volume_spike[high_vol_mask]
    )
    
    secondary_factor[high_vol_mask] = (
        price_acceleration[high_vol_mask] * 
        signed_volume[high_vol_mask] * 
        intraday_rejection[high_vol_mask]
    )
    
    # Normal Volatility Regime Processing
    normal_vol_mask = ~high_vol_mask
    momentum_convergence = price_vol_divergence * price_acceleration
    liquidity_momentum = volume_velocity * signed_volume
    
    primary_factor[normal_vol_mask] = (
        momentum_convergence[normal_vol_mask] * 
        liquidity_momentum[normal_vol_mask] * 
        intraday_rejection[normal_vol_mask]
    )
    
    secondary_factor[normal_vol_mask] = (
        close_to_mid_dev[normal_vol_mask] * 
        efficiency_asymmetry[normal_vol_mask] * 
        volume_velocity[normal_vol_mask]
    )
    
    # Volatility-Adjusted Final Factor
    # True range calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    true_range = true_range.replace(0, 0.0001)  # Handle zero range
    
    # Regime factor adjustment
    final_factor = pd.Series(index=df.index, dtype=float)
    final_factor[high_vol_mask] = primary_factor[high_vol_mask] / true_range[high_vol_mask]
    final_factor[normal_vol_mask] = primary_factor[normal_vol_mask] * (1 + secondary_factor[normal_vol_mask])
    
    # Momentum Persistence Enhancement
    factor_momentum = final_factor / final_factor.shift(3) - 1
    
    # Volatility streak integration
    vol_momentum_sign = np.sign(vol_momentum)
    vol_streak = pd.Series(0, index=df.index, dtype=int)
    current_streak = 0
    current_sign = 0
    
    for i in range(1, len(df)):
        if pd.notna(vol_momentum_sign.iloc[i]) and pd.notna(vol_momentum_sign.iloc[i-1]):
            if vol_momentum_sign.iloc[i] == vol_momentum_sign.iloc[i-1] and vol_momentum_sign.iloc[i] != 0:
                current_streak += 1
            else:
                current_streak = 1 if vol_momentum_sign.iloc[i] != 0 else 0
            vol_streak.iloc[i] = current_streak
    
    # Apply persistence multiplier
    enhanced_factor = final_factor * (1 + factor_momentum * (1 + vol_streak * 0.1))
    
    return enhanced_factor.fillna(0)
