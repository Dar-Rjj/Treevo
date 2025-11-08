import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    
    # Multi-Timeframe Momentum
    momentum_2d = df['close'] / df['close'].shift(2) - 1
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    
    # Volatility Regime Framework
    vol_3d = daily_returns.rolling(window=3).std()
    vol_10d = daily_returns.rolling(window=10).std()
    high_vol_regime = vol_3d > vol_10d
    
    # Volume-Price Dynamics
    # Volume calculations
    volume_ma_3 = df['volume'].rolling(window=3).mean()
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    
    volume_intensity = df['volume'] / volume_ma_3 - 1
    volume_momentum = volume_ma_3 / volume_ma_3.shift(5) - 1
    
    # Price-Volume Synchronization
    volume_changes = df['volume'].pct_change()
    
    def rolling_corr_3d(returns, vol_changes):
        corrs = []
        for i in range(len(returns)):
            if i < 2:
                corrs.append(np.nan)
            else:
                window_returns = returns.iloc[i-2:i+1]
                window_vol = vol_changes.iloc[i-2:i+1]
                if len(window_returns.dropna()) >= 2 and len(window_vol.dropna()) >= 2:
                    corr = window_returns.corr(window_vol)
                    corrs.append(corr if not pd.isna(corr) else 0)
                else:
                    corrs.append(0)
        return pd.Series(corrs, index=returns.index)
    
    def rolling_corr_5d(returns, vol_changes):
        corrs = []
        for i in range(len(returns)):
            if i < 4:
                corrs.append(np.nan)
            else:
                window_returns = returns.iloc[i-4:i+1]
                window_vol = vol_changes.iloc[i-4:i+1]
                if len(window_returns.dropna()) >= 3 and len(window_vol.dropna()) >= 3:
                    corr = window_returns.corr(window_vol)
                    corrs.append(corr if not pd.isna(corr) else 0)
                else:
                    corrs.append(0)
        return pd.Series(corrs, index=returns.index)
    
    corr_3d = rolling_corr_3d(daily_returns, volume_changes)
    corr_5d = rolling_corr_5d(daily_returns, volume_changes)
    
    # Volume Confirmation Strength
    strong_confirmation = (volume_intensity > 0) & (volume_momentum > 0)
    weak_confirmation = (volume_intensity > 0) | (volume_momentum > 0)
    divergence = (volume_intensity < 0) & (volume_momentum < 0)
    
    # Regime-Optimized Momentum Selection
    base_momentum = momentum_2d.copy()
    base_momentum[~high_vol_regime] = momentum_10d[~high_vol_regime]
    
    # Volume Confirmation Multiplier
    multiplier = pd.Series(1.0, index=df.index)
    multiplier[strong_confirmation] = 2.0
    multiplier[weak_confirmation] = 1.2
    multiplier[divergence] = 0.3
    
    # Correlation-Based Direction Filter
    direction_weight = pd.Series(1.0, index=df.index)
    positive_sync = (corr_3d > 0.1) & (corr_5d > 0.1)
    negative_sync = (corr_3d < -0.1) & (corr_5d < -0.1)
    mixed_signals = ~positive_sync & ~negative_sync
    
    direction_weight[positive_sync] = 1.0
    direction_weight[negative_sync] = -1.0
    direction_weight[mixed_signals] = (corr_3d[mixed_signals] + corr_5d[mixed_signals]) / 2
    
    # Final Factor Construction
    factor = base_momentum * multiplier * direction_weight
    
    return factor
