import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns and true range
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Multi-Scale Momentum Efficiency
    # 5-day momentum
    mom_5 = df['close'] / df['close'].shift(5) - 1
    # 15-day momentum  
    mom_15 = df['close'] / df['close'].shift(15) - 1
    
    # Momentum persistence ratio (sign consistency)
    mom_persistence = (mom_5 * mom_15 > 0).astype(float)
    
    # Momentum efficiency (magnitude relative to price range)
    price_range_5 = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close'].shift(5)
    mom_efficiency_5 = abs(mom_5) / (price_range_5 + 1e-8)
    
    price_range_15 = (df['high'].rolling(15).max() - df['low'].rolling(15).min()) / df['close'].shift(15)
    mom_efficiency_15 = abs(mom_15) / (price_range_15 + 1e-8)
    
    # Combined momentum efficiency score
    momentum_efficiency = (mom_efficiency_5 + mom_efficiency_15) / 2 * mom_persistence
    
    # Volume-Volatility Regime Detection
    # Volume regime using 10-day percentile and acceleration
    volume_10d_median = df['volume'].rolling(10).median()
    volume_percentile = df['volume'].rolling(10).apply(lambda x: (x[-1] > np.percentile(x, 70)).astype(float), raw=True)
    volume_acceleration = df['volume'].pct_change(3) > df['volume'].pct_change(3).rolling(10).mean()
    volume_regime = (volume_percentile + volume_acceleration.astype(float)) / 2
    
    # Volatility regime using 10-day true range average
    vol_10d_avg = df['true_range'].rolling(10).mean()
    vol_regime = (df['true_range'] > vol_10d_avg).astype(float)
    
    # Regime-Adaptive Signal Generation
    # High volume + high volatility regime
    high_regime = (volume_regime > 0.6) & (vol_regime > 0.5)
    # Low volume + low volatility regime  
    low_regime = (volume_regime < 0.4) & (vol_regime < 0.5)
    
    # Regime-specific momentum thresholds
    high_regime_signal = momentum_efficiency * (mom_5 > 0.02).astype(float) * (mom_15 > 0.05).astype(float)
    low_regime_signal = momentum_efficiency * (mom_5 > 0.01).astype(float) * (mom_15 > 0.02).astype(float)
    normal_regime_signal = momentum_efficiency * (mom_5 > 0.015).astype(float) * (mom_15 > 0.03).astype(float)
    
    # Validate signals with volume-volatility alignment
    volume_vol_alignment = (volume_regime * vol_regime).clip(0, 1)
    
    # Dynamic Alpha Factor Construction
    # Combine regime-conditional momentum signals
    regime_signal = np.where(high_regime, high_regime_signal,
                   np.where(low_regime, low_regime_signal, normal_regime_signal))
    
    # Generate risk-adjusted prediction score
    risk_adjustment = 1 / (df['true_range'].rolling(10).std() + 1e-8)
    alpha_factor = regime_signal * volume_vol_alignment * risk_adjustment
    
    return pd.Series(alpha_factor, index=df.index)
