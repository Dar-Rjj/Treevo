import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Asymmetry Regime Analysis factor
    Analyzes volatility regimes and asymmetric volume patterns to generate trading signals
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate daily returns and price ranges
    returns = df['close'].pct_change()
    price_range = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Volatility Regime Detection
    volatility_20d = returns.rolling(window=20, min_periods=10).std()
    volatility_regime = pd.Series(index=df.index, dtype=str)
    volatility_regime[:] = 'normal'
    volatility_regime[volatility_20d > volatility_20d.rolling(window=60).quantile(0.7)] = 'high'
    volatility_regime[volatility_20d < volatility_20d.rolling(window=60).quantile(0.3)] = 'low'
    
    # Asymmetric Volume Patterns
    up_days = returns > 0
    down_days = returns < 0
    
    # Rolling volume asymmetry (20-day window)
    up_volume = df['volume'].where(up_days, 0).rolling(window=20, min_periods=10).mean()
    down_volume = df['volume'].where(down_days, 0).rolling(window=20, min_periods=10).mean()
    volume_asymmetry_ratio = (up_volume - down_volume) / (up_volume + down_volume + 1e-8)
    
    # Volume-price directional consistency
    volume_price_corr = df['volume'].rolling(window=20, min_periods=10).corr(returns.abs())
    
    # Price Movement Quality Assessment
    directional_efficiency = returns.abs() / (price_range + 1e-8)
    gap_behavior = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Intraday momentum persistence
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    momentum_persistence = intraday_momentum.rolling(window=5, min_periods=3).mean()
    
    # Regime-Specific Asymmetry Signals
    for i in range(len(df)):
        if i < 20:  # Skip initial period for rolling calculations
            result.iloc[i] = 0
            continue
            
        current_vol_regime = volatility_regime.iloc[i]
        current_volume_asymmetry = volume_asymmetry_ratio.iloc[i]
        current_directional_efficiency = directional_efficiency.iloc[i]
        current_momentum_persistence = momentum_persistence.iloc[i]
        
        # Signal generation based on regime
        if current_vol_regime == 'high':
            # High volatility + volume asymmetry → momentum signal
            if current_volume_asymmetry > 0.1 and current_momentum_persistence > 0:
                signal = current_volume_asymmetry * current_momentum_persistence
            elif current_volume_asymmetry < -0.1 and current_momentum_persistence < 0:
                signal = current_volume_asymmetry * current_momentum_persistence
            else:
                signal = 0
                
        elif current_vol_regime == 'low':
            # Low volatility + price-volume divergence → breakout signal
            volume_price_divergence = abs(volume_price_corr.iloc[i])
            if volume_price_divergence < 0.2 and abs(current_volume_asymmetry) > 0.15:
                signal = current_volume_asymmetry * (1 - volume_price_divergence)
            else:
                signal = 0
                
        else:  # normal regime
            # Regime transition detection
            vol_persistence = volatility_20d.iloc[i] / volatility_20d.iloc[i-5:i].mean()
            if abs(vol_persistence - 1) > 0.3 and abs(current_volume_asymmetry) > 0.1:
                # Regime transition + asymmetry reversal → trend change signal
                signal = -current_volume_asymmetry * vol_persistence
            else:
                signal = current_volume_asymmetry * current_directional_efficiency
        
        # Apply regime-specific scaling
        if current_vol_regime == 'high':
            signal *= 1.5
        elif current_vol_regime == 'low':
            signal *= 0.8
            
        result.iloc[i] = signal
    
    # Normalize the final factor
    result = (result - result.rolling(window=60, min_periods=20).mean()) / result.rolling(window=60, min_periods=20).std()
    
    return result.fillna(0)
