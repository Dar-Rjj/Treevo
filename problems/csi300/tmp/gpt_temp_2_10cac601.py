import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Price Momentum with Decay
    momentum_5 = df['close'] - df['close'].shift(5)
    momentum_10 = df['close'] - df['close'].shift(10)
    
    # Apply exponential decay weighting (λ=0.95)
    decay_weights = np.array([0.95**i for i in range(10)])[::-1]
    decayed_momentum = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 10:
            recent_momentum = momentum_5.iloc[max(0, i-9):i+1].values
            if len(recent_momentum) == 10:
                decayed_momentum.iloc[i] = np.sum(recent_momentum * decay_weights)
            else:
                decayed_momentum.iloc[i] = np.nan
        else:
            decayed_momentum.iloc[i] = np.nan
    
    # Analyze Volume Confirmation Patterns
    volume_trend = df['volume'] / df['volume'].shift(5).rolling(window=5, min_periods=1).mean()
    
    price_change_pct = df['close'].pct_change()
    volume_20ma = df['volume'].rolling(window=20, min_periods=1).mean()
    high_volume_momentum_days = ((price_change_pct.abs() > 0.01) & (df['volume'] > volume_20ma)).astype(int)
    
    # Calculate volume-weighted price strength
    volume_weighted_strength = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 10:
            window_data = df.iloc[i-9:i+1]
            price_changes = window_data['close'].pct_change().fillna(0)
            volumes = window_data['volume']
            volume_weighted_strength.iloc[i] = np.sum(price_changes * np.sqrt(volumes))
        else:
            volume_weighted_strength.iloc[i] = np.nan
    
    # Detect Regime-Based Signals
    price_range = (df['high'] - df['low']) / df['close']
    volatility_regime = price_range.rolling(window=20, min_periods=1).mean()
    
    # Calculate ATR for regime detection
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean()
    atr_position = atr / df['close']
    
    # Identify trending vs mean-reverting periods
    price_ma_20 = df['close'].rolling(window=20, min_periods=1).mean()
    price_std_20 = df['close'].rolling(window=20, min_periods=1).std()
    price_position = (df['close'] - price_ma_20) / price_std_20
    trending_regime = (price_position.abs() > 1.0).astype(int)
    
    # Adjust momentum strength based on market regime
    regime_multiplier = pd.Series(1.0, index=df.index)
    high_vol_regime = (volatility_regime > volatility_regime.rolling(window=50, min_periods=1).quantile(0.7))
    regime_multiplier[high_vol_regime] = 0.7  # Reduce signal in high volatility
    regime_multiplier[trending_regime == 1] = 1.2  # Amplify in trending markets
    
    # Generate Composite Alpha Factor
    volume_confirmation = volume_trend * high_volume_momentum_days.rolling(window=5, min_periods=1).mean()
    
    composite_factor = (
        decayed_momentum * 
        volume_confirmation * 
        volume_weighted_strength * 
        regime_multiplier
    )
    
    return composite_factor
