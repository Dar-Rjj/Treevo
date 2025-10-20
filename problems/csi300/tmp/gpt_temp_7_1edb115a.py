import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Price Momentum Components
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_accel'] = df['momentum_5d'] - df['momentum_10d']
    
    # Volume Dynamics
    df['volume_trend'] = df['volume'].pct_change(5)
    df['volume_adjusted_range'] = np.log((df['high'] - df['low']) * df['volume'])
    df['volume_persistence'] = df['volume'].rolling(window=5).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    )
    
    # Volatility Regime
    df['volatility_5d'] = df['returns'].rolling(window=5).std()
    df['atr'] = (
        np.maximum(df['high'] - df['low'], 
                  np.maximum(abs(df['high'] - df['close'].shift(1)), 
                            abs(df['low'] - df['close'].shift(1))))
    )
    df['volatility_10d'] = df['atr'].rolling(window=10).mean()
    df['volatility_regime'] = df['volatility_5d'] / df['volatility_10d']
    
    # Price-Volume Divergence
    df['price_volume_divergence'] = (
        np.sign(df['momentum_5d']) * np.sign(df['volume_trend']) * 
        df['volume_adjusted_range']
    )
    
    # Momentum Acceleration Divergence
    df['momentum_accel_divergence'] = (
        np.sign(df['momentum_accel']) * np.sign(df['volume_persistence']) * 
        abs(df['momentum_accel'])
    )
    
    # Combined divergence score
    df['divergence_score'] = (
        df['price_volume_divergence'] + df['momentum_accel_divergence']
    )
    
    # Volatility-weighted alpha signal
    volatility_weight = np.where(
        df['volatility_regime'] > 1.2,  # High volatility regime
        1.5,
        np.where(
            df['volatility_regime'] < 0.8,  # Low volatility regime
            0.7,
            1.0  # Normal volatility regime
        )
    )
    
    # Final factor calculation
    factor = df['divergence_score'] * volatility_weight
    
    return factor
