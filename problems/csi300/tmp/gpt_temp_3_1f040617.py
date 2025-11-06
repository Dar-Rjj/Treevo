import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Regime Volume Divergence
    # Calculate true range
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Volatility regime classification using rolling percentiles
    df['volatility_20d_percentile'] = df['true_range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.2)) / (x.quantile(0.8) - x.quantile(0.2)) if x.quantile(0.8) != x.quantile(0.2) else 0.5
    )
    
    # Volume baseline by volatility regime
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma_20']
    
    # Volatility-regime volume divergence
    volatility_divergence = np.where(
        df['volatility_20d_percentile'] > 0.7,
        -df['volume_ratio'],  # High volatility + declining volume = exhaustion
        np.where(
            df['volatility_20d_percentile'] < 0.3,
            df['volume_ratio'],  # Low volatility + spiking volume = breakout precursor
            0
        )
    )
    
    # Price-Efficiency Momentum Gradient
    # Directional efficiency ratio
    df['price_range'] = df['high'] - df['low']
    df['net_move'] = abs(df['close'] - df['close'].shift(1))
    df['efficiency_ratio'] = df['net_move'] / df['price_range'].replace(0, np.nan)
    df['efficiency_ratio'] = df['efficiency_ratio'].fillna(0)
    
    # Efficiency momentum across periods
    df['eff_momentum_5'] = df['efficiency_ratio'].rolling(window=5).mean()
    df['eff_momentum_10'] = df['efficiency_ratio'].rolling(window=10).mean()
    
    # Efficiency gradient (acceleration/deceleration)
    efficiency_gradient = (df['eff_momentum_5'] - df['eff_momentum_10']) / df['eff_momentum_10'].replace(0, np.nan)
    efficiency_gradient = efficiency_gradient.fillna(0)
    
    # Intraday Pressure Imbalance Cascade
    # Multi-timeframe pressure calculation
    df['pressure_1d'] = (2 * df['close'] - df['low'] - df['high']) / (df['high'] - df['low']).replace(0, np.nan)
    df['pressure_1d'] = df['pressure_1d'].fillna(0)
    
    df['pressure_3d'] = (df['close'].rolling(window=3).mean() - df['low'].rolling(window=3).min()) / \
                       (df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()).replace(0, np.nan)
    df['pressure_3d'] = df['pressure_3d'].fillna(0.5)
    
    df['pressure_5d'] = (df['close'].rolling(window=5).mean() - df['low'].rolling(window=5).min()) / \
                       (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()).replace(0, np.nan)
    df['pressure_5d'] = df['pressure_5d'].fillna(0.5)
    
    # Pressure divergence across timeframes
    pressure_divergence = (df['pressure_1d'] - df['pressure_3d']) + (df['pressure_3d'] - df['pressure_5d'])
    
    # Range-Expansion Volume Confirmation
    # Range expansion detection
    df['range_5d_ma'] = (df['high'] - df['low']).rolling(window=5).mean()
    df['range_expansion'] = (df['high'] - df['low']) / df['range_5d_ma'].replace(0, np.nan) - 1
    df['range_expansion'] = df['range_expansion'].fillna(0)
    
    # Volume confirmation analysis
    range_volume_confirmation = df['range_expansion'] * df['volume_ratio']
    
    # Momentum-Regime Transition Matrix
    # Momentum state identification
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_acceleration'] = df['momentum_5'] - df['momentum_10']
    
    # Regime context with volume
    momentum_transition = df['momentum_acceleration'] * np.sign(df['volume_ratio'] - 1)
    
    # Combine all factors with weights
    factor = (
        0.25 * volatility_divergence +
        0.25 * efficiency_gradient +
        0.20 * pressure_divergence +
        0.15 * range_volume_confirmation +
        0.15 * momentum_transition
    )
    
    return pd.Series(factor, index=df.index)
