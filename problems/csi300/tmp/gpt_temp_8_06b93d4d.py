import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Lagged Volatility-Price Divergence Alpha Factor
    Detects predictive relationships between price momentum and volatility regimes
    """
    df = data.copy()
    
    # Calculate Price-Based Components
    # Rolling Price Momentum
    df['price_momentum_5d'] = df['close'].pct_change(5)
    df['price_momentum_10d'] = df['close'].pct_change(10)
    
    # Price Trend Strength using Directional Movement
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    df['plus_dm'] = np.where(
        (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
        np.maximum(df['high'] - df['high'].shift(1), 0),
        0
    )
    
    df['minus_dm'] = np.where(
        (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
        np.maximum(df['low'].shift(1) - df['low'], 0),
        0
    )
    
    # 8-day trend consistency
    df['trend_strength_8d'] = (
        df['plus_dm'].rolling(8).sum() - df['minus_dm'].rolling(8).sum()
    ) / df['tr'].rolling(8).sum()
    
    # Calculate Volatility-Based Components
    # Multiple Volatility Measures
    df['atr_5d'] = df['tr'].rolling(5).mean()
    df['hl_vol_10d'] = (df['high'] - df['low']).rolling(10).std()
    df['close_vol_15d'] = df['close'].pct_change().rolling(15).std()
    
    # Volatility Regime Analysis
    df['vol_acceleration'] = df['atr_5d'].pct_change(3)
    df['vol_persistence'] = df['atr_5d'].rolling(5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0
    )
    
    # Establish Lagged Relationships
    # Price Leading Volatility Analysis (1-3 day lags)
    price_vol_corrs = []
    for lag in [1, 2, 3]:
        corr = df['price_momentum_5d'].rolling(10).corr(
            df['atr_5d'].shift(-lag)
        )
        price_vol_corrs.append(corr)
    
    df['price_leading_vol'] = pd.concat(price_vol_corrs, axis=1).mean(axis=1)
    
    # Volatility Leading Price Analysis (1-3 day lags)
    vol_price_corrs = []
    for lag in [1, 2, 3]:
        corr = df['atr_5d'].rolling(10).corr(
            df['price_momentum_5d'].shift(-lag)
        )
        vol_price_corrs.append(corr)
    
    df['vol_leading_price'] = pd.concat(vol_price_corrs, axis=1).mean(axis=1)
    
    # Generate Divergence Signals
    # Price-Volatility Divergence Detection
    df['momentum_strength'] = df['price_momentum_5d'].abs()
    df['volatility_change'] = df['atr_5d'].pct_change()
    
    # Divergence conditions
    df['divergence_1'] = np.where(
        (df['momentum_strength'] > df['momentum_strength'].rolling(10).mean()) &
        (df['volatility_change'] < df['volatility_change'].rolling(10).mean()),
        df['momentum_strength'] * (1 - df['volatility_change']),
        0
    )
    
    df['divergence_2'] = np.where(
        (df['momentum_strength'] < df['momentum_strength'].rolling(10).mean()) &
        (df['volatility_change'] > df['volatility_change'].rolling(10).mean()),
        -df['momentum_strength'] * df['volatility_change'],
        0
    )
    
    # Signal Strength Assessment
    df['divergence_persistence'] = (
        df['divergence_1'].rolling(5).sum() + 
        df['divergence_2'].rolling(5).sum()
    )
    
    # Combine components with weights
    df['lagged_divergence_alpha'] = (
        0.3 * df['price_leading_vol'] +
        0.3 * df['vol_leading_price'] +
        0.2 * (df['divergence_1'] + df['divergence_2']) +
        0.2 * df['divergence_persistence']
    )
    
    # Final normalization
    alpha = df['lagged_divergence_alpha'].fillna(0)
    return alpha
