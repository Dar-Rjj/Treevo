import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    df['returns'] = df['close'].pct_change()
    
    # True Range calculation
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['close'].shift(1))
    df['tr3'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Price Efficiency: close-to-close returns / true range
    df['price_efficiency'] = abs(df['returns']) / df['true_range']
    df['price_efficiency'] = df['price_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volatility-adjusted price efficiency (using 10-day rolling volatility)
    df['volatility'] = df['returns'].rolling(window=10, min_periods=5).std()
    df['vol_adj_price_efficiency'] = df['price_efficiency'] / (df['volatility'] + 1e-8)
    
    # Volume-to-price impact ratio
    df['price_change'] = abs(df['close'] - df['close'].shift(1))
    df['volume_price_ratio'] = df['volume'] / (df['price_change'] + 1e-8)
    
    # Volume concentration analysis (20-day rolling quantiles)
    df['volume_quantile'] = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Volume efficiency divergence
    df['volume_efficiency'] = df['volume_price_ratio'].rolling(window=5, min_periods=3).mean()
    df['efficiency_divergence'] = df['price_efficiency'] - df['volume_efficiency'].rolling(window=5, min_periods=3).mean()
    
    # Multi-timeframe efficiency analysis
    # Short-term efficiency momentum (3-day rate of change)
    df['efficiency_momentum'] = df['price_efficiency'].pct_change(periods=3)
    
    # Medium-term efficiency stability (10-day variance)
    df['efficiency_stability'] = df['price_efficiency'].rolling(window=10, min_periods=5).var()
    
    # Normalize components for combination
    components = ['vol_adj_price_efficiency', 'efficiency_divergence', 'efficiency_momentum']
    
    for col in components:
        df[f'{col}_norm'] = (df[col] - df[col].rolling(window=20, min_periods=10).mean()) / \
                           (df[col].rolling(window=20, min_periods=10).std() + 1e-8)
    
    # Generate alpha factor: weighted combination of normalized components
    # Positive weights for efficiency and momentum, negative for divergence (when efficiency > volume efficiency)
    alpha = (0.4 * df['vol_adj_price_efficiency_norm'] + 
             0.3 * df['efficiency_momentum_norm'] - 
             0.3 * df['efficiency_divergence_norm'])
    
    # Clean up intermediate columns
    drop_cols = ['returns', 'tr1', 'tr2', 'tr3', 'true_range', 'price_efficiency', 
                'volatility', 'vol_adj_price_efficiency', 'price_change', 
                'volume_price_ratio', 'volume_quantile', 'volume_efficiency', 
                'efficiency_divergence', 'efficiency_momentum', 'efficiency_stability',
                'vol_adj_price_efficiency_norm', 'efficiency_divergence_norm', 'efficiency_momentum_norm']
    
    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return alpha
