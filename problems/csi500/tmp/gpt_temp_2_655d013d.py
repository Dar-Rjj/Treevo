import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors using microstructure dynamics and cross-asset signals
    """
    # Order Flow Imbalance Estimation
    # Estimate directional volume using price movements
    df['price_change'] = df['close'] - df['open']
    df['directional_volume'] = np.where(df['price_change'] > 0, df['volume'], 
                                       np.where(df['price_change'] < 0, -df['volume'], 0))
    
    # Tick-level directional volume aggregation (using rolling windows)
    df['of_imbalance_5d'] = df['directional_volume'].rolling(window=5).sum()
    df['of_imbalance_10d'] = df['directional_volume'].rolling(window=10).sum()
    
    # Imbalance persistence - autocorrelation of order flow
    df['of_persistence'] = df['directional_volume'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Large trade clustering intensity (using amount as proxy for trade size)
    df['large_trade_threshold'] = df['amount'].rolling(window=20).quantile(0.8)
    df['large_trade_count'] = (df['amount'] > df['large_trade_threshold']).rolling(window=5).sum()
    df['trade_clustering'] = df['large_trade_count'] / df['volume'].rolling(window=5).mean()
    
    # Cross-Asset Momentum Spillover (using sector relative strength)
    # Calculate relative strength within rolling window
    df['sector_rs'] = (df['close'] / df['close'].rolling(window=20).mean() - 1) - \
                      (df['close'].pct_change(periods=5).rolling(window=20).mean())
    
    # Commodity-currency cross impact (using volatility spillover)
    df['volatility_5d'] = (df['high'] - df['low']).rolling(window=5).std()
    df['volatility_20d'] = (df['high'] - df['low']).rolling(window=20).std()
    df['vol_spillover'] = df['volatility_5d'] / df['volatility_20d']
    
    # Volatility surface transmission effects
    df['intraday_range'] = (df['high'] - df['low']) / df['open']
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['vol_surface'] = df['intraday_range'] * (1 + abs(df['overnight_gap']))
    
    # Liquidity Regime Adaptive Factors
    # Bid-ask spread regime classification (using high-low range as proxy)
    df['spread_regime'] = pd.cut(
        df['intraday_range'], 
        bins=[0, 0.01, 0.03, 0.1, 1],
        labels=[1, 2, 3, 4]
    ).astype(float)
    
    # Market depth resilience scoring
    df['volume_trend'] = df['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-6)
    df['depth_resilience'] = df['volume_trend'] * (1 - df['price_efficiency'])
    
    # Liquidity shock propagation patterns
    df['volume_shock'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / \
                        df['volume'].rolling(window=20).std()
    df['price_shock'] = abs(df['close'].pct_change()) - df['close'].pct_change().rolling(window=20).std()
    df['liquidity_shock'] = df['volume_shock'] * df['price_shock']
    
    # Combine factors with appropriate weights
    factor = (
        0.25 * df['of_imbalance_10d'] / df['volume'].rolling(window=10).mean() +
        0.15 * df['of_persistence'].fillna(0) +
        0.10 * df['trade_clustering'].fillna(0) +
        0.20 * df['sector_rs'].fillna(0) +
        0.15 * df['vol_spillover'].fillna(0) +
        0.15 * df['depth_resilience'].fillna(0)
    )
    
    # Clean up intermediate columns
    cols_to_drop = ['price_change', 'directional_volume', 'of_imbalance_5d', 'of_imbalance_10d',
                   'of_persistence', 'large_trade_threshold', 'large_trade_count', 'trade_clustering',
                   'sector_rs', 'volatility_5d', 'volatility_20d', 'vol_spillover', 'intraday_range',
                   'overnight_gap', 'vol_surface', 'spread_regime', 'volume_trend', 'price_efficiency',
                   'depth_resilience', 'volume_shock', 'price_shock', 'liquidity_shock']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return factor
