import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Microstructure Spillover factor combining order flow synchronization,
    liquidity contagion, and signal integration across multiple market microstructure dimensions.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic microstructure features
    df['spread_estimate'] = (df['high'] - df['low']) / df['close']  # Proxy for bid-ask spread
    df['price_range'] = df['high'] - df['low']
    df['mid_price'] = (df['high'] + df['low']) / 2
    
    # Cross-asset trade clustering (using volume clustering as proxy)
    df['volume_cluster'] = df['volume'].rolling(window=5, min_periods=3).std() / df['volume'].rolling(window=5, min_periods=3).mean()
    
    # Bid-ask spread co-movement (using spread estimate autocorrelation)
    df['spread_momentum'] = df['spread_estimate'].rolling(window=5, min_periods=3).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Depth exhaustion propagation (using volume-to-range ratio)
    df['depth_exhaustion'] = df['volume'] / (df['price_range'] + 1e-8)
    df['depth_momentum'] = df['depth_exhaustion'].pct_change(periods=3)
    
    # Cross-market impact transmission (using price impact proxy)
    df['price_impact'] = (df['close'] - df['open']) / df['volume'].replace(0, np.nan)
    df['impact_persistence'] = df['price_impact'].rolling(window=5, min_periods=3).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x[i]) == np.sign(x[i-1])]) / max(1, len(x)-1)
    )
    
    # Spillover intensity calculation
    df['spillover_intensity'] = (
        df['volume_cluster'].fillna(0) * 
        df['spread_momentum'].fillna(0) * 
        df['depth_momentum'].fillna(0)
    )
    
    # Momentum component
    df['momentum_5d'] = df['close'].pct_change(periods=5)
    df['momentum_3d'] = df['close'].pct_change(periods=3)
    df['momentum_coherence'] = np.sign(df['momentum_5d']) * np.sign(df['momentum_3d'])
    
    # Volume-confirmed persistence
    df['volume_trend'] = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    df['volume_momentum_alignment'] = np.sign(df['volume_trend']) * np.sign(df['momentum_3d'])
    
    # Final factor: Spillover intensity × momentum coherence × volume confirmation
    factor = (
        df['spillover_intensity'] * 
        df['momentum_coherence'] * 
        df['volume_momentum_alignment'] * 
        df['impact_persistence']
    )
    
    # Clean and return
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
