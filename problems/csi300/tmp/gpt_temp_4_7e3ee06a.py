import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Microstructure Contagion Factor
    Detects information flow and contagion patterns across related instruments
    using microstructure relationships and order flow synchronization
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic microstructure features
    df['spread'] = (df['high'] - df['low']) / df['close']
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['price_range'] = df['high'] - df['low']
    df['volume_weighted_price'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan)
    
    # Calculate lead-lag microstructure relationships (using rolling correlations)
    window_size = 20
    
    # Price-based contagion measures
    df['price_momentum'] = df['close'].pct_change(periods=5)
    df['volatility'] = df['close'].pct_change().rolling(window=10).std()
    
    # Cross-market order flow synchronization (using volume and amount patterns)
    df['volume_momentum'] = df['volume'].pct_change(periods=5)
    df['amount_momentum'] = df['amount'].pct_change(periods=5)
    
    # Calculate microstructure correlation decay patterns
    df['micro_corr_5'] = df['close'].pct_change().rolling(window=5).corr(df['volume'].pct_change())
    df['micro_corr_10'] = df['close'].pct_change().rolling(window=10).corr(df['volume'].pct_change())
    
    # Liquidity spillover intensity (using spread and volume relationships)
    df['liquidity_ratio'] = df['volume'] / (df['spread'] + 1e-8)
    df['liquidity_momentum'] = df['liquidity_ratio'].pct_change(periods=5)
    
    # Volatility transmission pathways
    df['volatility_ratio'] = df['volatility'] / (df['volume'].rolling(window=10).std() + 1e-8)
    
    # Adaptive contagion weights based on transmission reliability
    df['transmission_reliability'] = (
        df['micro_corr_5'].abs().rolling(window=15).mean() * 
        (1 - df['volatility_ratio'].rolling(window=15).std())
    )
    
    # Calculate contagion persistence (autocorrelation of microstructure signals)
    df['contagion_persistence'] = (
        df['price_momentum'].rolling(window=10).apply(lambda x: x.autocorr(), raw=False) * 
        df['volume_momentum'].rolling(window=10).apply(lambda x: x.autocorr(), raw=False)
    )
    
    # Regime stability filter (low volatility periods indicate stable transmission)
    df['regime_stability'] = 1 / (1 + df['volatility'].rolling(window=20).std())
    
    # Cross-market information advantage (early detection of trends)
    df['info_advantage'] = (
        df['price_momentum'].rolling(window=5).mean() * 
        df['volume_momentum'].rolling(window=5).mean() * 
        df['transmission_reliability']
    )
    
    # Market integration degree (correlation between price and volume dynamics)
    df['market_integration'] = (
        df['micro_corr_10'].abs().rolling(window=15).mean() * 
        df['liquidity_momentum'].abs().rolling(window=15).mean()
    )
    
    # Combine all components into final contagion factor
    contagion_factor = (
        df['info_advantage'] * 
        df['transmission_reliability'] * 
        df['contagion_persistence'] * 
        df['regime_stability'] * 
        df['market_integration']
    )
    
    # Normalize the factor
    contagion_factor = (contagion_factor - contagion_factor.rolling(window=50).mean()) / (
        contagion_factor.rolling(window=50).std() + 1e-8
    )
    
    # Clean up intermediate columns
    cols_to_drop = ['spread', 'mid_price', 'price_range', 'volume_weighted_price', 
                   'price_momentum', 'volatility', 'volume_momentum', 'amount_momentum',
                   'micro_corr_5', 'micro_corr_10', 'liquidity_ratio', 'liquidity_momentum',
                   'volatility_ratio', 'transmission_reliability', 'contagion_persistence',
                   'regime_stability', 'info_advantage', 'market_integration']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return contagion_factor
