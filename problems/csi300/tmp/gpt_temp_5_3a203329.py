import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Regime Volatility-Liquidity Fragmentation Alpha
    Combines volatility transmission, microstructure fragmentation, and regime detection
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Basic price and volume calculations
    df['range'] = df['high'] - df['low']
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # 1. Price Impact Fragmentation (Volatility-adjusted)
    df['price_impact_frag'] = (df['range'] / df['amount']) - (df['range'].shift(1) / df['amount'].shift(1))
    
    # 2. Volume Fragmentation Momentum
    df['volume_frag_momentum'] = (df['volume'] / df['volume'].shift(1)) - (df['volume'].shift(1) / df['volume'].shift(2))
    
    # 3. Volatility Acceleration
    df['vol_acceleration'] = (df['range'] / df['range'].shift(1)) - (df['range'].shift(1) / df['range'].shift(2))
    
    # 4. Price-Fragmentation Divergence (3-day lookback)
    df['price_frag_divergence'] = (df['close'] / df['close'].shift(3) - 1) - (df['volume'] / df['volume'].shift(3) - 1)
    
    # 5. Smart Volatility Flow (3-day window)
    df['mid_price'] = (df['high'] + df['low']) / 2
    df['smart_vol_flow'] = 0.0
    
    for i in range(3, len(df)):
        window_data = df.iloc[i-3:i+1]
        flow = np.sign(np.sum(window_data['amount'] * (window_data['close'] - window_data['mid_price'])))
        df.iloc[i, df.columns.get_loc('smart_vol_flow')] = flow
    
    # 6. Volatility-Fragmentation Momentum
    df['vol_frag_momentum'] = (df['range'] / df['range'].shift(1)) - (df['volume'] / df['volume'].shift(1))
    
    # 7. Intraday volatility position (using open-close relative to range)
    df['intraday_vol_position'] = (df['close'] - df['open']) / df['range']
    
    # 8. Volume entropy approximation (using rolling volume variance)
    df['volume_entropy'] = df['volume'].rolling(window=5).std() / df['volume'].rolling(window=5).mean()
    
    # Regime detection components
    # High volatility regime signal
    vol_ma = df['range'].rolling(window=10).mean()
    df['high_vol_regime'] = (df['range'] > vol_ma * 1.2).astype(int)
    
    # Trending regime (volatility momentum)
    df['vol_momentum'] = df['range'].pct_change(periods=3)
    df['trending_regime'] = (abs(df['vol_momentum']) > 0.1).astype(int)
    
    # Compressed regime (low volatility concentration)
    df['compressed_regime'] = (df['range'] < vol_ma * 0.8).astype(int)
    
    # Composite alpha calculation
    # Regime-adaptive weighting
    df['regime_weight'] = (
        df['high_vol_regime'] * -df['price_impact_frag'] * df['range'] +
        df['trending_regime'] * df['vol_momentum'] * df['volume_frag_momentum'] +
        df['compressed_regime'] * df['intraday_vol_position'] * df['volume_entropy']
    )
    
    # Core fragmentation components
    df['core_fragmentation'] = (
        df['price_frag_divergence'] * 0.3 +
        df['vol_frag_momentum'] * 0.25 +
        df['smart_vol_flow'] * 0.2 +
        df['vol_acceleration'] * 0.25
    )
    
    # Final alpha factor
    alpha = df['regime_weight'] * 0.4 + df['core_fragmentation'] * 0.6
    
    # Normalize and clean
    alpha = (alpha - alpha.rolling(window=20).mean()) / alpha.rolling(window=20).std()
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return alpha
