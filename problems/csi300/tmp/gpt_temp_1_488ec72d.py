import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Association Pattern Recognition alpha factor combining multiple market microstructure signals
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Price-Volume Divergence Analysis
    # Intraday Volume-Price Correlation (5-day rolling correlation of returns vs volume changes)
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['price_volume_corr'] = data['returns'].rolling(window=5).corr(data['volume_change'])
    
    # Multi-day Accumulation/Distribution Patterns
    # Volume clustering on up/down days (3-day volume ratio on up vs down days)
    data['price_up'] = (data['close'] > data['open']).astype(int)
    data['up_day_volume'] = data['volume'].where(data['price_up'] == 1).rolling(window=3).mean()
    data['down_day_volume'] = data['volume'].where(data['price_up'] == 0).rolling(window=3).mean()
    data['accumulation_ratio'] = data['up_day_volume'] / data['down_day_volume']
    
    # Abnormal Volume Spike Detection
    data['volume_ratio_20d'] = data['volume'] / data['volume'].rolling(window=20).median()
    
    # Market Microstructure Anomalies (using amount as proxy for trade size)
    # Trade Size Distribution Skew
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['trade_size_skew'] = data['avg_trade_size'].rolling(window=10).apply(
        lambda x: (x.mean() - x.median()) / x.std() if x.std() > 0 else 0
    )
    
    # Cross-Sectional Relative Behavior (using rolling percentiles for relative positioning)
    # Volume Leadership Identification
    data['volume_rank_20d'] = data['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    # Temporal Pattern Recognition
    # Multi-period Reversal Detection (3-day vs 10-day momentum divergence)
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['reversal_signal'] = data['momentum_3d'] - data['momentum_10d']
    
    # Time-of-Day Effect Quantification (using high-low range as proxy)
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['range_momentum'] = data['intraday_range'].pct_change(3)
    
    # Combine signals with appropriate weights
    factors = []
    
    # Price-Volume Divergence component
    pv_divergence = (
        0.3 * data['price_volume_corr'].fillna(0) +
        0.4 * np.log(data['accumulation_ratio'].fillna(1)) +
        0.3 * np.log(data['volume_ratio_20d'].fillna(1))
    )
    factors.append(pv_divergence)
    
    # Microstructure component
    microstructure = data['trade_size_skew'].fillna(0) * 0.5
    factors.append(microstructure)
    
    # Cross-sectional component
    cross_sectional = data['volume_rank_20d'].fillna(0.5) * 0.3
    factors.append(cross_sectional)
    
    # Temporal component
    temporal = (
        0.6 * data['reversal_signal'].fillna(0) +
        0.4 * data['range_momentum'].fillna(0)
    )
    factors.append(temporal)
    
    # Final alpha factor - normalized combination
    alpha = sum(factors) / len(factors)
    
    # Z-score normalization
    alpha_normalized = (alpha - alpha.rolling(window=20).mean()) / alpha.rolling(window=20).std()
    
    return alpha_normalized.fillna(0)
