import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate momentum components
    df = df.copy()
    
    # Price momentum
    df['price_momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['price_momentum_8d'] = df['close'] / df['close'].shift(8) - 1
    
    # Volume momentum
    df['volume_momentum_3d'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_momentum_8d'] = df['volume'] / df['volume'].shift(8) - 1
    
    # Assess momentum alignment
    df['short_term_aligned'] = ((df['price_momentum_3d'] > 0) & (df['volume_momentum_3d'] > 0)) | \
                              ((df['price_momentum_3d'] < 0) & (df['volume_momentum_3d'] < 0))
    df['medium_term_aligned'] = ((df['price_momentum_8d'] > 0) & (df['volume_momentum_8d'] > 0)) | \
                               ((df['price_momentum_8d'] < 0) & (df['volume_momentum_8d'] < 0))
    
    df['alignment_score'] = df['short_term_aligned'].astype(int) + df['medium_term_aligned'].astype(int)
    
    # Calculate adaptive volatility
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_5d'] = df['daily_range'].rolling(window=5).mean()
    
    # Scale volatility by recent volume trend
    df['volume_trend_5d'] = df['volume'].rolling(window=5).apply(
        lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True
    )
    df['scaled_volatility'] = df['volatility_5d'] * (1 + np.abs(df['volume_trend_5d']))
    
    # Generate combined factor
    df['momentum_alignment_factor'] = df['alignment_score'] * df['scaled_volatility'] * np.sign(df['price_momentum_3d'])
    
    # Cross-sectional ranking (assuming sector data is available in df['sector'])
    if 'sector' in df.columns:
        df['sector_rank'] = df.groupby('sector')['momentum_alignment_factor'].rolling(
            window=20, min_periods=10
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False).reset_index(level=0, drop=True)
        result = df['sector_rank']
    else:
        # If no sector data, use simple cross-sectional rank
        result = df['momentum_alignment_factor'].rolling(
            window=20, min_periods=10
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    return result
