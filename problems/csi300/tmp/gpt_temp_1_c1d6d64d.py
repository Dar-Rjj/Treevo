import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Momentum Divergence factor combining volatility state classification,
    momentum divergence patterns, and volume-volatility coupling across multiple time scales.
    """
    
    # Calculate returns and volatility measures
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['volatility_5min'] = df['returns'].rolling(window=12, min_periods=8).std()  # 1-hour volatility
    
    # Volatility State Classification
    df['volatility_quintile'] = df['volatility_5min'].rolling(window=252, min_periods=126).apply(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop').iloc[-1] if len(x.dropna()) >= 126 else np.nan, 
        raw=False
    )
    
    # Regime persistence using rolling volatility stability
    df['volatility_persistence'] = df['volatility_5min'].rolling(window=21, min_periods=10).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x.dropna()) >= 10 else np.nan,
        raw=False
    )
    
    # Momentum measures across different timeframes
    df['momentum_short'] = df['close'] / df['close'].shift(5) - 1  # 1-week momentum
    df['momentum_medium'] = df['close'] / df['close'].shift(21) - 1  # 1-month momentum
    df['momentum_long'] = df['close'] / df['close'].shift(63) - 1  # 3-month momentum
    
    # Volume-volatility coupling
    df['volume_efficiency'] = (df['close'] - df['open']).abs() / (df['volume'] + 1e-8)
    df['volume_persistence'] = df['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: len(x[x > x.median()]) / len(x) if len(x.dropna()) >= 5 else np.nan,
        raw=False
    )
    
    # Cross-sectional momentum ranking within volatility regimes
    def regime_momentum_rank(group):
        if len(group.dropna()) >= 5:
            return group.rank(pct=True)
        return pd.Series(np.nan, index=group.index)
    
    df['momentum_regime_rank'] = df.groupby('volatility_quintile')['momentum_medium'].transform(regime_momentum_rank)
    
    # Volatility transition signals
    df['volatility_regime_change'] = df['volatility_quintile'].diff().abs()
    df['volatility_acceleration'] = df['volatility_5min'].pct_change(periods=5)
    
    # Multi-scale momentum divergence
    df['momentum_divergence'] = (
        (df['momentum_short'] - df['momentum_medium'].rolling(window=5, min_periods=3).mean()) * 
        (df['momentum_medium'] - df['momentum_long'].rolling(window=5, min_periods=3).mean())
    )
    
    # Volume breakdown as volatility regime change precursor
    df['volume_volatility_coupling'] = (
        df['volume_efficiency'].rolling(window=10, min_periods=5).std() * 
        df['volatility_persistence']
    )
    
    # Regime-conditional momentum scoring
    df['regime_momentum_score'] = (
        df['momentum_regime_rank'] * 
        (1 + df['volatility_regime_change']) * 
        np.sign(df['momentum_divergence'])
    )
    
    # Alpha factor construction
    alpha_factor = (
        df['regime_momentum_score'] * 
        (1 - df['volume_volatility_coupling'].rolling(window=5, min_periods=3).mean()) * 
        np.tanh(df['volatility_acceleration']) * 
        df['volume_persistence']
    )
    
    # Final smoothing and normalization
    alpha_factor = alpha_factor.rolling(window=5, min_periods=3).mean()
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=252, min_periods=126).mean()) / \
                   (alpha_factor.rolling(window=252, min_periods=126).std() + 1e-8)
    
    return alpha_factor
