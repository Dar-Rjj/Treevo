import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Confirmed Momentum with Volatility Regimes alpha factor
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Alpha factor values indexed by date
    """
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate momentum components
    df['M3'] = df['close'] / df['close'].shift(3) - 1
    df['M8'] = df['close'] / df['close'].shift(8) - 1
    df['M21'] = df['close'] / df['close'].shift(21) - 1
    
    # Volatility regime calculation
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['rolling_vol'] = df['daily_range'].rolling(window=5).std()
    
    # Calculate volatility percentiles
    vol_percentiles = df['rolling_vol'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == 20 else np.nan, 
        raw=False
    )
    
    # Initialize volatility regime
    df['vol_regime'] = 'normal'
    df.loc[vol_percentiles > 80, 'vol_regime'] = 'high'
    df.loc[vol_percentiles < 20, 'vol_regime'] = 'low'
    
    # Apply regime persistence (minimum 3 days)
    regime_changes = df['vol_regime'] != df['vol_regime'].shift(1)
    short_regimes = regime_changes.rolling(window=3).sum() > 1
    df.loc[short_regimes, 'vol_regime'] = df['vol_regime'].shift(1)
    
    # Volume confirmation calculation
    def volume_percentile(x):
        if len(x) == 20:
            current_vol = x[-1]
            return (pd.Series(x).rank(pct=True).iloc[-1] * 100)
        return np.nan
    
    df['volume_pct'] = df['volume'].rolling(window=20).apply(volume_percentile, raw=False)
    
    # Volume regime classification
    df['volume_regime'] = 'normal'
    df.loc[df['volume_pct'] > 80, 'volume_regime'] = 'high'
    df.loc[df['volume_pct'] < 20, 'volume_regime'] = 'low'
    
    # Momentum consistency check
    def momentum_consistency(row):
        if pd.isna(row['M3']) or pd.isna(row['M8']) or pd.isna(row['M21']):
            return 'mixed'
        if row['M3'] > 0 and row['M8'] > 0 and row['M21'] > 0:
            return 'bullish'
        elif row['M3'] < 0 and row['M8'] < 0 and row['M21'] < 0:
            return 'bearish'
        else:
            return 'mixed'
    
    df['momentum_type'] = df.apply(momentum_consistency, axis=1)
    
    # Calculate weighted momentum score
    def weighted_momentum(row):
        if row['momentum_type'] in ['bullish', 'bearish']:
            # Strong signal: average all three
            return (row['M3'] + row['M8'] + row['M21']) / 3
        else:
            # Mixed signal: weighted towards shorter-term
            return row['M3'] * 0.5 + row['M8'] * 0.3 + row['M21'] * 0.2
    
    df['base_momentum'] = df.apply(weighted_momentum, axis=1)
    
    # Regime-aware signal adjustment
    def regime_adjustment(row):
        if pd.isna(row['base_momentum']) or pd.isna(row['vol_regime']):
            return np.nan
            
        if row['vol_regime'] == 'high':
            # High volatility: reduce magnitude, focus on medium-term
            adjusted = row['M8'] * 0.5 if not pd.isna(row['M8']) else row['base_momentum'] * 0.5
        elif row['vol_regime'] == 'low':
            # Low volatility: enhance magnitude, focus on long-term
            adjusted = row['M21'] * 1.25 if not pd.isna(row['M21']) else row['base_momentum'] * 1.25
        else:
            # Normal volatility: use full weighted momentum
            adjusted = row['base_momentum']
            
        return adjusted
    
    df['regime_adjusted'] = df.apply(regime_adjustment, axis=1)
    
    # Volume multiplier
    def volume_multiplier(row):
        if pd.isna(row['volume_regime']) or pd.isna(row['momentum_type']):
            return 1.0
            
        if row['volume_regime'] == 'high':
            if (row['momentum_type'] == 'bullish' and row['regime_adjusted'] > 0) or \
               (row['momentum_type'] == 'bearish' and row['regime_adjusted'] < 0):
                return 1.5  # Strong confirmation
            else:
                return 0.5  # Misaligned
        elif row['volume_regime'] == 'low':
            return 0.5  # Weak signal
        else:
            return 1.0  # Normal
    
    df['volume_mult'] = df.apply(volume_multiplier, axis=1)
    
    # Final alpha calculation
    alpha = df['regime_adjusted'] * df['volume_mult']
    
    return alpha
