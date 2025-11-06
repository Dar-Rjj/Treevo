import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Calculate 20-day rolling volatility (standard deviation of returns)
    data['volatility_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Calculate 20-day rolling average volume
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Cross-sectional volatility classification
    # For each day, classify as high/low volatility based on cross-sectional median
    daily_volatility_median = data.groupby(data.index)['volatility_20d'].transform('median')
    data['volatility_regime'] = np.where(data['volatility_20d'] > daily_volatility_median, 'high', 'low')
    
    # Volume divergence signals
    # High volatility regime: volume surge with muted price movement
    high_vol_condition = (
        (data['volatility_regime'] == 'high') &
        (data['volume'] > 2 * data['volume_20d_avg']) &
        (abs(data['returns']) < data['volatility_20d'] / 2)
    )
    data['high_vol_signal'] = np.where(high_vol_condition, -1, 0)
    
    # Low volatility regime: sustained volume momentum
    # Calculate consecutive days with volume above 20-day average
    volume_above_avg = data['volume'] > data['volume_20d_avg']
    data['volume_streak'] = volume_above_avg.groupby(volume_above_avg.index).cumsum() - volume_above_avg.groupby(volume_above_avg.index).cumsum().where(~volume_above_avg).ffill().fillna(0)
    
    low_vol_condition = (
        (data['volatility_regime'] == 'low') &
        (data['volume_streak'] >= 3)
    )
    data['low_vol_signal'] = np.where(low_vol_condition, 1, 0)
    
    # Combine signals
    data['raw_signal'] = data['high_vol_signal'] + data['low_vol_signal']
    
    # Create sector groups (using volatility regime as proxy for sector grouping)
    # In practice, you would use actual sector classification
    data['sector_group'] = data['volatility_regime']
    
    # Sector-relative ranking
    data['sector_rank'] = data.groupby(['sector_group', data.index])['raw_signal'].transform(
        lambda x: x.rank(pct=True)
    )
    
    # Dynamic signal weighting based on recent 5-day performance
    # Calculate forward returns for performance evaluation (shifted to avoid lookahead)
    data['fwd_returns_5d'] = data['close'].pct_change(5).shift(-5)
    
    # Calculate signal performance over rolling 20-day window
    data['signal_perf'] = data.groupby('sector_group')['sector_rank'].transform(
        lambda x: x.rolling(window=20, min_periods=10).corr(data['fwd_returns_5d'])
    )
    
    # Apply dynamic weighting - higher weight for historically effective signals
    data['weighted_signal'] = data['sector_rank'] * (1 + data['signal_perf'].fillna(0))
    
    # Final factor value (normalized)
    factor = data['weighted_signal']
    
    return factor
