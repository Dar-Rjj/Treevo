import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Price Momentum Component
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Volatility Adjustment
    df['daily_range'] = df['high'] - df['low']
    df['avg_range_5d'] = df['daily_range'].rolling(window=5).mean()
    df['avg_range_10d'] = df['daily_range'].rolling(window=10).mean()
    
    df['vol_adj_momentum_5d'] = df['momentum_5d'] / df['avg_range_5d']
    df['vol_adj_momentum_10d'] = df['momentum_10d'] / df['avg_range_10d']
    
    # Volume-Price Divergence Analysis
    def calculate_slope(series, window=5):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    df['volume_slope_5d'] = calculate_slope(df['volume'], 5)
    df['price_slope_5d'] = calculate_slope(df['close'], 5)
    
    # Calculate volume slope threshold (median absolute value)
    volume_slope_threshold = df['volume_slope_5d'].abs().median()
    
    # Signal Strength Classification
    conditions = [
        # Strong divergence
        (np.sign(df['volume_slope_5d']) != np.sign(df['price_slope_5d'])) & 
        (df['volume_slope_5d'].abs() > volume_slope_threshold),
        # No divergence AND short-term momentum stronger
        (np.sign(df['volume_slope_5d']) == np.sign(df['price_slope_5d'])) & 
        (df['vol_adj_momentum_5d'].abs() > df['vol_adj_momentum_10d'].abs()),
        # Medium signal (weak divergence OR no divergence with medium-term momentum stronger)
        (np.sign(df['volume_slope_5d']) != np.sign(df['price_slope_5d'])) & 
        (df['volume_slope_5d'].abs() <= volume_slope_threshold) |
        ((np.sign(df['volume_slope_5d']) == np.sign(df['price_slope_5d'])) & 
         (df['vol_adj_momentum_5d'].abs() <= df['vol_adj_momentum_10d'].abs()))
    ]
    
    choices = ['weak', 'strong', 'medium']
    df['signal_strength'] = np.select(conditions, choices, default='medium')
    
    # Volatility Regime Detection
    df['returns'] = df['close'].pct_change()
    df['hv_20d'] = df['returns'].rolling(window=20).std()
    df['hv_median_60d'] = df['hv_20d'].rolling(window=60).median()
    df['vol_regime'] = np.where(df['hv_20d'] > df['hv_median_60d'], 'high', 'low')
    
    # Base signal selection
    conditions_base = [
        df['signal_strength'] == 'strong',
        df['signal_strength'] == 'medium',
        df['signal_strength'] == 'weak'
    ]
    choices_base = [
        df['vol_adj_momentum_5d'],
        df['vol_adj_momentum_10d'],
        0.3 * df['vol_adj_momentum_5d']
    ]
    df['base_signal'] = np.select(conditions_base, choices_base, default=0)
    
    # Regime-based direction adjustment
    df['alpha'] = np.where(df['vol_regime'] == 'high', -df['base_signal'], df['base_signal'])
    
    return df['alpha']
