import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, fixed_lookback=60, fixed_window=30, min_vol_window=10, max_vol_window=90):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=fixed_window).std().mean(axis=1)
    
    # Adjust Window Size Based on Volatility
    def adaptive_window(std):
        if std > df['volatility'].quantile(0.75):
            return min_vol_window
        elif std < df['volatility'].quantile(0.25):
            return max_vol_window
        else:
            return fixed_window
        
    df['window_size'] = df['volatility'].apply(adaptive_window)
    
    # Rolling Statistics
    def rolling_stats(series, window_series):
        result = series.rolling(window=window_series, min_periods=1).agg(['mean', 'std'])
        return result['mean'], result['std']
    
    df['vwc_mean'], df['vwc_std'] = rolling_stats(df['volume_weighted_return'], df['window_size'])
    
    # Momentum Factor
    df['price_momentum'] = df['close'].pct_change(fixed_lookback)
    
    # Integrate with Volume Weighted Close-to-Open Return
    df['integrated_factor'] = (df['vwc_mean'] / df['vwc_std']) * df['price_momentum']
    
    # Cross-Asset Correlation (assuming major_index is a column in df or a separate Series)
    if 'major_index' in df.columns:
        major_index = df['major_index']
    else:
        major_index = pd.Series(index=df.index, data=np.random.randn(len(df)))  # Placeholder for actual index data
    
    df['asset_index_corr'] = df['close'].rolling(window=fixed_window).corr(major_index)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['integrated_factor'] * df['asset_index_corr']
    
    return df['alpha_factor'].dropna()

# Example usage:
# alpha_factor = heuristics_v2(df)
