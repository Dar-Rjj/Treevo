import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Normalized Multi-Timeframe Momentum with Volume Divergence
    
    This factor combines momentum signals across multiple timeframes, normalized by volatility,
    and incorporates volume divergence analysis with regime-adaptive weighting.
    """
    
    df = data.copy()
    
    # Multi-Timeframe Momentum Calculation
    df['momentum_3d'] = df['close'].pct_change(periods=3)
    df['momentum_5d'] = df['close'].pct_change(periods=5)
    df['momentum_10d'] = df['close'].pct_change(periods=10)
    
    # Volatility Normalization
    returns = df['close'].pct_change()
    df['volatility_20d'] = returns.rolling(window=20).std()
    
    # Normalize momentum components by volatility (avoid division by zero)
    volatility_adj = df['volatility_20d'].replace(0, np.nan)
    df['norm_momentum_3d'] = df['momentum_3d'] / volatility_adj
    df['norm_momentum_5d'] = df['momentum_5d'] / volatility_adj
    df['norm_momentum_10d'] = df['momentum_10d'] / volatility_adj
    
    # Volume Divergence Detection
    # Calculate 5-day price and volume trends using linear regression slopes
    def calc_slope(series, window=5):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    df['price_slope_5d'] = calc_slope(df['close'], 5)
    df['volume_slope_5d'] = calc_slope(df['volume'], 5)
    
    # Volume divergence signal
    df['volume_divergence'] = 0
    # Positive divergence: price up but volume down (weak trend)
    pos_div_mask = (df['price_slope_5d'] > 0) & (df['volume_slope_5d'] < 0)
    # Negative divergence: price down but volume up (potential reversal)
    neg_div_mask = (df['price_slope_5d'] < 0) & (df['volume_slope_5d'] > 0)
    
    df.loc[pos_div_mask, 'volume_divergence'] = -1  # Bearish signal
    df.loc[neg_div_mask, 'volume_divergence'] = 1   # Bullish signal
    
    # Regime-Adaptive Weighting
    # Volatility regime classification
    vol_median = df['volatility_20d'].median()
    vol_std = df['volatility_20d'].std()
    
    df['volatility_regime'] = 0  # Normal regime
    high_vol_mask = df['volatility_20d'] > vol_median + 0.5 * vol_std
    low_vol_mask = df['volatility_20d'] < vol_median - 0.5 * vol_std
    
    df.loc[high_vol_mask, 'volatility_regime'] = 1   # High volatility
    df.loc[low_vol_mask, 'volatility_regime'] = -1   # Low volatility
    
    # Dynamic factor combination based on regime
    df['factor'] = 0
    
    # High volatility regime: emphasize volume divergence
    high_vol_factor = (
        0.3 * df['norm_momentum_3d'] +
        0.2 * df['norm_momentum_5d'] +
        0.1 * df['norm_momentum_10d'] +
        0.4 * df['volume_divergence']
    )
    
    # Low volatility regime: emphasize momentum persistence
    low_vol_factor = (
        0.4 * df['norm_momentum_3d'] +
        0.3 * df['norm_momentum_5d'] +
        0.2 * df['norm_momentum_10d'] +
        0.1 * df['volume_divergence']
    )
    
    # Normal regime: balanced approach
    normal_factor = (
        0.35 * df['norm_momentum_3d'] +
        0.25 * df['norm_momentum_5d'] +
        0.15 * df['norm_momentum_10d'] +
        0.25 * df['volume_divergence']
    )
    
    # Apply regime-specific weighting
    df.loc[df['volatility_regime'] == 1, 'factor'] = high_vol_factor
    df.loc[df['volatility_regime'] == -1, 'factor'] = low_vol_factor
    df.loc[df['volatility_regime'] == 0, 'factor'] = normal_factor
    
    # Final factor series
    factor_series = df['factor'].copy()
    
    return factor_series
