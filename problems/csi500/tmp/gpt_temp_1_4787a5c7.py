import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Multi-Timeframe Momentum with Volume Divergence
    Combines volatility-adjusted momentum across multiple timeframes with volume-price divergence
    and adaptive regime-based weighting.
    """
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Multi-timeframe returns
    ret_5d = df['close'].pct_change(5)
    ret_10d = df['close'].pct_change(10)
    ret_20d = df['close'].pct_change(20)
    
    # Rolling volatility for each timeframe (using daily returns)
    vol_5d = returns.rolling(5).std()
    vol_10d = returns.rolling(10).std()
    vol_20d = returns.rolling(20).std()
    
    # Volatility-normalized momentum components
    mom_5d_norm = ret_5d / (vol_5d + 1e-8)
    mom_10d_norm = ret_10d / (vol_10d + 1e-8)
    mom_20d_norm = ret_20d / (vol_20d + 1e-8)
    
    # Volume trends (moving averages)
    vol_ma_5d = df['volume'].rolling(5).mean()
    vol_ma_10d = df['volume'].rolling(10).mean()
    vol_ma_20d = df['volume'].rolling(20).mean()
    
    # Volume trend slopes (using linear regression coefficients)
    def volume_slope(series, window):
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
    
    vol_slope_5d = volume_slope(df['volume'], 5)
    vol_slope_10d = volume_slope(df['volume'], 10)
    vol_slope_20d = volume_slope(df['volume'], 20)
    
    # Price trend slopes
    price_slope_5d = volume_slope(df['close'], 5)
    price_slope_10d = volume_slope(df['close'], 10)
    price_slope_20d = volume_slope(df['close'], 20)
    
    # Volume divergence score (average across timeframes)
    vol_div_5d = np.sign(price_slope_5d) * (price_slope_5d - vol_slope_5d)
    vol_div_10d = np.sign(price_slope_10d) * (price_slope_10d - vol_slope_10d)
    vol_div_20d = np.sign(price_slope_20d) * (price_slope_20d - vol_slope_20d)
    
    vol_divergence = (vol_div_5d + vol_div_10d + vol_div_20d) / 3
    
    # Market regime detection
    recent_vol = returns.rolling(20).std()
    historical_vol_median = returns.expanding().std().rolling(60, min_periods=20).median()
    
    high_vol_regime = recent_vol > historical_vol_median
    
    # Adaptive weighting based on regime
    alpha_scores = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient data
            alpha_scores.iloc[i] = 0
            continue
            
        is_high_vol = high_vol_regime.iloc[i]
        
        if is_high_vol:
            # High volatility regime weights
            mom_weights = [0.2, 0.3, 0.5]  # Higher weight to longer-term
            vol_div_weight = 0.1
        else:
            # Low volatility regime weights
            mom_weights = [0.33, 0.33, 0.34]  # Balanced weights
            vol_div_weight = 0.3
        
        # Weighted momentum average
        momentum_score = (
            mom_weights[0] * mom_5d_norm.iloc[i] +
            mom_weights[1] * mom_10d_norm.iloc[i] +
            mom_weights[2] * mom_20d_norm.iloc[i]
        )
        
        # Final alpha score with interaction term
        vol_div_score = vol_divergence.iloc[i] if not pd.isna(vol_divergence.iloc[i]) else 0
        
        # Interaction term: momentum * volume divergence (enhances signals when aligned)
        interaction = momentum_score * vol_div_score
        
        alpha_scores.iloc[i] = (
            (1 - vol_div_weight) * momentum_score +
            vol_div_weight * vol_div_score +
            0.1 * interaction  # Small interaction term
        )
    
    return alpha_scores
