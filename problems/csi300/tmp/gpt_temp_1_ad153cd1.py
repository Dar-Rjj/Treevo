import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum with Volume-Price Divergence alpha factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate returns for momentum
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['ret_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Calculate True Range and ATR
    data['prev_close'] = data['close'].shift(1)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    data['atr_5d'] = data['tr'].rolling(window=5).mean()
    data['atr_10d'] = data['tr'].rolling(window=10).mean()
    data['atr_20d'] = data['tr'].rolling(window=20).mean()
    data['atr_60d'] = data['tr'].rolling(window=60).mean()
    
    # Volatility-normalized momentum
    data['norm_mom_5d'] = data['ret_5d'] / data['atr_5d']
    data['norm_mom_10d'] = data['ret_10d'] / data['atr_10d']
    
    # Volume and price slopes
    def calc_slope(series, window):
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                if len(y) == window:
                    slope, _, _, _, _ = linregress(x, y)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['vol_slope_5d'] = calc_slope(data['volume'], 5)
    data['vol_slope_10d'] = calc_slope(data['volume'], 10)
    data['price_slope_5d'] = calc_slope(data['close'], 5)
    data['price_slope_10d'] = calc_slope(data['close'], 10)
    data['price_slope_20d'] = calc_slope(data['close'], 20)
    
    # Volume-price divergence
    data['div_5d'] = (np.sign(data['price_slope_5d']) != np.sign(data['vol_slope_5d'])).astype(int)
    data['div_10d'] = (np.sign(data['price_slope_10d']) != np.sign(data['vol_slope_10d'])).astype(int)
    data['div_score'] = (data['div_5d'] + data['div_10d']) / 2
    
    # Market regime detection
    data['vol_regime'] = (data['atr_20d'] > data['atr_60d']).astype(int)  # 1=high, 0=low
    
    # Trend regime threshold (0.001 represents 0.1% daily trend)
    trend_threshold = 0.001 * 20  # Adjusted for 20-day period
    data['trend_regime'] = (abs(data['price_slope_20d']) > trend_threshold).astype(int)  # 1=trending, 0=ranging
    
    # Adaptive signal integration
    alpha_values = []
    
    for i in range(len(data)):
        if i < 60:  # Need enough data for calculations
            alpha_values.append(np.nan)
            continue
            
        row = data.iloc[i]
        
        # Base momentum selection
        if row['trend_regime'] == 1:  # Trending regime
            base_momentum = row['norm_mom_10d']  # Emphasize medium-term
        else:  # Ranging regime
            base_momentum = row['norm_mom_5d']   # Emphasize short-term
        
        # Volume divergence adjustment
        if row['div_score'] > 0:  # Divergence detected
            divergence_factor = 0.5  # Negative multiplier
        else:  # Convergence
            divergence_factor = 1.2  # Positive multiplier
        
        # Volatility regime weighting
        if row['vol_regime'] == 1:  # High volatility
            volatility_scaling = 0.7  # Reduce signal magnitude
        else:  # Low volatility
            volatility_scaling = 1.0  # Maintain signal magnitude
        
        # Final alpha calculation
        alpha = base_momentum * divergence_factor * volatility_scaling
        alpha_values.append(alpha)
    
    result = pd.Series(alpha_values, index=data.index, name='regime_adaptive_momentum')
    return result
