import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility Regime with Volume-Price Divergence factor
    """
    data = df.copy()
    
    # Volatility Regime Identification
    # Historical volatility calculation
    data['returns'] = data['close'].pct_change()
    data['vol_short'] = data['returns'].rolling(window=5).std()
    data['vol_medium'] = data['returns'].rolling(window=20).std()
    data['vol_ratio'] = data['vol_short'] / data['vol_medium']
    
    # Regime classification
    conditions = [
        data['vol_ratio'] > 1.2,
        (data['vol_ratio'] >= 0.8) & (data['vol_ratio'] <= 1.2),
        data['vol_ratio'] < 0.8
    ]
    choices = ['high', 'normal', 'low']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Volume-Price Divergence Analysis
    # Price trend calculation
    def calc_slope(series, window):
        slopes = np.full(len(series), np.nan)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) == window and not np.all(np.isnan(y)):
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes[i] = slope
        return slopes
    
    data['price_slope_10'] = calc_slope(data['close'], 10)
    data['price_slope_5'] = calc_slope(data['close'], 5)
    data['price_accel'] = data['price_slope_5'] - data['price_slope_10']
    
    # Volume trend calculation
    data['volume_slope_10'] = calc_slope(data['volume'], 10)
    data['volume_slope_5'] = calc_slope(data['volume'], 5)
    data['volume_accel'] = data['volume_slope_5'] - data['volume_slope_10']
    
    # Divergence detection
    data['price_trend'] = np.sign(data['price_slope_10'])
    data['volume_trend'] = np.sign(data['volume_slope_10'])
    
    conditions_div = [
        (data['price_trend'] < 0) & (data['volume_trend'] > 0),  # Positive divergence
        (data['price_trend'] > 0) & (data['volume_trend'] < 0),  # Negative divergence
        data['price_trend'] * data['volume_trend'] > 0           # Convergence
    ]
    choices_div = [1, -1, 0]
    data['divergence'] = np.select(conditions_div, choices_div, default=0)
    
    # Recent divergence strength (3-day sum)
    data['recent_divergence'] = data['divergence'].rolling(window=3).sum()
    
    # Regime-Adaptive Signal Combination
    factor_values = np.zeros(len(data))
    
    for i in range(len(data)):
        if pd.isna(data['vol_regime'].iloc[i]) or pd.isna(data['price_accel'].iloc[i]):
            continue
            
        regime = data['vol_regime'].iloc[i]
        price_accel = data['price_accel'].iloc[i]
        recent_div = data['recent_divergence'].iloc[i] if not pd.isna(data['recent_divergence'].iloc[i]) else 0
        divergence = data['divergence'].iloc[i] if not pd.isna(data['divergence'].iloc[i]) else 0
        
        if regime == 'high':
            # Focus on recent divergences and weight volume signals more heavily
            signal = 0.7 * recent_div + 0.3 * price_accel
            
        elif regime == 'normal':
            # Balanced price-volume signals with medium-term trends
            price_signal = data['price_slope_10'].iloc[i] if not pd.isna(data['price_slope_10'].iloc[i]) else 0
            volume_signal = data['volume_slope_10'].iloc[i] if not pd.isna(data['volume_slope_10'].iloc[i]) else 0
            signal = 0.5 * price_signal + 0.3 * volume_signal + 0.2 * divergence
            
        else:  # low volatility
            # Emphasize price acceleration and require stronger divergence confirmation
            div_strength = abs(divergence) * abs(data['volume_accel'].iloc[i]) if not pd.isna(data['volume_accel'].iloc[i]) else 0
            signal = 0.8 * price_accel + 0.2 * div_strength * np.sign(divergence)
        
        factor_values[i] = signal
    
    # Normalize the factor
    factor_series = pd.Series(factor_values, index=data.index)
    factor_series = (factor_series - factor_series.rolling(window=20, min_periods=10).mean()) / factor_series.rolling(window=20, min_periods=10).std()
    
    return factor_series
