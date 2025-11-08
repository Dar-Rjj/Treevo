import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Volatility-Normalized Multi-Timeframe Momentum with Volume Divergence
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate returns for volatility estimation
    returns = data['close'].pct_change()
    
    # Multi-Timeframe Momentum Calculation
    # Short-term momentum (3-day)
    mom_short = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    vol_short = returns.rolling(window=10, min_periods=5).std()
    norm_mom_short = mom_short / (vol_short + 1e-8)
    
    # Medium-term momentum (10-day)
    mom_medium = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    vol_medium = returns.rolling(window=20, min_periods=10).std()
    norm_mom_medium = mom_medium / (vol_medium + 1e-8)
    
    # Long-term momentum (20-day)
    mom_long = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    vol_long = returns.rolling(window=40, min_periods=20).std()
    norm_mom_long = mom_long / (vol_long + 1e-8)
    
    # Volume Divergence Analysis
    # Price-Volume Trend Comparison
    def calculate_trend(series, window):
        trends = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope = linregress(x, y).slope
                trends.iloc[i] = slope
        return trends / (series.rolling(window).std() + 1e-8)
    
    price_trend = calculate_trend(data['close'], 5)
    volume_trend = calculate_trend(data['volume'], 5)
    volume_divergence = price_trend - volume_trend
    
    # Volume Acceleration Signal
    vol_3day_rate = data['volume'].pct_change(3)
    vol_5day_rate = data['volume'].pct_change(5)
    volume_acceleration = vol_3day_rate - vol_5day_rate
    
    # Regime Detection and Weighting
    vol_20day = returns.rolling(window=20, min_periods=10).std()
    vol_60day = returns.rolling(window=60, min_periods=30).std()
    vol_ratio = vol_20day / vol_60day
    
    # Regime classification
    high_vol_regime = (vol_ratio > 1.0).astype(int)
    low_vol_regime = (vol_ratio < 1.0).astype(int)
    transition_regime = ((vol_ratio >= 0.9) & (vol_ratio <= 1.1)).astype(int)
    
    # Adaptive Momentum Weighting
    def get_regime_weights(high_vol, low_vol, transition):
        weights_short = high_vol * 0.6 + low_vol * 0.2 + transition * 0.33
        weights_medium = high_vol * 0.3 + low_vol * 0.3 + transition * 0.33
        weights_long = high_vol * 0.1 + low_vol * 0.5 + transition * 0.33
        return weights_short, weights_medium, weights_long
    
    w_short, w_medium, w_long = get_regime_weights(high_vol_regime, low_vol_regime, transition_regime)
    
    # Timeframe alignment bonus
    alignment_bonus = ((norm_mom_short > 0) & (norm_mom_medium > 0) & (norm_mom_long > 0)).astype(int) - \
                     ((norm_mom_short < 0) & (norm_mom_medium < 0) & (norm_mom_long < 0)).astype(int)
    
    # Weighted Momentum Score
    weighted_momentum = (w_short * norm_mom_short + 
                        w_medium * norm_mom_medium + 
                        w_long * norm_mom_long + 
                        alignment_bonus * 0.1)
    
    # Volume Confirmation Adjustment
    volume_confirmation = np.zeros_like(weighted_momentum)
    
    # Positive divergence enhances bullish momentum
    bullish_condition = (weighted_momentum > 0) & (volume_divergence > 0)
    volume_confirmation[bullish_condition] = volume_divergence[bullish_condition] * 0.5
    
    # Negative divergence enhances bearish momentum
    bearish_condition = (weighted_momentum < 0) & (volume_divergence < 0)
    volume_confirmation[bearish_condition] = volume_divergence[bearish_condition] * 0.5
    
    # Volume acceleration adjustment
    volume_confirmation += volume_acceleration * 0.2
    
    # Final Alpha Signal
    alpha_signal = weighted_momentum + volume_confirmation
    
    # Regime-adaptive scaling
    regime_scaling = high_vol_regime * 0.8 + low_vol_regime * 1.2 + transition_regime * 1.0
    final_alpha = alpha_signal * regime_scaling
    
    return final_alpha
