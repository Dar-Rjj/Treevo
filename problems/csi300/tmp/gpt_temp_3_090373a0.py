import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate an adaptive alpha factor combining volatility-normalized momentum,
    volume-price divergence, and dynamic regime classification.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Enhanced Volatility Measurement
    # True Range Calculation
    high_low = data['high'] - data['low']
    high_prev_close = abs(data['high'] - data['close'].shift(1))
    low_prev_close = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Volatility Normalization
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    avg_true_range_5d = true_range.rolling(window=5).mean()
    volatility_adjusted_momentum = momentum_5d / (avg_true_range_5d + 1e-8)
    
    # Multi-timeframe Volatility
    short_term_vol = true_range.rolling(window=5).mean()
    medium_term_vol = true_range.rolling(window=20).mean()
    volatility_ratio = short_term_vol / (medium_term_vol + 1e-8)
    
    # Volume-Price Divergence Detection
    # Volume Trend Analysis
    def volume_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        return stats.linregress(x, series.values)[0]
    
    volume_slope_5d = data['volume'].rolling(window=5).apply(volume_slope, raw=False)
    volume_acceleration = volume_slope_5d - volume_slope_5d.shift(5)
    volume_momentum = data['volume'] / data['volume'].shift(5) - 1
    
    # Price-Volume Alignment
    price_direction = np.sign(data['close'] - data['close'].shift(1))
    volume_direction = np.sign(data['volume'] - data['volume'].shift(1))
    alignment_score = price_direction * volume_direction
    
    # Divergence Signals
    price_down = (data['close'] < data['close'].shift(1))
    volume_down = (data['volume'] < data['volume'].shift(1))
    alignment_improving = (alignment_score > alignment_score.shift(1))
    
    price_up = (data['close'] > data['close'].shift(1))
    alignment_worsening = (alignment_score < alignment_score.shift(1))
    
    bullish_divergence = price_down & volume_down & alignment_improving
    bearish_divergence = price_up & volume_down & alignment_worsening
    
    divergence_strength = abs(momentum_5d) * abs(volume_momentum)
    
    # Dynamic Regime Classification
    # Volatility Regime
    returns = data['close'].pct_change()
    historical_volatility = returns.rolling(window=20).std()
    vol_threshold = historical_volatility.rolling(window=60).median()
    high_volatility = historical_volatility > vol_threshold
    low_volatility = historical_volatility <= vol_threshold
    
    # Trend Regime
    def price_slope(series):
        if len(series) < 20:
            return np.nan
        x = np.arange(len(series))
        return stats.linregress(x, series.values)[0]
    
    trend_slope = data['close'].rolling(window=20).apply(price_slope, raw=False)
    trend_strength = abs(trend_slope) / (data['close'].rolling(window=20).std() + 1e-8)
    trend_threshold = trend_strength.rolling(window=60).median()
    strong_trend = trend_strength > trend_threshold
    weak_trend = trend_strength <= trend_threshold
    
    # Market State
    trending_market = strong_trend & low_volatility
    mean_reverting_market = weak_trend & high_volatility
    transition_market = ~(trending_market | mean_reverting_market)
    
    # Adaptive Signal Integration
    # Base Alpha Components
    base_signal = volatility_adjusted_momentum
    
    # Volume Confirmation Multiplier
    strong_confirmation = (alignment_score > 0) & (volume_momentum > 0)
    weak_confirmation = (alignment_score > 0) & (volume_momentum <= 0)
    divergence_confirmation = (alignment_score <= 0)
    
    volume_multiplier = pd.Series(1.0, index=data.index)
    volume_multiplier[strong_confirmation] = 1.5
    volume_multiplier[weak_confirmation] = 1.0
    volume_multiplier[divergence_confirmation] = 0.5
    
    # Regime-Adaptive Weights
    regime_weight = pd.Series(1.0, index=data.index)
    regime_weight[trending_market] = 1.2  # Emphasize momentum
    regime_weight[mean_reverting_market] = -1.2  # Emphasize reversal
    regime_weight[transition_market] = 0.8  # Reduced signal strength
    
    # Final Alpha Construction
    volume_adjusted_signal = base_signal * volume_multiplier
    regime_adjusted_signal = volume_adjusted_signal * regime_weight
    final_alpha = regime_adjusted_signal * trend_strength
    
    return final_alpha
