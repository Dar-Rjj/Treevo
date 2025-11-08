import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Volume-Weighted Momentum factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate returns for momentum components
    data['returns'] = data['close'].pct_change()
    
    # Multi-Timeframe Momentum with Recursive Smoothing
    def exponential_smooth(series, alpha=0.9):
        smoothed = series.copy()
        for i in range(1, len(series)):
            smoothed.iloc[i] = alpha * smoothed.iloc[i-1] + (1 - alpha) * series.iloc[i]
        return smoothed
    
    # Short-term momentum (5-day)
    mom_short = data['close'] / data['close'].shift(5) - 1
    mom_short_smooth = exponential_smooth(mom_short, 0.9)
    
    # Medium-term momentum (20-day)
    mom_medium = data['close'] / data['close'].shift(20) - 1
    mom_medium_smooth = exponential_smooth(mom_medium, 0.9)
    
    # Long-term momentum (60-day)
    mom_long = data['close'] / data['close'].shift(60) - 1
    mom_long_smooth = exponential_smooth(mom_long, 0.9)
    
    # Combined momentum score (weighted average)
    momentum_score = (0.4 * mom_short_smooth + 0.35 * mom_medium_smooth + 0.25 * mom_long_smooth)
    
    # Volume Confirmation
    # Volume trend: recent vs historical average
    volume_ma_20 = data['volume'].rolling(window=20).mean()
    volume_trend = data['volume'] / volume_ma_20 - 1
    
    # Volume acceleration (5-day change in volume trend)
    volume_accel = volume_trend.diff(5)
    
    # Volume-weighted price signals
    # Volume-weighted high-low range
    vwap_hl_range = (data['high'] - data['low']) * data['volume']
    vwap_hl_norm = vwap_hl_range / vwap_hl_range.rolling(window=20).mean()
    
    # Volume-weighted close-to-open gap
    vwap_co_gap = (data['close'] - data['open']) * data['volume']
    vwap_co_norm = vwap_co_gap / vwap_co_gap.rolling(window=20).mean()
    
    # Volume confirmation score
    volume_score = 0.4 * volume_trend + 0.3 * volume_accel + 0.15 * vwap_hl_norm + 0.15 * vwap_co_norm
    
    # Volatility Normalization
    # Daily range
    daily_range = data['high'] - data['low']
    
    # Recent volatility (5-day std of returns)
    vol_5d = data['returns'].rolling(window=5).std()
    
    # Range normalization
    momentum_normalized = momentum_score / (daily_range / data['close'])
    
    # Volatility stability scaling
    vol_stability = vol_5d.rolling(window=20).std()
    vol_stability_adj = 1 / (1 + vol_stability)
    
    # Regime Detection
    # Price trend regime (using 20-day moving average)
    price_ma_20 = data['close'].rolling(window=20).mean()
    price_trend_regime = np.where(data['close'] > price_ma_20, 1, -1)
    
    # Volatility regime (using 20-day volatility vs historical)
    vol_ma_20 = vol_5d.rolling(window=20).mean()
    vol_regime = np.where(vol_5d > vol_ma_20, 1, 0)  # 1 = high vol, 0 = low vol
    
    # Adaptive scaling based on regime
    regime_multiplier = np.where(vol_regime == 1, 0.7, 1.3)  # Reduce in high vol, enhance in low vol
    
    # Combine all components
    raw_factor = momentum_normalized * volume_score * vol_stability_adj
    regime_aware_factor = raw_factor * regime_multiplier * price_trend_regime
    
    # Final factor with additional smoothing
    factor = exponential_smooth(pd.Series(regime_aware_factor, index=data.index), 0.95)
    
    return factor
