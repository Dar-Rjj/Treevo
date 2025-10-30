import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volatility-adjusted momentum, 
    price-volume efficiency, range-based signals with regime detection.
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Volatility-Adjusted Momentum
    # Short-term (5-day) risk-adjusted return
    returns_5d = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    daily_returns = data['close'].pct_change()
    vol_5d = daily_returns.rolling(window=5).std()
    risk_adj_momentum_5d = returns_5d / (vol_5d + 1e-8)
    
    # Medium-term (20-day) risk-adjusted return
    returns_20d = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    vol_20d = daily_returns.rolling(window=20).std()
    risk_adj_momentum_20d = returns_20d / (vol_20d + 1e-8)
    
    # Combined volatility-adjusted momentum
    volatility_adj_momentum = (risk_adj_momentum_5d + risk_adj_momentum_20d) / 2
    
    # 2. Price-Volume Efficiency Factors
    # Daily price efficiency
    daily_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Efficiency momentum
    efficiency_momentum_5d = daily_efficiency / (daily_efficiency.shift(5) + 1e-8) - 1
    efficiency_momentum_20d = daily_efficiency / (daily_efficiency.shift(20) + 1e-8) - 1
    
    # Volume intensity
    volume_intensity = data['amount'] / (data['high'] - data['low'] + 1e-8)
    
    # Efficiency-volume correlation (20-day)
    efficiency_volume_corr = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        window_data = data.iloc[i-19:i+1]
        corr = np.corrcoef(window_data.index.map(daily_efficiency.get), 
                          window_data.index.map(volume_intensity.get))[0,1]
        efficiency_volume_corr.iloc[i] = corr if not np.isnan(corr) else 0
    
    # Volume-adjusted efficiency signal
    volume_adj_efficiency = efficiency_momentum_20d * efficiency_volume_corr
    
    # 3. Range-Based Volatility Signals
    # Normalized daily range
    normalized_range = (data['high'] - data['low']) / (data['close'] + 1e-8)
    
    # Range momentum
    range_momentum_5d = normalized_range / (normalized_range.shift(5) + 1e-8) - 1
    range_momentum_20d = normalized_range / (normalized_range.shift(20) + 1e-8) - 1
    
    # Volume momentum
    volume_momentum_20d = data['volume'] / (data['volume'].shift(20) + 1e-8) - 1
    
    # Range-volume divergence
    range_volume_divergence = range_momentum_20d - volume_momentum_20d
    
    # 4. Volatility Regime Detection
    vol_60d = daily_returns.rolling(window=60).std()
    regime_high = vol_20d > (1.5 * vol_60d)
    regime_low = vol_20d < (0.67 * vol_60d)
    regime_normal = ~regime_high & ~regime_low
    
    # 5. Regime-Adaptive Factor Integration
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(60, len(data)):
        if regime_high.iloc[i]:
            # High volatility regime: Range signals primary, momentum secondary
            alpha_signal.iloc[i] = (0.5 * range_volume_divergence.iloc[i] + 
                                   0.3 * volatility_adj_momentum.iloc[i] + 
                                   0.2 * volume_adj_efficiency.iloc[i])
        elif regime_low.iloc[i]:
            # Low volatility regime: Efficiency primary, volume secondary
            alpha_signal.iloc[i] = (0.5 * volume_adj_efficiency.iloc[i] + 
                                   0.3 * volatility_adj_momentum.iloc[i] + 
                                   0.2 * range_volume_divergence.iloc[i])
        else:
            # Normal volatility regime: Balanced approach
            alpha_signal.iloc[i] = (volatility_adj_momentum.iloc[i] + 
                                   volume_adj_efficiency.iloc[i] + 
                                   range_volume_divergence.iloc[i]) / 3
    
    # Clean and normalize the final signal
    alpha_signal = alpha_signal.replace([np.inf, -np.inf], np.nan)
    alpha_signal = alpha_signal.fillna(method='ffill')
    
    return alpha_signal
