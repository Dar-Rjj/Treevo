import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Measure Price Efficiency
    # Daily price efficiency
    daily_efficiency = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    daily_efficiency = daily_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # 5-day efficiency ratio
    numerator_5d = abs(data['close'] - data['close'].shift(1)).rolling(window=5, min_periods=3).sum()
    denominator_5d = (data['high'] - data['low']).rolling(window=5, min_periods=3).sum()
    efficiency_5d = numerator_5d / denominator_5d
    efficiency_5d = efficiency_5d.replace([np.inf, -np.inf], np.nan)
    
    # Efficiency decay (current vs 5-day MA)
    efficiency_ma_5d = daily_efficiency.rolling(window=5, min_periods=3).mean()
    efficiency_decay = daily_efficiency - efficiency_ma_5d
    
    # 2. Analyze Volume-Price Relationship
    # Volume-weighted efficiency
    volume_weighted_efficiency = daily_efficiency * np.log(data['volume'].replace(0, 1))
    
    # Volume persistence (5-day autocorrelation)
    volume_persistence = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=True
    )
    
    # Volume clustering
    volume_median_20d = data['volume'].rolling(window=20, min_periods=10).median()
    volume_clustering = data['volume'] / volume_median_20d
    
    # 3. Identify Market Regimes
    # Trend regime (10-day vs 50-day MA crossover)
    ma_10d = data['close'].rolling(window=10, min_periods=5).mean()
    ma_50d = data['close'].rolling(window=50, min_periods=25).mean()
    trend_regime = (ma_10d > ma_50d).astype(int)  # 1 for uptrend, 0 for downtrend
    
    # Volatility state (20-day ATR vs 60-day median)
    tr = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hc': abs(data['high'] - data['close'].shift(1)),
        'lc': abs(data['low'] - data['close'].shift(1))
    }).max(axis=1)
    atr_20d = tr.rolling(window=20, min_periods=10).mean()
    atr_median_60d = atr_20d.rolling(window=60, min_periods=30).median()
    volatility_state = atr_20d / atr_median_60d
    
    # Liquidity regime (volume clustering + spread proxy)
    spread_proxy = (data['high'] - data['low']) / data['close']
    liquidity_regime = volume_clustering * (1 - spread_proxy.rolling(window=10, min_periods=5).mean())
    
    # 4. Evaluate Efficiency-Persistence Dynamics
    # High efficiency + declining volume → strong momentum signal
    volume_change = data['volume'].pct_change(periods=1)
    momentum_signal = (efficiency_5d > efficiency_5d.rolling(window=20, min_periods=10).median()) & \
                     (volume_change < 0)
    
    # Low efficiency + volume clustering → reversal potential
    reversal_signal = (efficiency_5d < efficiency_5d.rolling(window=20, min_periods=10).median()) & \
                     (volume_clustering > 1.2)
    
    # Trend regime + volume persistence → continuation likelihood
    continuation_signal = trend_regime & (volume_persistence > 0.3)
    
    # High volatility + efficiency decay → mean reversion setup
    mean_reversion_signal = (volatility_state > 1.2) & (efficiency_decay < 0)
    
    # 5. Generate Composite Alpha Signal
    # Base signal combining efficiency metrics
    base_signal = efficiency_5d * volume_weighted_efficiency * (1 + efficiency_decay)
    
    # Apply regime-specific scaling
    # Trend regime scaling
    trend_scaling = np.where(trend_regime == 1, 1.2, 0.8)
    
    # Volatility scaling (inverse relationship for stability)
    vol_scaling = 1 / (1 + volatility_state)
    
    # Liquidity scaling
    liq_scaling = np.where(liquidity_regime > liquidity_regime.rolling(window=20, min_periods=10).median(), 
                          1.1, 0.9)
    
    # Combine signals with regime weights
    composite_signal = base_signal.copy()
    
    # Apply momentum/reversal logic
    composite_signal = np.where(momentum_signal, composite_signal * 1.3, composite_signal)
    composite_signal = np.where(reversal_signal, composite_signal * -1.2, composite_signal)
    composite_signal = np.where(continuation_signal, composite_signal * 1.1, composite_signal)
    composite_signal = np.where(mean_reversion_signal, composite_signal * -1.1, composite_signal)
    
    # Final regime scaling
    final_signal = composite_signal * trend_scaling * vol_scaling * liq_scaling
    
    # Weight by volume persistence strength
    volume_persistence_weight = 0.5 + (volume_persistence.fillna(0) * 0.5)
    final_signal = final_signal * volume_persistence_weight
    
    # Return as pandas Series
    return pd.Series(final_signal, index=data.index, name='alpha_factor')
