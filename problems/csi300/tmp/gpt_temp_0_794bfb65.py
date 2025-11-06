import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Momentum
    close = df['close']
    volume = df['volume']
    
    # Multi-timeframe Momentum
    mom_5 = close / close.shift(5) - 1
    mom_10 = close / close.shift(10) - 1
    mom_20 = close / close.shift(20) - 1
    
    # Volume Confirmation
    vol_ratio_5 = volume / volume.shift(5)
    vol_ratio_10 = volume / volume.shift(10)
    vol_acceleration = vol_ratio_5 / (volume.shift(5) / volume.shift(10))
    
    # Volume persistence (count of days with volume > previous day's volume over last 5 days)
    vol_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window = volume.iloc[i-5:i+1]
        vol_persistence.iloc[i] = (window > window.shift(1)).sum() - 1  # exclude current day comparison
    
    # Divergence Signal
    bullish_div = ((mom_5 > mom_10) & (vol_ratio_5 < vol_ratio_10)).astype(int)
    bearish_div = ((mom_5 < mom_10) & (vol_ratio_5 > vol_ratio_10)).astype(int)
    divergence_signal = bullish_div - bearish_div
    strength_weighted_div = divergence_signal * abs(mom_5)
    
    # High-Low Range Efficiency
    high = df['high']
    low = df['low']
    
    # Daily Efficiency
    raw_efficiency = abs(close - close.shift(1)) / (high - low)
    
    gap_adjusted_efficiency = abs(close - close.shift(1)) / (
        np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1))
    )
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    true_range_efficiency = abs(close - close.shift(1)) / tr
    
    # Multi-day Persistence
    price_change_abs = abs(close - close.shift(1))
    
    efficiency_3d = price_change_abs.rolling(window=3).sum() / tr.rolling(window=3).sum()
    efficiency_5d = price_change_abs.rolling(window=5).sum() / tr.rolling(window=5).sum()
    efficiency_trend = efficiency_3d / efficiency_5d
    
    # Efficiency consistency (count of days with efficiency > 0.5 over last 5 days)
    efficiency_consistency = (raw_efficiency > 0.5).rolling(window=5).sum()
    
    # Signal Generation
    low_efficiency_reversal = (efficiency_3d < 0.3) * (close / close.shift(1) - 1)
    high_efficiency_momentum = (efficiency_3d > 0.7) * mom_5
    efficiency_break = raw_efficiency / raw_efficiency.rolling(window=5).mean()
    
    # Volume-Confirmed Extreme Reversal
    # Extreme Move Detection
    price_dev_3d = (close - close.rolling(window=3).mean().shift(1)) / close.rolling(window=3).std().shift(1)
    price_zscore_5d = (close - close.rolling(window=5).mean().shift(1)) / close.rolling(window=5).std().shift(1)
    range_position = (close - low) / (high - low)
    
    # Volume Confirmation
    volume_spike = volume / volume.rolling(window=5).mean().shift(1)
    volume_persistence_ratio = volume / volume.shift(1)
    volume_clustering = (volume > volume.rolling(window=5).mean()).rolling(window=3).sum()
    
    # Reversal Signal (using next period return as target)
    next_return = close.shift(-1) / close - 1
    overbought_reversal = (price_dev_3d > 2) * (volume_spike > 1.5) * next_return
    oversold_bounce = (price_dev_3d < -2) * (volume_spike > 1.5) * next_return
    volume_weighted_reversal = (overbought_reversal + oversold_bounce) * volume
    
    # Amount Flow Persistence
    amount = df['amount']
    
    # Directional Flow
    up_flow = amount.where(close > close.shift(1), 0)
    down_flow = amount.where(close < close.shift(1), 0)
    net_flow = up_flow - down_flow
    
    # Flow Momentum
    net_flow_3d = net_flow.rolling(window=3).sum()
    net_flow_5d = net_flow.rolling(window=5).sum()
    flow_acceleration = net_flow_3d / net_flow_5d
    
    # Flow direction consistency
    flow_sign = np.sign(net_flow)
    flow_consistency = (flow_sign == flow_sign.shift(1)).rolling(window=5).sum()
    
    # Persistence Signal
    strong_inflow_momentum = (net_flow_3d > 0) * (flow_acceleration > 1) * mom_5
    flow_exhaustion = (net_flow_3d > 0) * (flow_acceleration < 0.8) * next_return
    flow_persistence_score = flow_consistency * abs(net_flow_3d)
    
    # Volatility-Adaptive Volume Patterns
    # Volatility Regime
    range_volatility_10d = (high.rolling(window=10).max() - low.rolling(window=10).min()) / close.shift(10)
    vol_ratio_5d = close.rolling(window=5).std() / close.shift(5).rolling(window=5).std()
    volatility_trend = close.rolling(window=5).std() / close.rolling(window=10).std()
    
    # Volume Regime
    volume_volatility = volume.rolling(window=5).std() / volume.rolling(window=5).mean()
    volume_spike_frequency = (volume > 2 * volume.rolling(window=10).mean()).rolling(window=10).sum()
    volume_persistence_ratio_5d = (volume > volume.shift(1)).rolling(window=5).sum() / 5
    
    # Regime Signals
    returns_abs = abs(close / close.shift(1) - 1)
    vol_volume_corr = volume.rolling(window=10).corr(returns_abs)
    
    high_vol_breakout = (volatility_trend > 1.2) * (volume_spike_frequency > 2) * mom_5
    low_vol_accumulation = (volatility_trend < 0.8) * (volume_persistence_ratio_5d > 0.6) * next_return
    volatility_volume_correlation = np.sign(vol_volume_corr) * abs(volatility_trend)
    
    # Combine all factors with equal weights
    factors = [
        strength_weighted_div,
        low_efficiency_reversal,
        high_efficiency_momentum,
        efficiency_break,
        overbought_reversal,
        oversold_bounce,
        volume_weighted_reversal,
        strong_inflow_momentum,
        flow_exhaustion,
        flow_persistence_score,
        high_vol_breakout,
        low_vol_accumulation,
        volatility_volume_correlation
    ]
    
    # Normalize and combine
    normalized_factors = []
    for factor in factors:
        if len(factor.dropna()) > 0:
            normalized = (factor - factor.mean()) / factor.std()
            normalized_factors.append(normalized)
    
    # Equal-weighted combination
    combined_factor = pd.concat(normalized_factors, axis=1).mean(axis=1)
    
    return combined_factor
