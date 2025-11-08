import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Intraday Elasticity Momentum with Volume Confirmation
    # Calculate Intraday Price Elasticity
    daily_range = (data['high'] - data['low']) / data['open']
    avg_range_10d = daily_range.rolling(window=10).mean()
    
    # Identify high volume days (top 20% by volume)
    volume_quantile = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).quantile(0.8), raw=False
    )
    high_volume_mask = data['volume'] > volume_quantile
    
    # Compute range expansion ratio on high volume days
    range_expansion = np.where(
        high_volume_mask,
        daily_range / avg_range_10d,
        1.0
    )
    
    # Intraday Momentum Efficiency
    morning_momentum = (data['high'] - data['open']) / data['open']
    afternoon_momentum = (data['close'] - data['low']) / data['low']
    intraday_momentum_ratio = morning_momentum / (afternoon_momentum + 1e-8)
    
    # Volume-Weighted Confirmation
    volume_momentum = data['volume'].pct_change(periods=5)
    vwap = data['amount'] / (data['volume'] + 1e-8)
    vwap_position = (data['close'] - vwap) / (vwap + 1e-8)
    
    # Weight elasticity-momentum by volume strength
    elasticity_momentum = range_expansion * intraday_momentum_ratio
    volume_weighted_elasticity = elasticity_momentum * (1 + volume_momentum)
    
    # Regime-Adaptive Signal Generation
    # Volatility Regime Classification
    realized_vol = (data['high'] - data['low']).rolling(window=10).std()
    vol_20d_avg = realized_vol.rolling(window=20).mean()
    
    high_vol_regime = realized_vol > (1.5 * vol_20d_avg)
    low_vol_regime = realized_vol < (0.7 * vol_20d_avg)
    
    # Regime-Specific Signal Scaling
    signal_high_vol = -volume_weighted_elasticity * 1.2  # Mean reversion emphasis
    signal_low_vol = volume_weighted_elasticity * 0.8    # Momentum emphasis
    signal_normal = volume_weighted_elasticity
    
    regime_signal = np.where(
        high_vol_regime, signal_high_vol,
        np.where(low_vol_regime, signal_low_vol, signal_normal)
    )
    
    # Volume-Regime Alignment Filter
    volume_regime = data['volume'] > data['volume'].rolling(window=20).mean()
    regime_alignment = np.where(
        (high_vol_regime & volume_regime) | (low_vol_regime & ~volume_regime),
        regime_signal * 1.1,
        regime_signal * 0.9
    )
    
    # Price-Volume Divergence Persistence
    # Detect Multi-Timeframe Divergence
    price_return_5d = data['close'].pct_change(periods=5)
    volume_return_5d = data['volume'].pct_change(periods=5)
    price_return_10d = data['close'].pct_change(periods=10)
    volume_return_10d = data['volume'].pct_change(periods=10)
    
    divergence_5d = price_return_5d - volume_return_5d
    divergence_10d = price_return_10d - volume_return_10d
    
    divergence_acceleration = divergence_5d - divergence_10d.shift(5)
    
    # Order Flow Persistence Analysis
    order_flow = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    order_flow_autocorr = order_flow.rolling(window=8).apply(
        lambda x: pd.Series(x).autocorr(), raw=False
    )
    
    volume_trend = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Liquidity Threshold Validation
    turnover_rate = data['volume'] / data['volume'].rolling(window=20).mean()
    liquidity_score = data['volume'].rolling(window=20).rank(pct=True)
    liquidity_filter = liquidity_score > 0.3
    
    # Signal Refinement
    persistence_strength = order_flow_autocorr * (1 + np.abs(volume_trend))
    weighted_divergence = divergence_acceleration * persistence_strength
    directional_divergence = weighted_divergence * np.sign(divergence_5d)
    
    # Apply exponential decay
    decay_factor = 0.9
    decayed_divergence = directional_divergence.ewm(alpha=1-decay_factor).mean()
    
    # Combine signals with regime adaptation
    final_signal = regime_alignment + decayed_divergence
    
    # Apply liquidity filter
    final_signal = np.where(liquidity_filter, final_signal, 0)
    
    return pd.Series(final_signal, index=data.index)
