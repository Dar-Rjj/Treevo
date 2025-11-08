import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multiple technical heuristics combined.
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Combined alpha factor values indexed by date
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor components
    momentum_quality = pd.Series(index=data.index, dtype=float)
    range_breakout = pd.Series(index=data.index, dtype=float)
    volatility_regime = pd.Series(index=data.index, dtype=float)
    liquidity_pressure = pd.Series(index=data.index, dtype=float)
    intraday_trend = pd.Series(index=data.index, dtype=float)
    volume_return = pd.Series(index=data.index, dtype=float)
    price_volume_trend = pd.Series(index=data.index, dtype=float)
    range_expansion = pd.Series(index=data.index, dtype=float)
    efficiency_momentum = pd.Series(index=data.index, dtype=float)
    gap_mean_reversion = pd.Series(index=data.index, dtype=float)
    
    # 1. Momentum Quality
    momentum_10d = data['close'].pct_change(periods=10)
    momentum_20d = data['close'].pct_change(periods=20)
    momentum_divergence = momentum_10d - momentum_20d
    volume_ratio = data['volume'] / data['volume'].rolling(window=5).mean()
    momentum_quality = momentum_divergence * volume_ratio
    
    # 2. Range Breakout Strength
    prev_close = data['close'].shift(1)
    true_range = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - prev_close),
            abs(data['low'] - prev_close)
        )
    )
    range_ratio = true_range / true_range.rolling(window=5).mean()
    dollar_volume_ratio = data['amount'] / data['amount'].rolling(window=10).mean()
    range_breakout = range_ratio * dollar_volume_ratio
    
    # 3. Volatility Regime Change
    returns = data['close'].pct_change()
    vol_5d = returns.rolling(window=5).std()
    vol_20d = returns.rolling(window=20).std()
    vol_ratio = vol_5d / vol_20d
    volatility_regime = vol_ratio
    
    # 4. Liquidity-Pressure Interaction
    price_pressure = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    liquidity_inverse = 1 / (data['volume'] * data['amount'] + 1e-8)
    liquidity_pressure = price_pressure * liquidity_inverse
    
    # 5. Intraday Trend Persistence
    intraday_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    intraday_sign = np.sign(intraday_strength)
    consecutive_days = intraday_sign.groupby(intraday_sign.ne(intraday_sign.shift()).cumsum()).cumcount() + 1
    volume_trend = data['volume'] / data['volume'].rolling(window=5).mean()
    intraday_trend = intraday_strength * consecutive_days * volume_trend
    
    # 6. Volume-Return Anomaly
    volume_anomaly = data['volume'] / data['volume'].rolling(window=20).mean()
    daily_returns = data['close'].pct_change()
    volume_return = volume_anomaly * daily_returns
    
    # 7. Price-Volume Trend Alignment
    price_trend = data['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    volume_trend_slope = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    price_volume_trend = price_trend * volume_trend_slope
    
    # 8. Range Expansion Momentum
    daily_range_pct = (data['high'] - data['low']) / prev_close
    range_expansion_ratio = daily_range_pct / daily_range_pct.rolling(window=10).mean()
    directional_position = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    range_expansion = range_expansion_ratio * directional_position
    
    # 9. Efficiency Momentum Quality
    net_change = data['close'] - data['open']
    total_movement = abs(data['close'] - data['open']) + abs(data['high'] - data['close']) + abs(data['low'] - data['close'])
    efficiency_ratio = net_change / (total_movement + 1e-8)
    efficiency_acceleration = efficiency_ratio.diff()
    volume_pattern = data['volume'] / data['volume'].rolling(window=5).mean()
    efficiency_momentum = efficiency_acceleration * volume_pattern
    
    # 10. Gap Mean Reversion
    gap = data['open'] - prev_close
    gap_size = abs(gap) / prev_close
    fill_probability = 1 / (gap_size * data['volume'] + 1e-8)
    gap_mean_reversion = -gap * fill_probability  # Negative for mean reversion
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'momentum_quality': momentum_quality,
        'range_breakout': range_breakout,
        'volatility_regime': volatility_regime,
        'liquidity_pressure': liquidity_pressure,
        'intraday_trend': intraday_trend,
        'volume_return': volume_return,
        'price_volume_trend': price_volume_trend,
        'range_expansion': range_expansion,
        'efficiency_momentum': efficiency_momentum,
        'gap_mean_reversion': gap_mean_reversion
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=20).mean()) / x.rolling(window=20).std())
    
    # Equal-weighted combination
    combined_factor = factors_normalized.mean(axis=1)
    
    return combined_factor
