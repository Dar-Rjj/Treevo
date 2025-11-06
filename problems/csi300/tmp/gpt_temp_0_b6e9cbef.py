import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum with volume confirmation and asymmetric smoothing
    returns = df['close'].pct_change()
    volume_ratio = df['volume'] / df['volume'].rolling(window=3, min_periods=1).mean()
    
    # Asymmetric momentum: different decay for up/down moves
    up_momentum = returns.where(returns > 0, 0).ewm(span=2, adjust=False).mean()
    down_momentum = returns.where(returns < 0, 0).ewm(span=5, adjust=False).mean()
    
    # Volume-confirmed momentum divergence
    momentum_signal = (up_momentum - down_momentum) * volume_ratio
    
    # Price pressure with range normalization
    prev_close = df['close'].shift(1)
    daily_range = df['high'] - df['low'] + 1e-7
    
    # Asymmetric pressure components
    high_rejection = (df['high'] - prev_close) / daily_range
    low_support = (prev_close - df['low']) / daily_range
    
    # Different smoothing for rejections vs support
    rejection_smooth = high_rejection.ewm(span=2, adjust=False).mean()
    support_smooth = low_support.ewm(span=4, adjust=False).mean()
    
    # Trade efficiency impact
    avg_price = df['amount'] / (df['volume'] + 1e-7)
    trade_efficiency = avg_price / avg_price.rolling(window=5, min_periods=1).mean()
    
    pressure_signal = (rejection_smooth - support_smooth) * trade_efficiency
    
    # Gap persistence with range efficiency
    overnight_return = (df['open'] - prev_close) / prev_close
    intraday_efficiency = (df['close'] - df['open']) / daily_range
    
    # Gap momentum with persistence filtering
    gap_strength = overnight_return.rolling(window=3, min_periods=1).apply(
        lambda x: x[-1] if abs(x[-1]) > 0.5 * abs(x[:-1].mean()) else 0
    )
    
    gap_efficiency_signal = intraday_efficiency * gap_strength
    
    # Volume-amount alignment trend
    vwap_trend = (df['amount'] / df['volume']).pct_change().ewm(span=4, adjust=False).mean()
    volume_acceleration = df['volume'].pct_change().ewm(span=3, adjust=False).mean()
    
    flow_signal = vwap_trend * volume_acceleration
    
    # Volatility scaling
    volatility = returns.rolling(window=5, min_periods=1).std() + 1e-7
    
    # Economically weighted combination
    factor = (
        0.40 * momentum_signal / volatility +
        0.25 * pressure_signal / volatility +
        0.20 * gap_efficiency_signal / volatility +
        0.15 * flow_signal / volatility
    )
    
    return factor
