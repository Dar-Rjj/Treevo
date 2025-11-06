import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Price momentum divergence (correlation between price and volume momentum)
    price_momentum = df['close'].pct_change()
    volume_momentum = df['volume'].pct_change()
    momentum_divergence = price_momentum.rolling(window=10).corr(volume_momentum)
    
    # Volatility-regime adjusted range efficiency
    daily_range = df['high'] - df['low']
    range_volatility = daily_range.rolling(window=20).std()
    normalized_range_efficiency = (df['close'] - df['low']) / (daily_range + 1e-7)
    regime_adjusted_range = normalized_range_efficiency * (1 / (range_volatility + 1e-7))
    
    # Price acceleration with volume confirmation
    price_acceleration = price_momentum.diff()
    volume_acceleration = volume_momentum.diff()
    confirmed_acceleration = price_acceleration.rolling(window=5).corr(volume_acceleration)
    
    # Microstructure pressure (opening gap relative to recent volatility)
    recent_volatility = df['close'].pct_change().rolling(window=10).std()
    opening_gap_pressure = (df['open'] - df['close'].shift(1)) / (recent_volatility + 1e-7)
    
    # Liquidity efficiency (amount per volume vs price impact)
    trade_efficiency = df['amount'] / (df['volume'] + 1e-7)
    price_impact = df['close'].pct_change().abs()
    liquidity_signal = trade_efficiency.rolling(window=10).corr(price_impact)
    
    # Trend persistence (autocorrelation of range efficiency)
    range_efficiency = (df['close'] - df['low']) / (daily_range + 1e-7)
    trend_persistence = range_efficiency.rolling(window=15).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
    )
    
    # Weighted combination with emphasis on confirmed signals
    factor = (
        momentum_divergence * 0.25 +
        regime_adjusted_range.diff() * 0.20 +
        confirmed_acceleration * 0.18 +
        opening_gap_pressure * 0.15 +
        liquidity_signal * 0.12 +
        trend_persistence * 0.10
    )
    
    return factor
