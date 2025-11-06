import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-scale momentum with adaptive efficiency and volatility filtering.
    
    Combines short-term (3-day), medium-term (8-day), and long-term (21-day) price momentum
    with adaptive efficiency measures across volume, range, and amount dimensions.
    Uses multi-timeframe volatility filtering that adapts to different market regimes
    and incorporates efficiency persistence across time horizons.
    
    Economic intuition: True momentum signals exhibit persistence across multiple timeframes
    and are characterized by consistent efficiency patterns. The adaptive filtering mechanism
    distinguishes sustainable momentum from noise by considering efficiency stability and
    volatility regime appropriateness across different investment horizons.
    """
    short_window = 3
    medium_window = 8
    long_window = 21
    
    # Multi-scale momentum components
    short_momentum = df['close'] / df['close'].shift(short_window) - 1
    medium_momentum = df['close'] / df['close'].shift(medium_window) - 1
    long_momentum = df['close'] / df['close'].shift(long_window) - 1
    
    # Adaptive efficiency measures
    price_change = abs(df['close'] - df['close'].shift(1))
    
    # Volume efficiency with multi-timeframe persistence
    volume_efficiency = price_change / (df['volume'] + 1e-7)
    volume_persistence_short = volume_efficiency / volume_efficiency.rolling(window=short_window).mean()
    volume_persistence_medium = volume_efficiency / volume_efficiency.rolling(window=medium_window).mean()
    volume_persistence_long = volume_efficiency / volume_efficiency.rolling(window=long_window).mean()
    
    # Range efficiency with adaptive normalization
    daily_range = (df['high'] - df['low']) / df['close']
    price_movement = abs(df['close'] - df['close'].shift(1)) / df['close']
    range_efficiency = price_movement / (daily_range + 1e-7)
    
    range_stability_short = range_efficiency.rolling(window=short_window).std()
    range_stability_medium = range_efficiency.rolling(window=medium_window).std()
    range_stability_long = range_efficiency.rolling(window=long_window).std()
    
    # Amount efficiency with trend consistency
    amount_efficiency = price_change / (df['amount'] + 1e-7)
    amount_trend_short = amount_efficiency.rolling(window=short_window).apply(lambda x: (x[-1] - x[0]) / (abs(x).mean() + 1e-7))
    amount_trend_medium = amount_efficiency.rolling(window=medium_window).apply(lambda x: (x[-1] - x[0]) / (abs(x).mean() + 1e-7))
    amount_trend_long = amount_efficiency.rolling(window=long_window).apply(lambda x: (x[-1] - x[0]) / (abs(x).mean() + 1e-7))
    
    # Multi-timeframe volatility regimes
    returns = df['close'].pct_change()
    short_volatility = returns.rolling(window=short_window).std()
    medium_volatility = returns.rolling(window=medium_window).std()
    long_volatility = returns.rolling(window=long_window).std()
    
    # Volatility regime adaptation factors
    vol_regime_short = short_volatility / medium_volatility
    vol_regime_medium = medium_volatility / long_volatility
    vol_regime_long = long_volatility / long_volatility.rolling(window=63).mean()
    
    # Efficiency composite with volatility adaptation
    efficiency_short = (volume_persistence_short * (1 - range_stability_short) * amount_trend_short) / (vol_regime_short + 1e-7)
    efficiency_medium = (volume_persistence_medium * (1 - range_stability_medium) * amount_trend_medium) / (vol_regime_medium + 1e-7)
    efficiency_long = (volume_persistence_long * (1 - range_stability_long) * amount_trend_long) / (vol_regime_long + 1e-7)
    
    # Multi-scale momentum with adaptive efficiency weighting
    short_component = short_momentum * efficiency_short
    medium_component = medium_momentum * efficiency_medium
    long_component = long_momentum * efficiency_long
    
    # Hierarchical combination with time decay
    alpha_factor = short_component + 0.6 * medium_component + 0.3 * long_component
    
    return alpha_factor
