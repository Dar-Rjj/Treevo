import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Novel alpha factor: Volatility-Adjusted Momentum with Volume Divergence
    # Factor interpretation:
    # - Captures price momentum strength relative to volatility environment
    # - Incorporates volume divergence to detect conviction behind price moves
    # - Higher values indicate strong momentum with supporting volume patterns
    # - Avoids normalization, uses raw relationships for interpretability
    
    # Price momentum components
    short_momentum = df['close'] / df['close'].shift(3) - 1
    medium_momentum = df['close'] / df['close'].shift(8) - 1
    
    # Volatility estimation using high-low range
    daily_range = (df['high'] - df['low']) / df['close']
    volatility = daily_range.rolling(window=5).std()
    
    # Volume divergence: current volume vs recent trend
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    volume_momentum = df['volume'] / df['volume'].rolling(window=10).mean() - 1
    
    # Combined factor: momentum weighted by volatility environment and volume confirmation
    factor = (short_momentum * medium_momentum * (1 + volume_trend)) / (volatility + 1e-7)
    
    return factor
