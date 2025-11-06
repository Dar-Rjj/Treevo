import pandas as pd
import pandas as pd

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum with volume-amount confirmation and adaptive volatility scaling
    # Uses blended signals across short, medium, and long timeframes for robust predictions
    
    # Triple-timeframe momentum with exponential weighting (shorter periods get higher weight)
    momentum_short = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_medium = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    momentum_long = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    blended_momentum = 0.6 * momentum_short + 0.3 * momentum_medium + 0.1 * momentum_long
    
    # Volume confirmation across multiple horizons
    volume_ratio_short = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_ratio_medium = df['volume'] / df['volume'].rolling(window=20).mean()
    volume_confirmation = 0.7 * volume_ratio_short + 0.3 * volume_ratio_medium
    
    # Amount-based intensity signals
    avg_trade_size = df['amount'] / (df['volume'] + 1e-7)
    trade_size_momentum = (avg_trade_size - avg_trade_size.rolling(window=10).mean()) / (avg_trade_size.rolling(window=10).mean() + 1e-7)
    
    # Price range efficiency (how much of daily range was utilized for price movement)
    range_utilization = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    range_efficiency = range_utilization.rolling(window=10).mean()
    
    # Multi-measure volatility using range and gap signals
    daily_range = (df['high'] - df['low']) / df['close']
    gap_volatility = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    blended_volatility = 0.8 * daily_range + 0.2 * gap_volatility
    
    # Adaptive volatility scaling using rolling percentiles
    vol_scale = blended_volatility.rolling(window=50).apply(lambda x: x.rank(pct=True).iloc[-1])
    
    # Final factor: momentum amplified by volume confirmation and trade size momentum,
    # enhanced by range efficiency, and adaptively scaled by volatility regime
    factor = (blended_momentum * volume_confirmation * trade_size_momentum * 
              range_efficiency / (vol_scale + 1e-7))
    
    return factor
