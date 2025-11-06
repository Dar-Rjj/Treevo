import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum convergence with volatility regime adjustment
    # Uses 30-day and 60-day timeframes for robust trend identification
    # Incorporates volume confirmation across multiple periods
    
    # Dual timeframe price momentum (30-day and 60-day)
    price_momentum_30d = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    price_momentum_60d = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    
    # Momentum convergence score - stronger when both timeframes align
    momentum_convergence = price_momentum_30d * price_momentum_60d
    
    # Volume trend persistence across multiple periods
    volume_ma_short = df['volume'].rolling(window=10, min_periods=5).mean()
    volume_ma_medium = df['volume'].rolling(window=30, min_periods=15).mean()
    volume_ma_long = df['volume'].rolling(window=60, min_periods=30).mean()
    
    # Volume trend alignment (positive when all moving averages are trending up)
    volume_trend_strength = (volume_ma_short / volume_ma_medium) * (volume_ma_medium / volume_ma_long)
    
    # Volatility regime adjustment using rolling percentiles
    daily_range_ratio = (df['high'] - df['low']) / df['close']
    volatility_regime = daily_range_ratio.rolling(window=60, min_periods=30).apply(
        lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-7)
    )
    
    # Price efficiency within daily range (captures buying/selling pressure)
    close_to_high_ratio = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    open_to_close_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    price_efficiency = close_to_high_ratio * open_to_close_efficiency
    
    # Amount-based liquidity factor (using amount instead of just volume)
    amount_momentum = (df['amount'] - df['amount'].shift(30)) / (df['amount'].shift(30) + 1e-7)
    
    # Combined factor: emphasizes stocks with:
    # - Converging momentum across timeframes
    # - Strong volume trend confirmation
    # - Favorable volatility regime (moderate volatility)
    # - High price efficiency within daily range
    # - Positive amount momentum indicating institutional interest
    alpha_factor = (momentum_convergence * 
                   volume_trend_strength * 
                   (1 - np.abs(volatility_regime - 0.5)) * 
                   price_efficiency * 
                   np.sign(amount_momentum))
    
    return alpha_factor
