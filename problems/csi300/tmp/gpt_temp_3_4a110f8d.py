import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volume-weighted momentum convergence with volatility regime adaptation and price efficiency scoring.
    
    Interpretation:
    - Triple timeframe momentum (intraday, overnight, multi-day) with volume confirmation
    - Volatility regime detection using rolling percentiles for adaptive scaling
    - Volume-pressure scoring based on recent distribution patterns
    - Price efficiency measure combining momentum persistence and volatility efficiency
    - Momentum convergence scoring enhances signal reliability across different market phases
    - Positive values indicate strong momentum with volume confirmation in appropriate volatility regime
    - Negative values suggest weak momentum or reversal patterns with volume distribution signals
    """
    
    # Multi-timeframe momentum components
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    short_term_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Volatility regime detection using rolling percentiles
    daily_range = (df['high'] - df['low']) / df['open']
    vol_5d = daily_range.rolling(window=5).std()
    vol_percentile = vol_5d.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1, raw=False)
    
    # Volume-pressure scoring
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_10 = df['volume'].rolling(window=10).mean()
    volume_pressure = (df['volume'] / volume_ma_5) * (volume_ma_5 / volume_ma_10)
    
    # Price efficiency measure
    high_low_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    open_close_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    price_efficiency = (high_low_efficiency + open_close_efficiency) / 2
    
    # Momentum convergence scoring
    momentum_alignment = np.sign(intraday_return * overnight_return * short_term_momentum)
    momentum_strength = (abs(intraday_return) + abs(overnight_return) + abs(short_term_momentum)) / 3
    
    # Volume-weighted momentum components
    volume_weighted_intraday = intraday_return * volume_pressure
    volume_weighted_overnight = overnight_return * volume_pressure
    volume_weighted_shortterm = short_term_momentum * volume_pressure
    
    # Regime-adaptive weights
    high_vol_weight = np.where(vol_percentile >= 2, 0.6, 1.0)
    medium_vol_weight = np.where(vol_percentile == 1, 0.8, 1.0)
    low_vol_weight = np.where(vol_percentile == 0, 1.2, 1.0)
    
    vol_regime_weight = high_vol_weight * medium_vol_weight * low_vol_weight
    
    # Combined alpha factor with convergence enhancement
    alpha_factor = (
        volume_weighted_intraday * 0.3 +
        volume_weighted_overnight * 0.25 +
        volume_weighted_shortterm * 0.35 +
        price_efficiency * 0.1 +
        momentum_alignment * momentum_strength * 0.15
    ) * vol_regime_weight
    
    return alpha_factor
