import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Percentile-based momentum acceleration with volume divergence and multiplicative smoothing.
    
    Interpretation:
    - Momentum acceleration measured through percentile ranking of price changes across multiple timeframes
    - Volume divergence detects when current trading activity deviates from recent patterns
    - Multiplicative smoothing combines signals across short and medium-term horizons
    - Volatility scaling uses price range to normalize momentum signals
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest bearish momentum acceleration with volume distribution
    """
    
    # Price range for volatility scaling
    price_range = df['high'] - df['low']
    
    # Momentum components normalized by price range
    intraday_return = (df['close'] - df['open']) / (price_range + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (price_range + 1e-7)
    daily_return = (df['close'] - df['close'].shift(1)) / (price_range + 1e-7)
    
    # Momentum acceleration using percentile ranking
    intraday_accel = intraday_return.rolling(window=5).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)) * 1.0 + (x.iloc[-1] < x.quantile(0.4)) * -1.0, raw=False)
    overnight_accel = overnight_return.rolling(window=5).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)) * 1.0 + (x.iloc[-1] < x.quantile(0.4)) * -1.0, raw=False)
    daily_accel = daily_return.rolling(window=5).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)) * 1.0 + (x.iloc[-1] < x.quantile(0.4)) * -1.0, raw=False)
    
    # Combined momentum acceleration
    momentum_accel = (intraday_accel + overnight_accel + daily_accel) / 3.0
    
    # Volume divergence using percentile comparison
    volume_rank = df['volume'].rolling(window=10).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 1.0 + (x.iloc[-1] < x.quantile(0.3)) * -1.0, raw=False)
    amount_rank = df['amount'].rolling(window=10).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 1.0 + (x.iloc[-1] < x.quantile(0.3)) * -1.0, raw=False)
    
    # Volume divergence signal
    volume_divergence = volume_rank * amount_rank
    
    # Multiplicative smoothing across timeframes
    short_term_signal = momentum_accel.rolling(window=3).mean()
    medium_term_signal = momentum_accel.rolling(window=8).mean()
    
    # Combined factor with multiplicative interaction
    alpha_factor = (
        short_term_signal * medium_term_signal * 
        np.sign(short_term_signal * medium_term_signal) * 
        volume_divergence
    )
    
    return alpha_factor
