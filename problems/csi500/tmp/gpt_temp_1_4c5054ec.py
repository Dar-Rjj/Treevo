import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Volatility Scaling
    # Calculate Price Momentum Signals
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_8d = df['close'] / df['close'].shift(8) - 1
    momentum_20d = df['close'] / df['close'].shift(20) - 1
    
    # Compute Volatility Estimates
    daily_range_vol = (df['high'] - df['low']) / df['close'].shift(1)
    rolling_5d_vol = daily_range_vol.rolling(window=5).mean()
    
    # Volatility-Scaled Momentum
    vol_scaled_momentum_3d = momentum_3d / rolling_5d_vol
    vol_scaled_momentum_8d = momentum_8d / rolling_5d_vol
    vol_scaled_momentum_20d = momentum_20d / rolling_5d_vol
    
    # Volume Regime Detection
    # Volume Trend Analysis
    volume_ma_5d = df['volume'].rolling(window=5).mean()
    volume_ma_20d = df['volume'].rolling(window=20).mean()
    volume_ratio = volume_ma_5d / volume_ma_20d
    volume_trend = np.tanh(volume_ratio - 1)
    
    # Volume Percentile Scoring
    volume_percentile = df['volume'].rolling(window=20).apply(
        lambda x: (x.rank(pct=True).iloc[-1] * 100), raw=False
    )
    volume_percentile_score = (volume_percentile / 100) ** (1/3)
    
    # Volume Regime Composite
    volume_regime_composite = (volume_trend + volume_percentile_score) / 2
    
    # Momentum-Volume Alignment
    # Match Timeframe Alignment
    aligned_short = vol_scaled_momentum_3d * volume_trend
    aligned_medium = vol_scaled_momentum_8d * volume_percentile_score
    aligned_long = vol_scaled_momentum_20d * volume_regime_composite
    
    # Volatility Scaling Application
    # (Already applied in vol_scaled_momentum calculations)
    
    # Composite Alpha Generation
    # Multiplicative Combination
    product_factors = aligned_short * aligned_medium * aligned_long
    cube_root_product = np.sign(product_factors) * np.abs(product_factors) ** (1/3)
    
    # Bounded Output
    final_alpha = np.tanh(cube_root_product)
    
    return final_alpha
