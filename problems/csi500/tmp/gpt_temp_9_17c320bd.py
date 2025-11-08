import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum Dynamics with Volume Confirmation
    # Intraday Momentum Decay Analysis
    df['intraday_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['momentum_decay_5d'] = df['intraday_momentum'].ewm(span=5, adjust=False).mean()
    df['momentum_decay_accel'] = df['momentum_decay_5d'].diff(2)
    
    # Volume Acceleration Profile
    df['volume_pct_change'] = df['volume'].pct_change()
    df['volume_ma_5d'] = df['volume'].rolling(window=5).mean()
    df['volume_accel_3d'] = df['volume_pct_change'].diff(3)
    df['volume_clustering'] = (df['volume'] > df['volume_ma_5d'] * 1.2).astype(int)
    
    # Momentum-Volume Phase Analysis
    df['momentum_volume_divergence'] = (df['intraday_momentum'] * df['volume_pct_change']).rolling(window=5).std()
    
    # Price Range Compression Breakout Framework
    # Range Compression Measurement
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_20d'] = df['true_range'].rolling(window=20).mean()
    df['range_compression'] = (df['high'] - df['low']) / df['atr_20d']
    
    # Compression duration tracking
    compression_mask = df['range_compression'] < df['range_compression'].rolling(window=20).quantile(0.3)
    df['compression_duration'] = compression_mask.astype(int).groupby((~compression_mask).astype(int).cumsum()).cumsum()
    
    # Volume Behavior During Compression Phases
    df['compression_volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['volume_trend_slope'] = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Breakout Probability Assessment
    df['volume_accumulation'] = df['volume'].rolling(window=5).sum() / df['volume'].rolling(window=20).sum()
    df['breakout_probability'] = (df['compression_duration'] * df['volume_accumulation']) / df['range_compression']
    
    # Market Context Integration
    # Volatility Regime Context
    df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
    df['volatility_regime'] = (df['volatility_20d'] > df['volatility_20d'].rolling(window=60).quantile(0.7)).astype(int)
    
    # Trend Regime Detection
    df['price_slope_5d'] = df['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['volume_trend_5d'] = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['trend_regime'] = ((df['price_slope_5d'] > 0) & (df['volume_trend_5d'] > 0)).astype(int)
    
    # Adaptive Factor Weighting
    df['regime_weight'] = 1 + 0.5 * df['volatility_regime'] + 0.3 * df['trend_regime']
    
    # Final Alpha Factor Synthesis
    # Momentum-Regime Adaptive Factor
    df['momentum_regime_factor'] = (
        df['regime_weight'] * df['momentum_decay_5d'] + 
        df['volume_accel_3d'] * df['momentum_decay_5d'] +
        df['momentum_volume_divergence']
    )
    
    # Compression-Breakout Predictive Factor
    df['compression_breakout_factor'] = (
        df['breakout_probability'] * df['regime_weight'] +
        df['compression_duration'] * df['volume_accumulation'] +
        df['compression_volume_ratio']
    )
    
    # Combined Alpha Output
    alpha_factor = (
        0.6 * df['momentum_regime_factor'] + 
        0.4 * df['compression_breakout_factor']
    )
    
    return alpha_factor
