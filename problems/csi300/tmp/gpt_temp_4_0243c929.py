import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Volume Divergence factor
    Combines multi-timeframe acceleration with regime detection and smart money analysis
    """
    
    # Calculate basic price features
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Multi-Timeframe Acceleration
    # Price acceleration (short-term vs medium-term velocity)
    short_ma = df['close'].rolling(window=5).mean()
    medium_ma = df['close'].rolling(window=20).mean()
    price_accel = (short_ma - short_ma.shift(5)) / (medium_ma - medium_ma.shift(5))
    
    # Volume acceleration
    volume_roc = df['volume'].pct_change(periods=5)
    volume_trend = df['volume'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_accel = volume_roc * volume_trend
    
    # Asymmetric momentum
    upside_returns = df['returns'].where(df['returns'] > 0, 0)
    downside_returns = df['returns'].where(df['returns'] < 0, 0)
    upside_accel = upside_returns.rolling(window=10).mean()
    downside_accel = downside_returns.rolling(window=10).mean()
    asym_momentum = upside_accel / (abs(downside_accel) + 1e-8)
    
    # Market Regime Detection
    # Volatility state
    vol_ratio = df['true_range'] / df['true_range'].rolling(window=20).mean()
    
    # Range efficiency (price completion relative to daily range)
    range_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    range_efficiency_smooth = range_efficiency.rolling(window=10).mean()
    
    # Smart Money & Divergence
    # Large trade analysis using amount-based flow
    avg_trade_size = df['amount'] / (df['volume'] + 1e-8)
    large_trade_bias = (avg_trade_size - avg_trade_size.rolling(window=20).mean()) / avg_trade_size.rolling(window=20).std()
    
    # Acceleration-volume divergence
    price_momentum = df['close'].pct_change(periods=5)
    volume_momentum = df['volume'].pct_change(periods=5)
    accel_volume_divergence = price_momentum - volume_momentum
    
    # Regime-Specific Signal Construction
    # High volatility regime (vol_ratio > 1.2)
    high_vol_signal = (price_accel * vol_ratio * accel_volume_divergence).where(vol_ratio > 1.2, 0)
    
    # Normal regime (0.8 <= vol_ratio <= 1.2)
    normal_signal = (asym_momentum * (1 - abs(range_efficiency_smooth)) * price_accel).where(
        (vol_ratio >= 0.8) & (vol_ratio <= 1.2), 0
    )
    
    # Low volatility regime (vol_ratio < 0.8)
    low_vol_signal = (price_accel.rolling(window=5).mean() * large_trade_bias * volume_accel).where(vol_ratio < 0.8, 0)
    
    # Combine regime-specific signals
    regime_adaptive_factor = high_vol_signal + normal_signal + low_vol_signal
    
    # Final normalization and smoothing
    factor = regime_adaptive_factor.rolling(window=5).mean()
    factor = (factor - factor.rolling(window=50).mean()) / factor.rolling(window=50).std()
    
    return factor
