import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Acceleration Metrics
    price_acceleration = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume-Weighted Acceleration (3-day rolling)
    volume_weighted_numerator = (df['volume'] * (df['close'] - df['open'])).rolling(window=3, min_periods=1).sum()
    volume_weighted_denominator = (df['volume'] * (df['high'] - df['low'])).rolling(window=3, min_periods=1).sum()
    volume_weighted_acceleration = volume_weighted_numerator / volume_weighted_denominator.replace(0, np.nan)
    
    # Acceleration Trend
    acceleration_trend = price_acceleration - price_acceleration.rolling(window=3, min_periods=1).mean()
    
    # Divergence Detection
    # Price-Volume Divergence (5-day correlation)
    price_volume_corr = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        window_data = df.iloc[i-4:i+1]
        if len(window_data) >= 3:
            price_volume_corr.iloc[i] = window_data['close'].diff().corr(window_data['volume'])
    
    # High-Low Divergence
    high_close_ratio = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    close_low_ratio = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    hl_divergence = (high_close_ratio - close_low_ratio).diff()
    
    # Acceleration Momentum (slope over 3 days)
    acceleration_momentum = acceleration_trend.rolling(window=3, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / 2 if len(x) >= 2 else 0, raw=False
    )
    
    # Regime Identification
    range_5d_std = (df['high'] - df['low']).rolling(window=5, min_periods=1).std()
    range_10d_std = (df['high'] - df['low']).rolling(window=10, min_periods=1).std()
    volatility_ratio = range_5d_std / range_10d_std.replace(0, np.nan)
    
    high_vol_regime = volatility_ratio > 1.0
    low_vol_regime = volatility_ratio < 0.7
    transition_regime = (volatility_ratio >= 0.7) & (volatility_ratio <= 1.0)
    
    # Signal Construction
    bullish_divergence = (price_volume_corr < 0) & (acceleration_momentum > 0)
    bearish_divergence = (price_volume_corr > 0) & (acceleration_momentum < 0)
    
    # Volume Confirmation
    volume_avg = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_confirmation = df['volume'] > volume_avg
    
    # Regime-Enhanced Signals
    bullish_signal = bullish_divergence & volume_confirmation
    bearish_signal = bearish_divergence & volume_confirmation
    
    # Enhanced signals based on volatility regime
    enhanced_bullish = bullish_signal & low_vol_regime
    enhanced_bearish = bearish_signal & high_vol_regime
    
    # Final factor construction
    factor = pd.Series(0, index=df.index, dtype=float)
    factor[enhanced_bullish] = 1.0
    factor[enhanced_bearish] = -1.0
    factor[bullish_signal & ~enhanced_bullish] = 0.5
    factor[bearish_signal & ~enhanced_bearish] = -0.5
    
    # Add acceleration trend as a continuous component
    factor = factor + 0.1 * acceleration_trend.fillna(0)
    
    return factor
