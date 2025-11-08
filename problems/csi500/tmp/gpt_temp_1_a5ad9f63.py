import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multi-timeframe momentum acceleration with volume anomaly detection
    and regime-aware dynamic weighting for improved return prediction.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Multi-Timeframe Momentum Acceleration
    # Short-term momentum (5-day ROC)
    momentum_short = df['close'].pct_change(5)
    
    # Medium-term momentum (15-day price slope)
    def price_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series) - window + 1):
            y = series.iloc[i:i+window].values
            if len(y) == window and not np.any(np.isnan(y)):
                slope = np.polyfit(x, y, 1)[0] / series.iloc[i]
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index[window-1:])
    
    momentum_medium = price_slope(df['close'], 15)
    momentum_medium = momentum_medium.reindex(df.index)
    
    # Long-term trend strength (30-day normalized trend)
    momentum_long = (df['close'] - df['close'].rolling(30).mean()) / df['close'].rolling(30).std()
    
    # Momentum derivatives and acceleration
    momentum_change_short = momentum_short.diff(3)  # First derivative
    momentum_accel_short = momentum_change_short.diff(3)  # Second derivative
    
    momentum_change_medium = momentum_medium.diff(5)
    momentum_accel_medium = momentum_change_medium.diff(5)
    
    # Volume Anomaly Detection
    volume_ma_short = df['volume'].rolling(5).mean()
    volume_ma_long = df['volume'].rolling(20).mean()
    volume_zscore = (df['volume'] - volume_ma_short) / volume_ma_short.rolling(10).std()
    
    # Volume-price divergence signals
    price_up = df['close'] > df['close'].shift(1)
    volume_down = df['volume'] < df['volume'].shift(1)
    price_down = df['close'] < df['close'].shift(1)
    volume_up = df['volume'] > df['volume'].shift(1)
    
    volume_price_divergence = pd.Series(0, index=df.index)
    volume_price_divergence[(price_up & volume_down)] = -1  # Weakness signal
    volume_price_divergence[(price_down & volume_up)] = 1   # Strength signal
    
    # Momentum-Volume Divergence
    momentum_up = momentum_short > momentum_short.shift(1)
    volume_trend_up = df['volume'] > volume_ma_short
    
    bullish_divergence = ((df['close'] < df['close'].shift(5)) & 
                         (momentum_short > momentum_short.shift(5)) & 
                         volume_trend_up)
    
    bearish_divergence = ((df['close'] > df['close'].shift(5)) & 
                         (momentum_short < momentum_short.shift(5)) & 
                         (~volume_trend_up))
    
    # Regime Identification
    # Volatility regime (20-day rolling volatility)
    volatility = df['close'].pct_change().rolling(20).std()
    high_vol_regime = volatility > volatility.rolling(50).quantile(0.7)
    low_vol_regime = volatility < volatility.rolling(50).quantile(0.3)
    
    # Trend regime (ADX-like measure)
    high_low_range = df['high'] - df['low']
    high_close_range = abs(df['high'] - df['close'].shift(1))
    low_close_range = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    
    directional_move_up = df['high'] - df['high'].shift(1)
    directional_move_down = df['low'].shift(1) - df['low']
    
    plus_dm = np.where((directional_move_up > directional_move_down) & (directional_move_up > 0), directional_move_up, 0)
    minus_dm = np.where((directional_move_down > directional_move_up) & (directional_move_down > 0), directional_move_down, 0)
    
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / true_range.rolling(14).mean()
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / true_range.rolling(14).mean()
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(14).mean()
    
    trending_regime = adx > 25
    mean_reverting_regime = adx < 20
    
    # Dynamic weighting based on regimes
    momentum_weight = pd.Series(0.6, index=df.index)  # Base weight
    momentum_weight[trending_regime] = 0.8
    momentum_weight[mean_reverting_regime] = 0.4
    
    volume_weight = pd.Series(0.4, index=df.index)  # Base weight
    volume_weight[high_vol_regime] = 0.6
    volume_weight[low_vol_regime] = 0.3
    
    # Volume quality assessment
    volume_persistence = (df['volume'] > volume_ma_short).rolling(5).sum() / 5
    volume_spike = volume_zscore > 2
    
    # Volume during momentum changes
    volume_momentum_alignment = pd.Series(0, index=df.index)
    volume_accel_positive = (momentum_accel_short > 0) & (df['volume'] > volume_ma_short)
    volume_accel_negative = (momentum_accel_short < 0) & (df['volume'] > volume_ma_short)
    
    volume_momentum_alignment[volume_accel_positive] = 1
    volume_momentum_alignment[volume_accel_negative] = -1
    
    # Composite factor calculation
    # Normalize components
    momentum_composite = (0.4 * momentum_short.rank(pct=True) + 
                         0.35 * momentum_medium.rank(pct=True) + 
                         0.25 * momentum_long.rank(pct=True))
    
    volume_composite = (0.3 * volume_zscore.rank(pct=True) + 
                       0.25 * volume_price_divergence.rank(pct=True) + 
                       0.2 * volume_persistence.rank(pct=True) + 
                       0.15 * pd.Series(bullish_divergence.astype(int) - bearish_divergence.astype(int)).rank(pct=True) + 
                       0.1 * volume_momentum_alignment.rank(pct=True))
    
    # Apply regime-aware dynamic weighting
    final_factor = (momentum_weight * momentum_composite + 
                   volume_weight * volume_composite)
    
    # Add regime-specific adjustments
    final_factor[trending_regime] = final_factor[trending_regime] * 1.2
    final_factor[mean_reverting_regime] = final_factor[mean_reverting_regime] * 0.8
    
    # Volume confirmation requirement
    volume_confirmation = volume_zscore > 0.5
    final_factor[~volume_confirmation & high_vol_regime] = final_factor[~volume_confirmation & high_vol_regime] * 0.7
    
    result = final_factor
    
    return result
