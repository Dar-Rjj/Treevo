import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe volatility-scaled momentum,
    volume-price divergence, intraday momentum convergence, and multi-timeframe integration.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # Volatility-Scaled Momentum
    # Multi-Timeframe Momentum Alignment
    mom_3d = close / close.shift(3) - 1
    mom_5d = close / close.shift(5) - 1
    mom_10d = close / close.shift(10) - 1
    
    # Volatility Scaling
    def calc_volatility(returns_series):
        return returns_series.rolling(window=len(returns_series)).std()
    
    returns_3d = close.pct_change(periods=1).rolling(window=3).apply(lambda x: x.std() if len(x) == 3 else np.nan)
    returns_5d = close.pct_change(periods=1).rolling(window=5).apply(lambda x: x.std() if len(x) == 5 else np.nan)
    returns_10d = close.pct_change(periods=1).rolling(window=10).apply(lambda x: x.std() if len(x) == 10 else np.nan)
    
    vol_scaled_mom_3d = mom_3d / returns_3d.replace(0, np.nan)
    vol_scaled_mom_5d = mom_5d / returns_5d.replace(0, np.nan)
    vol_scaled_mom_10d = mom_10d / returns_10d.replace(0, np.nan)
    
    # Combined volatility-scaled momentum signal
    vol_momentum_signal = np.sign(vol_scaled_mom_3d * vol_scaled_mom_5d * vol_scaled_mom_10d) * \
                         (np.abs(vol_scaled_mom_3d * vol_scaled_mom_5d * vol_scaled_mom_10d)) ** (1/3)
    
    # Volume-Price Divergence
    # Price Strength Indicators
    intraday_confirmation = (close - open_price) / (high - low).replace(0, np.nan)
    high_low_positioning = (close - low) / (high - low).replace(0, np.nan)
    price_persistence = (close - close.shift(1)) / (high - low).replace(0, np.nan)
    
    # Volume Divergence Analysis
    volume_ratio_1d = volume / volume.shift(1)
    volume_ratio_5d = volume / volume.rolling(window=4, min_periods=1).mean().shift(1)
    volume_ratio_10d = volume / volume.rolling(window=9, min_periods=1).mean().shift(1)
    
    # Volume trend alignment
    volume_trend_alignment = np.sign(volume_ratio_1d * volume_ratio_5d * volume_ratio_10d) * \
                           (np.abs(volume_ratio_1d * volume_ratio_5d * volume_ratio_10d)) ** (1/3)
    
    # Combined volume-price signal
    price_strength_geo_mean = np.sign(intraday_confirmation * high_low_positioning * price_persistence) * \
                             (np.abs(intraday_confirmation * high_low_positioning * price_persistence)) ** (1/3)
    volume_price_signal = price_strength_geo_mean * volume_trend_alignment
    
    # Intraday Momentum Convergence
    # Morning Session Analysis
    opening_momentum = (open_price - close.shift(1)) / close.shift(1)
    morning_strength = (high - open_price) / open_price.replace(0, np.nan)
    morning_weakness = (open_price - low) / open_price.replace(0, np.nan)
    
    # Afternoon Session Analysis
    afternoon_momentum = (close - high) / high.replace(0, np.nan)
    afternoon_strength = (close - low) / low.replace(0, np.nan)
    closing_momentum = (close - open_price) / open_price.replace(0, np.nan)
    
    # Intraday Confirmation
    session_alignment = (morning_strength - morning_weakness) * (afternoon_strength - afternoon_momentum)
    intraday_consistency = opening_momentum * closing_momentum
    
    # Combined intraday signal
    intraday_signal = np.sign(session_alignment * intraday_consistency) * \
                     (np.abs(session_alignment * intraday_consistency)) ** (1/2)
    
    # Multi-Timeframe Signal Integration
    # Short-term Framework (1-3 days)
    price_momentum_st = close / close.shift(2) - 1
    volume_acceleration = volume / volume.shift(2)
    range_efficiency = (close - close.shift(1)) / (high - low).replace(0, np.nan)
    
    # Medium-term Framework (5-10 days)
    price_trend_mt = close / close.shift(7) - 1
    volume_trend_mt = volume / volume.rolling(window=6, min_periods=1).mean().shift(1)
    
    high_low_range = high - low
    volatility_persistence = high_low_range / high_low_range.rolling(window=7, min_periods=1).mean()
    
    # Multi-timeframe Convergence
    price_alignment = np.sign(price_momentum_st * price_trend_mt) * \
                     (np.abs(price_momentum_st * price_trend_mt)) ** (1/2)
    volume_alignment = np.sign(volume_acceleration * volume_trend_mt) * \
                      (np.abs(volume_acceleration * volume_trend_mt)) ** (1/2)
    
    volatility_adjusted_price = price_alignment / volatility_persistence.replace(0, np.nan)
    
    # Final Combined Alpha
    # Multi-dimensional convergence using geometric mean of all major components
    alpha_components = pd.DataFrame({
        'vol_momentum': vol_momentum_signal,
        'volume_price': volume_price_signal,
        'intraday': intraday_signal,
        'multi_timeframe': volatility_adjusted_price
    }).fillna(0)
    
    # Geometric mean of absolute values with sign preservation
    def geometric_mean_with_sign(row):
        non_zero_vals = row[row != 0]
        if len(non_zero_vals) == 0:
            return 0
        sign = np.sign(np.prod(non_zero_vals))
        abs_geo_mean = np.exp(np.mean(np.log(np.abs(non_zero_vals))))
        return sign * abs_geo_mean
    
    final_alpha = alpha_components.apply(geometric_mean_with_sign, axis=1)
    
    return final_alpha
