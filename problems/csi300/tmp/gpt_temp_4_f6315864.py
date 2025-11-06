import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Dynamic Volatility-Adjusted Momentum
    # Calculate Momentum
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    open_price = df['open']
    
    # Calculate 10-day price change
    momentum_10d = close.pct_change(10)
    
    # Calculate 10-day Average True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10d = tr.rolling(window=10).mean()
    
    # Volatility-adjusted momentum
    vol_adj_momentum = momentum_10d / atr_10d
    
    # Volume confirmation with 10-day volume z-score
    volume_mean_10d = volume.rolling(window=10).mean()
    volume_std_10d = volume.rolling(window=10).std()
    volume_zscore = (volume - volume_mean_10d) / volume_std_10d
    
    # Final dynamic volatility-adjusted momentum
    factor1 = vol_adj_momentum * volume_zscore
    
    # Volatility Regime Adaptive Factor
    # Calculate 20-day realized volatility
    returns_20d = close.pct_change().rolling(window=20).std()
    
    # Calculate 60-day historical volatility
    hist_vol_60d = close.pct_change().rolling(window=60).std()
    
    # Identify volatility regime (1 for high, 0 for low)
    vol_regime = (returns_20d > hist_vol_60d).astype(int)
    
    # High volatility regime components
    returns_5d = close.pct_change(5)
    mean_reversion = -returns_5d  # Inverse weighting for mean reversion
    
    # Low volatility regime components
    returns_20d_momentum = close.pct_change(20)
    volume_20d_avg = volume.rolling(window=20).mean()
    volume_breakout = volume / volume_20d_avg
    
    # Combine regime-specific signals
    high_vol_signal = mean_reversion / returns_20d  # Volatility scaling
    low_vol_signal = returns_20d_momentum * volume_breakout
    
    # Final volatility regime adaptive factor
    factor2 = vol_regime * high_vol_signal + (1 - vol_regime) * low_vol_signal
    
    # Intraday Pressure Accumulation
    # Calculate normalized intraday range
    intraday_range = (high - low) / open_price
    
    # Remove outliers using 20-day median
    range_median_20d = intraday_range.rolling(window=20).median()
    range_std_20d = intraday_range.rolling(window=20).std()
    normalized_range = (intraday_range - range_median_20d) / range_std_20d
    
    # Close-to-open gap
    gap = (close - open_price) / open_price
    
    # Intraday pressure
    intraday_pressure = normalized_range * gap
    
    # Exponential decay accumulation
    decay_factor = 0.9
    pressure_accumulation = intraday_pressure.ewm(alpha=1-decay_factor).mean()
    
    # Volume trend (5-day slope)
    volume_trend = volume.rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Final intraday pressure accumulation
    factor3 = pressure_accumulation * volume_trend
    
    # Liquidity-Adjusted Reversal
    # Short-term reversal (2-day return with reversed sign)
    reversal_2d = -close.pct_change(2)
    
    # Calculate average trade size
    trade_size = amount / volume
    avg_trade_size_5d = trade_size.rolling(window=5).mean()
    
    # Scale reversal by trade size (inverse relationship)
    factor4 = reversal_2d / avg_trade_size_5d
    
    # Trend Persistence Score
    # Short-term trend (3-day slope)
    def calc_slope(x):
        if len(x) < 3:
            return np.nan
        return np.polyfit(range(len(x)), x, 1)[0]
    
    trend_3d = close.rolling(window=3).apply(calc_slope, raw=True)
    volume_3d = volume.rolling(window=3).apply(calc_slope, raw=True)
    short_term_trend = trend_3d * volume_3d
    
    # Medium-term trend (10-day slope with volatility adjustment)
    trend_10d = close.rolling(window=10).apply(calc_slope, raw=True)
    vol_10d = close.pct_change().rolling(window=10).std()
    medium_term_trend = trend_10d / vol_10d
    
    # Long-term trend (30-day slope with consistency)
    trend_30d = close.rolling(window=30).apply(calc_slope, raw=True)
    # Consistency measure: ratio of positive to negative returns
    returns_30d = close.pct_change(30)
    consistency = (returns_30d > 0).astype(int).rolling(window=30).mean()
    long_term_trend = trend_30d * consistency
    
    # Combine trend signals with weights
    combined_trend = (0.3 * short_term_trend + 
                      0.5 * medium_term_trend + 
                      0.2 * long_term_trend)
    
    # Volume trend consistency
    volume_trend_consistency = volume.rolling(window=10).apply(
        lambda x: len([i for i in range(1, len(x)) if x[i] > x[i-1]]) / (len(x)-1)
        if len(x) > 1 else np.nan
    )
    
    factor5 = combined_trend * volume_trend_consistency
    
    # Price-Volume Divergence Detection
    # Price momentum
    price_roc_5d = close.pct_change(5)
    price_roc_10d = close.pct_change(10)
    
    # Volume momentum
    volume_roc_5d = volume.pct_change(5)
    volume_roc_10d = volume.pct_change(10)
    
    # Calculate correlation between price and volume
    def rolling_corr(x, y, window):
        return x.rolling(window=window).corr(y)
    
    # Short-term price-volume correlation
    pv_corr_5d = rolling_corr(price_roc_5d, volume_roc_5d, 10)
    
    # Long-term price-volume correlation
    pv_corr_10d = rolling_corr(price_roc_10d, volume_roc_10d, 20)
    
    # Divergence signal
    divergence_signal = pv_corr_5d - pv_corr_10d
    
    # Alignment between short-term and long-term momentum
    momentum_alignment = (price_roc_5d * price_roc_10d > 0).astype(int)
    
    factor6 = divergence_signal * momentum_alignment
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'factor1': factor1,
        'factor2': factor2,
        'factor3': factor3,
        'factor4': factor4,
        'factor5': factor5,
        'factor6': factor6
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Equal weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
