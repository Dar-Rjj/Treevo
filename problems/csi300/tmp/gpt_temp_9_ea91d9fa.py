import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Adjusted Momentum
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Historical volatility (20-day rolling std of True Range)
    hist_vol = true_range.rolling(window=20, min_periods=10).std()
    vol_regime = hist_vol > hist_vol.rolling(window=60, min_periods=30).median()
    
    # Price momentum across different horizons
    mom_5 = df['close'] / df['close'].shift(5) - 1
    mom_10 = df['close'] / df['close'].shift(10) - 1
    mom_20 = df['close'] / df['close'].shift(20) - 1
    
    # Adjust momentum by volatility regime (amplify in low vol, dampen in high vol)
    vol_adjusted_mom = (mom_5 + mom_10 + mom_20) / 3
    vol_adjusted_mom = np.where(vol_regime, vol_adjusted_mom * 0.7, vol_adjusted_mom * 1.3)
    
    # Volume-Price Divergence Strength
    # Price trend (10-day linear regression slope)
    def linear_slope(series, window):
        x = np.arange(window)
        slopes = series.rolling(window=window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if len(y) == window else np.nan, 
            raw=True
        )
        return slopes
    
    price_trend = linear_slope(df['close'], 10)
    volume_trend = linear_slope(df['volume'], 10)
    
    # Volume-price correlation (10-day rolling)
    def rolling_corr(x, y, window):
        return x.rolling(window=window).corr(y)
    
    vp_corr = rolling_corr(df['close'], df['volume'], 10)
    
    # Divergence strength
    divergence_strength = (abs(price_trend) * abs(volume_trend)) * (1 - vp_corr.abs())
    
    # Intraday Range Efficiency
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    price_efficiency = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    volume_percentile = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, 
        raw=False
    )
    range_efficiency = daily_range * price_efficiency * volume_percentile
    
    # Liquidity-Adjusted Reversal
    short_term_reversal = -df['close'].pct_change(1)
    liquidity_measure = df['volume'] * abs(df['close'] - df['close'].shift(1))
    liquidity_percentile = liquidity_measure.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0,
        raw=False
    )
    # Stronger reversal signals in illiquid conditions (low liquidity percentile)
    liquidity_adjusted_reversal = short_term_reversal * (1 - liquidity_percentile)
    
    # Opening Gap Persistence
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_filled = np.where(
        (opening_gap > 0) & (df['low'] <= df['close'].shift(1)), True,
        np.where((opening_gap < 0) & (df['high'] >= df['close'].shift(1)), True, False)
    )
    # Unfilled gaps predict continuation, filled gaps predict reversal
    gap_persistence = opening_gap * np.where(gap_filled, -1, 1)
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'vol_mom': vol_adjusted_mom,
        'divergence': divergence_strength,
        'range_eff': range_efficiency,
        'liq_rev': liquidity_adjusted_reversal,
        'gap_persist': gap_persistence
    })
    
    # Z-score normalization for each factor
    normalized_factors = factors.apply(
        lambda x: (x - x.rolling(window=60, min_periods=30).mean()) / 
                  x.rolling(window=60, min_periods=30).std()
    )
    
    # Final alpha factor (equal weighted combination)
    alpha = normalized_factors.mean(axis=1)
    
    return alpha
