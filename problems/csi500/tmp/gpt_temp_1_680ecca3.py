import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volatility Regime Adjusted Momentum
    # Calculate Historical Volatility using True Range
    df['tr'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
    hist_vol = df['tr'].rolling(window=20, min_periods=10).mean()
    
    # Calculate Price Momentum
    momentum = df['close'].pct_change(periods=10)
    
    # Calculate Volume trend
    vol_5d = df['volume'].rolling(window=5, min_periods=3).mean()
    vol_20d = df['volume'].rolling(window=20, min_periods=10).mean()
    vol_trend = vol_5d / vol_20d
    
    # Adjust Momentum by Volatility Regime
    vol_regime_momentum = momentum / hist_vol * vol_trend
    
    # Volume-Weighted Price Acceleration
    # First derivative - price velocity
    price_velocity = df['close'].diff(periods=3).rolling(window=5, min_periods=3).mean()
    
    # Second derivative - price acceleration
    price_acceleration = price_velocity.diff(periods=1).rolling(window=3, min_periods=2).mean()
    
    # Volume trend strength using linear regression slope
    def volume_slope(vol_series):
        if len(vol_series) < 2:
            return 0
        x = np.arange(len(vol_series))
        slope, _, _, _, _ = stats.linregress(x, vol_series)
        return slope
    
    vol_trend_strength = df['volume'].rolling(window=10, min_periods=5).apply(
        volume_slope, raw=False
    )
    
    # Weight acceleration by volume persistence
    weighted_acceleration = price_acceleration * vol_trend_strength
    
    # Intraday Strength Persistence
    # Calculate Intraday Strength Ratio
    intraday_range = df['high'] - df['low']
    intraday_range = intraday_range.replace(0, np.nan)  # Avoid division by zero
    strength_ratio = (df['close'] - df['open']) / intraday_range
    
    # Winsorize extreme values (top and bottom 5%)
    strength_ratio_winsorized = strength_ratio.clip(
        lower=strength_ratio.quantile(0.05),
        upper=strength_ratio.quantile(0.95)
    )
    
    # Calculate autocorrelation with lags 1 and 2 over 15 days
    def autocorr_func(series, lag):
        if len(series) < lag + 1:
            return 0
        return series.autocorr(lag=lag)
    
    acf_lag1 = strength_ratio_winsorized.rolling(window=15, min_periods=8).apply(
        lambda x: autocorr_func(x, 1), raw=False
    )
    acf_lag2 = strength_ratio_winsorized.rolling(window=15, min_periods=8).apply(
        lambda x: autocorr_func(x, 2), raw=False
    )
    
    # Persistence measure weighted by consistency
    persistence = (acf_lag1 + acf_lag2) / 2
    vol_confirmation = df['volume'].rolling(window=5, min_periods=3).mean()
    intraday_persistence = persistence * vol_confirmation
    
    # Liquidity-Adjusted Reversal
    # Identify Recent Extreme Moves
    returns_5d = df['close'].pct_change(periods=5)
    returns_std = returns_5d.rolling(window=20, min_periods=10).std()
    extreme_down = returns_5d < (-2 * returns_std)
    extreme_up = returns_5d > (2 * returns_std)
    
    # Calculate Liquidity Conditions
    liquidity_ratio = df['amount'] / df['volume'].replace(0, np.nan)
    liquidity_ratio_ma = liquidity_ratio.rolling(window=10, min_periods=5).mean()
    
    # Compute Conditional Reversal Signal
    reversal_signal = pd.Series(0, index=df.index)
    # Positive reversal for extreme down moves with high liquidity
    reversal_signal[extreme_down] = liquidity_ratio_ma[extreme_down]
    # Negative reversal for extreme up moves with low liquidity
    reversal_signal[extreme_up] = -1 / liquidity_ratio_ma[extreme_up].replace(0, np.nan)
    
    # Pressure Accumulation Divergence
    # Calculate Buying and Selling Pressure
    up_days = df['close'] > df['open']
    down_days = df['close'] < df['open']
    
    buying_pressure = (df['volume'] * up_days).rolling(window=10, min_periods=5).sum()
    selling_pressure = (df['volume'] * down_days).rolling(window=10, min_periods=5).sum()
    
    # Pressure difference normalized by total volume
    total_volume_10d = df['volume'].rolling(window=10, min_periods=5).sum()
    pressure_diff = (buying_pressure - selling_pressure) / total_volume_10d.replace(0, np.nan)
    
    # Price trend using linear regression slope
    def price_slope(price_series):
        if len(price_series) < 2:
            return 0
        x = np.arange(len(price_series))
        slope, _, _, _, _ = stats.linregress(x, price_series)
        return slope
    
    price_trend = df['close'].rolling(window=10, min_periods=5).apply(
        price_slope, raw=False
    )
    
    # Divergence signal when pressure and price trend move in opposite directions
    divergence_signal = pressure_diff * np.sign(pressure_diff) * np.sign(-price_trend)
    
    # Combine all factors with equal weights
    combined_factor = (
        vol_regime_momentum.fillna(0) +
        weighted_acceleration.fillna(0) +
        intraday_persistence.fillna(0) +
        reversal_signal.fillna(0) +
        divergence_signal.fillna(0)
    )
    
    return combined_factor
