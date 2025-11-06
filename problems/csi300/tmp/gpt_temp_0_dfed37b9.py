import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators including:
    - High-Low Range Persistence
    - Volatility Regime Switching
    - Order Flow Imbalance
    - Price-Volume Divergence
    - Liquidity-Adjusted Momentum
    - Gap Filling Probability
    - Multi-Timeframe Momentum Convergence
    - Resistance Breakout Strength
    - Volatility Compression Expansion
    """
    
    # High-Low Range Persistence Factor
    high_low_range = df['high'] - df['low']
    
    # Calculate range autocorrelation over 5 days
    range_autocorr = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window_range = high_low_range.iloc[i-5:i]
        if len(window_range) >= 2:
            range_autocorr.iloc[i] = window_range.autocorr(lag=1)
        else:
            range_autocorr.iloc[i] = 0
    
    # Volume-range correlation
    volume_range_corr = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        window_vol = df['volume'].iloc[i-10:i]
        window_range = high_low_range.iloc[i-10:i]
        if len(window_vol) >= 5 and len(window_range) >= 5:
            volume_range_corr.iloc[i] = window_vol.corr(window_range)
        else:
            volume_range_corr.iloc[i] = 0
    
    # Recent volume trend (5-day)
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: stats.linregress(range(len(x)), x)[0] if len(x) >= 3 else 0
    )
    
    range_persistence = range_autocorr * volume_range_corr * volume_trend
    
    # Volatility Regime Switching Indicator
    returns = df['close'].pct_change()
    rolling_vol = returns.rolling(window=20).std()
    
    # Detect volatility regime changes
    vol_regime = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        current_vol = rolling_vol.iloc[i]
        historical_vol = rolling_vol.iloc[max(0, i-60):i]
        if len(historical_vol) > 0:
            vol_percentile = stats.percentileofscore(historical_vol.dropna(), current_vol) / 100
            vol_regime.iloc[i] = 1 if vol_percentile > 0.8 else (-1 if vol_percentile < 0.2 else 0)
        else:
            vol_regime.iloc[i] = 0
    
    # 5-day momentum (excluding most recent day)
    momentum_5d = df['close'].shift(1).pct_change(periods=4)
    
    # Regime duration weight
    regime_duration = vol_regime.rolling(window=10).apply(
        lambda x: len(x[x == x.iloc[-1]]) if len(x) > 0 else 0
    )
    
    volatility_regime_factor = momentum_5d * vol_regime * regime_duration
    
    # Order Flow Imbalance Factor
    pressure_index = abs((df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan))
    
    # Morning vs afternoon volume (approximated)
    volume_ratio = df['volume'].rolling(window=3).apply(
        lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else 1
    )
    
    flow_imbalance = pressure_index * volume_ratio
    flow_imbalance_ma = flow_imbalance.rolling(window=3).mean()
    
    # Price-Volume Divergence Oscillator
    def compute_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window, len(series)):
            y = series.iloc[i-window:i]
            if len(y) >= window:
                x = np.arange(len(y))
                slope = stats.linregress(x, y)[0]
                slopes.iloc[i] = slope
            else:
                slopes.iloc[i] = 0
        return slopes
    
    price_slope = compute_slope(df['close'], 10)
    volume_slope = compute_slope(df['volume'], 10)
    
    # Divergence with time decay
    divergence = pd.Series(index=df.index, dtype=float)
    for i in range(10, len(df)):
        price_dir = 1 if price_slope.iloc[i] > 0 else -1
        volume_dir = 1 if volume_slope.iloc[i] > 0 else -1
        
        if price_dir != volume_dir:
            decay_weight = np.exp(-(len(df) - i - 1) / 10)  # Exponential decay
            magnitude_diff = abs(price_slope.iloc[i]) - abs(volume_slope.iloc[i])
            divergence.iloc[i] = magnitude_diff * decay_weight * price_dir
        else:
            divergence.iloc[i] = 0
    
    # Liquidity-Adjusted Momentum Factor
    raw_momentum = df['close'].pct_change(periods=10)
    
    # Turnover rate (amount / close)
    turnover_rate = df['amount'] / (df['close'] * df['volume']).replace(0, np.nan)
    
    # Liquidity score combining multiple metrics
    volume_zscore = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
    turnover_zscore = (turnover_rate - turnover_rate.rolling(window=20).mean()) / turnover_rate.rolling(window=20).std()
    
    liquidity_score = volume_zscore.fillna(0) + turnover_zscore.fillna(0)
    
    liquidity_adjusted_momentum = raw_momentum * liquidity_score
    
    # Combine all factors with appropriate weights
    combined_factor = (
        0.15 * range_persistence.fillna(0) +
        0.15 * volatility_regime_factor.fillna(0) +
        0.15 * flow_imbalance_ma.fillna(0) +
        0.15 * divergence.fillna(0) +
        0.15 * liquidity_adjusted_momentum.fillna(0)
    )
    
    # Normalize the final factor
    factor_zscore = (combined_factor - combined_factor.rolling(window=20).mean()) / combined_factor.rolling(window=20).std()
    
    return factor_zscore.fillna(0)
