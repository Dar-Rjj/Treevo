import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Adjusted Momentum
    # Short-Term Return (t-5 to t)
    short_term_return = df['close'] / df['close'].shift(5) - 1
    
    # Medium-Term Return (t-20 to t)
    medium_term_return = df['close'] / df['close'].shift(20) - 1
    
    # Recent Volatility (t-20 to t-1) - using past 20 days excluding current day
    returns_20d = df['close'].pct_change().shift(1).rolling(window=20, min_periods=10).std()
    
    # Volatility-adjusted momentum components
    vol_adj_short = short_term_return / (returns_20d + 1e-8)
    vol_adj_medium = medium_term_return / (returns_20d + 1e-8)
    
    # Weighted momentum (more weight to short-term)
    weighted_momentum = 0.6 * vol_adj_short + 0.4 * vol_adj_medium
    
    # Volume Divergence
    # Volume Trend (t-5 to t) - linear regression slope
    def volume_slope(volume_series):
        if len(volume_series) < 2:
            return np.nan
        x = np.arange(len(volume_series))
        return np.polyfit(x, volume_series, 1)[0]
    
    volume_trend = df['volume'].rolling(window=6, min_periods=3).apply(volume_slope, raw=True)
    
    # Price-Volume Correlation (t-20 to t)
    def price_volume_corr(window_df):
        if len(window_df) < 5:
            return np.nan
        price_returns = window_df['close'].pct_change().dropna()
        volume_changes = window_df['volume'].pct_change().dropna()
        if len(price_returns) < 3 or len(volume_changes) < 3:
            return np.nan
        min_len = min(len(price_returns), len(volume_changes))
        return np.corrcoef(price_returns.iloc[:min_len], volume_changes.iloc[:min_len])[0, 1]
    
    # Create rolling window for correlation calculation
    price_volume_corr_series = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window_df = df.iloc[i-20:i+1]
        price_volume_corr_series.iloc[i] = price_volume_corr(window_df)
    
    # Volume divergence score
    volume_divergence = volume_trend * price_volume_corr_series
    
    # Volume multiplier (normalized and bounded)
    volume_multiplier = volume_divergence.rolling(window=20, min_periods=10).apply(
        lambda x: np.tanh(np.mean(x)) if not np.isnan(np.mean(x)) else 0, raw=False
    )
    
    # Combined Factor
    combined_factor = weighted_momentum * (1 + volume_multiplier)
    
    return combined_factor
