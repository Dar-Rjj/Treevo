import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Volume-Price Divergence Momentum
    # 5-day volume slope
    volume_slope = df['volume'].rolling(window=5).apply(
        lambda x: linregress(range(5), x)[0] if not x.isna().any() else np.nan, 
        raw=False
    )
    
    # 10-day price ROC sign and magnitude
    price_roc = (df['close'] / df['close'].shift(10) - 1)
    price_roc_sign_mag = np.sign(price_roc) * np.abs(price_roc)
    
    # High-Low Range Efficiency
    # True Range calculation
    tr1 = df['high'] - df['low']
    tr2 = np.abs(df['high'] - df['close'].shift(1))
    tr3 = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Efficiency Ratio
    price_change = np.abs(df['close'] - df['close'].shift(5))
    efficiency_ratio = price_change / (true_range.rolling(window=5).sum())
    
    # Volume Surge Volatility Adjustment
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_surge = df['volume'] / volume_ma
    
    # High-low volatility
    hl_volatility = (df['high'] - df['low']).rolling(window=10).std()
    
    # Pressure Reversal Detection
    midpoint = (df['high'] + df['low']) / 2
    pressure_index = (df['close'] - midpoint) * df['volume']
    pressure_trend = pressure_index.rolling(window=5).apply(
        lambda x: linregress(range(5), x)[0] if not x.isna().any() else np.nan,
        raw=False
    )
    
    # Liquidity-Weighted Returns
    returns_5 = df['close'].pct_change(5)
    returns_10 = df['close'].pct_change(10)
    returns_20 = df['close'].pct_change(20)
    
    # Turnover efficiency
    turnover_efficiency = df['amount'] / (df['volume'] * df['close'])
    liquidity_weighted_returns = (returns_5 + returns_10 + returns_20) * turnover_efficiency
    
    # Gap Fill Dynamics
    overnight_gap = df['open'] / df['close'].shift(1) - 1
    gap_size = np.abs(overnight_gap)
    gap_fill_prob = 1 / (1 + gap_size * df['volume'].rolling(window=5).std())
    
    # Volume-Cluster Support/Resistance
    # Volume concentration at extremes
    high_volume_ratio = df['volume'] / df['volume'].rolling(window=20).quantile(0.8)
    low_volume_ratio = df['volume'] / df['volume'].rolling(window=20).quantile(0.2)
    volume_extremes = np.where(df['close'] > df['close'].rolling(window=20).mean(), 
                              high_volume_ratio, low_volume_ratio)
    
    # Momentum Decay Characteristics
    rolling_returns = df['close'].pct_change(10).rolling(window=10)
    decay_factor = rolling_returns.apply(
        lambda x: np.exp(-np.abs(np.mean(x))) if not x.isna().any() else np.nan,
        raw=False
    )
    
    # Volume-Volatility Relationship Break
    volume_vol_corr = df['volume'].rolling(window=20).corr(hl_volatility)
    corr_change = volume_vol_corr.diff(5)
    
    # Efficiency-Enhanced Trend
    price_trend = df['close'].rolling(window=10).apply(
        lambda x: linregress(range(10), x)[0] if not x.isna().any() else np.nan,
        raw=False
    )
    noise_ratio = true_range.rolling(window=10).std() / price_trend.rolling(window=10).std()
    volume_trend_impact = volume_slope * (1 - noise_ratio)
    
    # Combine all components
    factor = (
        volume_slope * price_roc_sign_mag * 0.15 +
        efficiency_ratio * volume_surge * 0.12 +
        pressure_trend * hl_volatility * 0.10 +
        liquidity_weighted_returns * 0.18 +
        gap_fill_prob * volume_extremes * 0.11 +
        decay_factor * corr_change * 0.14 +
        price_trend * volume_trend_impact * 0.20
    )
    
    return factor
