import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum with Volume Confirmation
    df = df.copy()
    
    # Calculate 5-day and 10-day price momentum
    momentum_5 = df['close'] / df['close'].shift(5) - 1
    momentum_10 = df['close'] / df['close'].shift(10) - 1
    momentum_ratio = momentum_5 / (momentum_10 + 1e-8)
    
    # Calculate 5-day and 10-day volume trends
    volume_trend_5 = df['volume'] / df['volume'].rolling(5).mean()
    volume_trend_10 = df['volume'] / df['volume'].rolling(10).mean()
    volume_alignment = volume_trend_5 / (volume_trend_10 + 1e-8)
    
    factor1 = momentum_ratio * volume_alignment
    
    # High-Low Range Breakout with Volume Spike
    high_low_range = df['high'].rolling(20).max() - df['low'].rolling(20).min()
    current_range = df['high'] - df['low']
    range_breakout = current_range / (high_low_range + 1e-8)
    
    volume_avg_5 = df['volume'].rolling(5).mean()
    volume_spike = df['volume'] / (volume_avg_5 + 1e-8)
    
    factor2 = range_breakout * volume_spike
    
    # Volatility-Adjusted Return Reversal
    return_3d = df['close'] / df['close'].shift(3) - 1
    
    # True range calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    volatility_10d = true_range.rolling(10).mean()
    
    factor3 = return_3d / (volatility_10d + 1e-8)
    
    # Opening Gap Mean Reversion
    opening_gap = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    gap_mean = opening_gap.rolling(20).mean()
    gap_std = opening_gap.rolling(20).std()
    gap_zscore = (opening_gap - gap_mean) / (gap_std + 1e-8)
    
    factor4 = -gap_zscore  # Negative for mean reversion
    
    # Volume-Weighted Price Acceleration
    price_change_1d = df['close'].pct_change()
    price_acceleration = price_change_1d.diff()
    
    volume_rank = df['volume'].rolling(20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    factor5 = price_acceleration * volume_rank
    
    # Intraday Strength Persistence
    intraday_strength = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    strength_trend = intraday_strength.rolling(5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0], raw=False
    )
    
    factor6 = strength_trend
    
    # Amount-Based Large Trade Impact
    avg_trade_size = df['amount'] / (df['volume'] + 1e-8)
    large_trade_ratio = avg_trade_size / avg_trade_size.rolling(20).mean()
    price_change_impact = df['close'].pct_change() * large_trade_ratio
    
    factor7 = price_change_impact
    
    # Multi-timeframe Volume Divergence
    volume_trend_3d = df['volume'].rolling(3).apply(
        lambda x: np.polyfit(range(3), x, 1)[0], raw=False
    )
    volume_trend_10d = df['volume'].rolling(10).apply(
        lambda x: np.polyfit(range(10), x, 1)[0], raw=False
    )
    price_trend_10d = df['close'].rolling(10).apply(
        lambda x: np.polyfit(range(10), x, 1)[0], raw=False
    )
    
    volume_divergence = volume_trend_3d - volume_trend_10d
    factor8 = volume_divergence * np.sign(price_trend_10d)
    
    # Close-to-VWAP Distance Signal
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    vwap_distance = (df['close'] - vwap) / vwap
    vwap_zscore = vwap_distance / (vwap_distance.rolling(20).std() + 1e-8)
    
    factor9 = -vwap_zscore  # Negative for mean reversion
    
    # High-Frequency Price Elasticity
    intraday_ranges = [
        (df['high'] - df['open']) / (df['open'] + 1e-8),
        (df['close'] - df['open']) / (df['open'] + 1e-8),
        (df['close'] - df['low']) / (df['low'] + 1e-8)
    ]
    
    # Calculate autocorrelation of intraday returns
    def calc_autocorr(series, lag=1):
        return series.autocorr(lag=lag)
    
    autocorr_values = pd.Series([
        calc_autocorr(ret, 1) if len(ret.dropna()) > 10 else 0 
        for ret in intraday_ranges
    ]).mean()
    
    factor10 = pd.Series(autocorr_values, index=df.index)
    
    # Combine all factors with equal weights
    factors = [factor1, factor2, factor3, factor4, factor5, 
               factor6, factor7, factor8, factor9, factor10]
    
    # Standardize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        factor_std = (factor - factor.rolling(50).mean()) / (factor.rolling(50).std() + 1e-8)
        combined_factor += factor_std
    
    return combined_factor
