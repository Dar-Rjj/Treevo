import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum with Volume Confirmation
    df = df.copy()
    
    # Calculate Price Momentum
    close_5d_momentum = df['close'] / df['close'].shift(5) - 1
    close_10d_momentum = df['close'] / df['close'].shift(10) - 1
    momentum_ratio = close_5d_momentum / close_10d_momentum
    
    # Confirm with Volume Trend
    volume_5d_trend = df['volume'] / df['volume'].rolling(window=5).mean()
    volume_10d_trend = df['volume'] / df['volume'].rolling(window=10).mean()
    volume_alignment = np.sign(volume_5d_trend) * np.sign(volume_10d_trend)
    factor1 = momentum_ratio * volume_alignment
    
    # High-Low Range Breakout with Volume Spike
    high_low_range = df['high'] - df['low']
    range_20d_avg = high_low_range.rolling(window=20).mean()
    range_breakout = (high_low_range > range_20d_avg * 1.1).astype(int)
    
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_spike = (df['volume'] > volume_5d_avg * 1.2).astype(int)
    factor2 = range_breakout * volume_spike
    
    # Volatility-Adjusted Return Reversal
    return_3d = df['close'].pct_change(3)
    high_low_range_daily = df['high'] - df['low']
    true_range_10d = high_low_range_daily.rolling(window=10).mean()
    volatility_adjusted_return = return_3d / (true_range_10d + 1e-8)
    factor3 = -volatility_adjusted_return  # Reversal signal
    
    # Opening Gap Mean Reversion
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_20d_mean = opening_gap.rolling(window=20).mean()
    gap_20d_std = opening_gap.rolling(window=20).std()
    extreme_gap = np.abs(opening_gap - gap_20d_mean) > 1.5 * gap_20d_std
    gap_reversion = -opening_gap * extreme_gap
    factor4 = gap_reversion
    
    # Volume-Weighted Price Acceleration
    price_velocity = df['close'].pct_change()
    price_acceleration = price_velocity.diff()
    
    volume_rank = df['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    factor5 = price_acceleration * volume_rank
    
    # Intraday Strength Persistence
    intraday_strength = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    strength_persistence = intraday_strength.rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0, 
        raw=False
    )
    factor6 = strength_persistence
    
    # Amount-Based Large Trade Impact
    avg_trade_size = df['amount'] / (df['volume'] + 1e-8)
    large_trade_impact = avg_trade_size.rolling(window=10).apply(
        lambda x: np.corrcoef(x, df['close'].loc[x.index].pct_change().fillna(0))[0,1] 
        if len(x) > 1 and np.std(x) > 0 else 0,
        raw=False
    )
    factor7 = large_trade_impact
    
    # Multi-timeframe Volume Divergence
    volume_3d_trend = df['volume'].rolling(window=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    volume_10d_trend = df['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    price_3d_trend = df['close'].rolling(window=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    
    volume_divergence = np.sign(volume_3d_trend) * np.sign(volume_10d_trend) * np.sign(price_3d_trend)
    factor8 = volume_divergence
    
    # Close-to-VWAP Distance Signal
    vwap = (df['amount'].cumsum() / df['volume'].cumsum()).fillna(method='ffill')
    vwap_deviation = (df['close'] - vwap) / vwap
    vwap_reversion = -vwap_deviation
    factor9 = vwap_reversion
    
    # High-Frequency Price Elasticity
    intraday_returns = (df['close'] - df['open']) / df['open']
    high_freq_elasticity = intraday_returns.rolling(window=5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 and np.std(x) > 0 else 0,
        raw=False
    )
    factor10 = high_freq_elasticity
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'f1': factor1, 'f2': factor2, 'f3': factor3, 'f4': factor4, 'f5': factor5,
        'f6': factor6, 'f7': factor7, 'f8': factor8, 'f9': factor9, 'f10': factor10
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=20).mean()) / (x.rolling(window=20).std() + 1e-8))
    
    # Equal-weighted combination
    combined_factor = factors_normalized.mean(axis=1)
    
    return combined_factor
