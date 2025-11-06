import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price Momentum with Volume Confirmation
    # Calculate 5-day and 10-day close price momentum ratios
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    momentum_10d = data['close'] / data['close'].shift(10) - 1
    momentum_ratio = momentum_5d / (momentum_10d + 1e-8)
    
    # Volume trend alignment (5-day vs 10-day)
    vol_5d_avg = data['volume'].rolling(window=5).mean()
    vol_10d_avg = data['volume'].rolling(window=10).mean()
    volume_alignment = vol_5d_avg / (vol_10d_avg + 1e-8)
    
    momentum_factor = momentum_ratio * volume_alignment
    
    # 2. High-Low Range Breakout Detection
    # Identify breakouts from 20-day high-low range
    high_20d = data['high'].rolling(window=20).max()
    low_20d = data['low'].rolling(window=20).min()
    range_20d = high_20d - low_20d
    
    # Breakout signal: close above 20-day high or below 20-day low
    breakout_high = (data['close'] > high_20d.shift(1)).astype(int)
    breakout_low = (data['close'] < low_20d.shift(1)).astype(int)
    breakout_signal = breakout_high - breakout_low
    
    # Volume spike relative to 5-day average
    vol_spike = data['volume'] / (data['volume'].rolling(window=5).mean() + 1e-8)
    breakout_factor = breakout_signal * vol_spike
    
    # 3. Volatility-Adjusted Return Reversal
    # Compute 3-day close price returns
    ret_3d = data['close'].pct_change(3)
    
    # Calculate 10-day true range volatility
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    vol_10d = true_range.rolling(window=10).mean()
    
    volatility_adjusted_factor = ret_3d / (vol_10d + 1e-8)
    
    # 4. Opening Gap Mean Reversion
    # Calculate open/close gap percentages
    gap_pct = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Identify extreme deviations from 20-day gap patterns
    gap_20d_mean = gap_pct.rolling(window=20).mean()
    gap_20d_std = gap_pct.rolling(window=20).std()
    gap_zscore = (gap_pct - gap_20d_mean) / (gap_20d_std + 1e-8)
    
    gap_reversion_factor = -gap_zscore  # Negative for mean reversion
    
    # 5. Volume-Weighted Price Acceleration
    # Calculate second derivative of close prices
    price_change_1d = data['close'].pct_change()
    price_acceleration = price_change_1d.diff()
    
    # Weight by volume percentile ranking (20-day window)
    volume_rank = data['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    acceleration_factor = price_acceleration * volume_rank
    
    # 6. Intraday Strength Persistence
    # Measure position within daily range (open-high-close)
    daily_range = data['high'] - data['low']
    intraday_position = (data['close'] - data['low']) / (daily_range + 1e-8)
    
    # Analyze 5-day strength sequence patterns
    strength_5d = intraday_position.rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0, raw=False
    )
    strength_factor = strength_5d.fillna(0)
    
    # 7. Large Trade Impact Analysis
    # Calculate average trade size from amount/volume
    avg_trade_size = data['amount'] / (data['volume'] + 1e-8)
    
    # Correlate with concurrent price movements (5-day window)
    def trade_impact_corr(x):
        if len(x) < 2:
            return 0
        prices = data.loc[x.index, 'close'].pct_change().fillna(0)
        trade_sizes = x.values
        if len(prices) == len(trade_sizes):
            return np.corrcoef(prices, trade_sizes)[0,1] if not np.isnan(np.corrcoef(prices, trade_sizes)[0,1]) else 0
        return 0
    
    trade_impact = avg_trade_size.rolling(window=5).apply(
        trade_impact_corr, raw=False
    )
    
    # 8. Multi-timeframe Volume Divergence
    # Compare 3-day vs 10-day volume trends
    vol_3d_trend = data['volume'].rolling(window=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    vol_10d_trend = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False
    )
    
    volume_divergence = vol_3d_trend - vol_10d_trend
    
    # 9. VWAP Deviation Signal
    # Calculate volume-weighted average price
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
    
    # Measure close price distance from VWAP
    vwap_deviation = (data['close'] - vwap) / (vwap + 1e-8)
    vwap_factor = vwap_deviation
    
    # 10. High-Frequency Price Elasticity
    # Analyze micro-structure return sequences (using 5-period momentum persistence)
    returns_5min = data['close'].pct_change().rolling(window=5)
    momentum_persistence = returns_5min.apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) > 0 else 0, raw=False
    )
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'momentum': momentum_factor,
        'breakout': breakout_factor,
        'vol_adj': volatility_adjusted_factor,
        'gap_rev': gap_reversion_factor,
        'accel': acceleration_factor,
        'strength': strength_factor,
        'trade_impact': trade_impact,
        'vol_div': volume_divergence,
        'vwap': vwap_factor,
        'momentum_persist': momentum_persistence
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=50).mean()) / (x.rolling(window=50).std() + 1e-8))
    
    # Equal-weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
