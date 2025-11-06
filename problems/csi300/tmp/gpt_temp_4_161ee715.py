import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining momentum dynamics, volume-price efficiency,
    gap analysis, and microstructure signals.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # 1. Momentum Dynamics
    # Momentum Acceleration
    mom_5d = df['close'].pct_change(5)
    mom_10d = df['close'].pct_change(10)
    momentum_acceleration = mom_5d - mom_10d
    
    # Volatility-Adjusted Momentum
    # Calculate ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close_prev = np.abs(df['high'] - df['close'].shift(1))
    low_close_prev = np.abs(df['low'] - df['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr_10d = true_range.rolling(window=10, min_periods=10).mean()
    
    # Short-term returns scaled by volatility
    ret_3d = df['close'].pct_change(3)
    ret_5d = df['close'].pct_change(5)
    vol_adj_momentum = (ret_3d + ret_5d) / (atr_10d + 1e-8)
    
    # 2. Volume-Price Efficiency
    # Volume-Price Divergence
    price_trend_5d = df['close'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0])
    volume_trend_5d = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / (x[0] + 1e-8))
    volume_price_divergence = price_trend_5d - volume_trend_5d
    
    # Price Efficiency Ratio
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    price_efficiency_ratio = daily_efficiency.rolling(window=5).mean()
    
    # Volume-Price correlation persistence
    volume_price_corr = df['close'].rolling(window=10).corr(df['volume'])
    
    # 3. Gap & Breakout Analysis
    # Gap Reaction Framework
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    intraday_range = (df['high'] - df['low']) / df['open']
    gap_reaction = overnight_gap / (intraday_range + 1e-8)
    
    # Volatility-Adaptive Breakout
    resistance_10d = df['high'].rolling(window=10).max()
    support_10d = df['low'].rolling(window=10).min()
    volatility_10d = df['close'].pct_change().rolling(window=10).std()
    
    # Breakout signals adjusted by volatility
    upper_breakout = (df['close'] - resistance_10d) / (volatility_10d + 1e-8)
    lower_breakout = (df['close'] - support_10d) / (volatility_10d + 1e-8)
    volatility_adaptive_breakout = upper_breakout - lower_breakout
    
    # 4. Microstructure Synthesis
    # Order Flow Imbalance
    range_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    ofi_signal = range_position.rolling(window=5).apply(
        lambda x: 1 if (x > 0.7).sum() > (x < 0.3).sum() else -1 if (x < 0.3).sum() > (x > 0.7).sum() else 0
    )
    
    # Market Efficiency Assessment
    # Price path complexity (simplified as return autocorrelation)
    returns = df['close'].pct_change()
    autocorr_1d = returns.rolling(window=10).apply(lambda x: x.autocorr(lag=1), raw=False)
    
    # Trend quality across timeframes
    trend_quality_short = df['close'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0
    )
    trend_quality_medium = df['close'].rolling(window=10).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0
    )
    trend_quality = trend_quality_short - trend_quality_medium
    
    # Combine all signals with appropriate weights
    alpha = (
        0.15 * momentum_acceleration.rank(pct=True) +
        0.15 * vol_adj_momentum.rank(pct=True) +
        0.12 * volume_price_divergence.rank(pct=True) +
        0.10 * price_efficiency_ratio.rank(pct=True) +
        0.08 * volume_price_corr.rank(pct=True) +
        0.10 * gap_reaction.rank(pct=True) +
        0.12 * volatility_adaptive_breakout.rank(pct=True) +
        0.08 * ofi_signal.rank(pct=True) +
        0.05 * autocorr_1d.rank(pct=True) +
        0.05 * trend_quality.rank(pct=True)
    )
    
    # Final normalization
    alpha = (alpha - alpha.mean()) / (alpha.std() + 1e-8)
    
    return alpha
