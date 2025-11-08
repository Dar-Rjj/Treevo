import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor combining volatility-scaled momentum, volume-price divergence,
    intraday confirmation signals, and multi-timeframe convergence.
    """
    df = data.copy()
    
    # Volatility-Scaled Momentum
    # Multi-Timeframe Momentum Alignment
    mom_3d = df['close'] / df['close'].shift(3) - 1
    mom_5d = df['close'] / df['close'].shift(5) - 1
    mom_10d = df['close'] / df['close'].shift(10) - 1
    
    # Volatility Scaling
    ret = df['close'].pct_change()
    vol_3d = ret.rolling(window=3).std()
    vol_5d = ret.rolling(window=5).std()
    vol_10d = ret.rolling(window=10).std()
    
    # Combined Signal
    momentum_convergence = np.sign(mom_3d * mom_5d * mom_10d) * (np.abs(mom_3d * mom_5d * mom_10d))**(1/3)
    vol_scaling = (vol_3d * vol_5d * vol_10d)**(1/3)
    vol_scaled_momentum = momentum_convergence / (vol_scaling + 1e-8)
    
    # Volume-Price Divergence
    # Price Strength Indicators
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    price_positioning = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    momentum_confirmation = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8)
    
    # Volume Divergence Metrics
    volume_ratio = df['volume'] / (df['volume'].shift(1) * df['volume'].shift(2) * df['volume'].shift(3))**(1/3)
    volume_acceleration = (df['volume'] / df['volume'].shift(1)) * (df['volume'].shift(1) / df['volume'].shift(2))
    volume_persistence = df['volume'] / (df['volume'].shift(4) * df['volume'].shift(5) * df['volume'].shift(6))**(1/3)
    
    # Combined Signal
    core_price_strength = np.sign(intraday_efficiency * price_positioning * momentum_confirmation) * \
                         (np.abs(intraday_efficiency * price_positioning * momentum_confirmation))**(1/3)
    volume_confirmation = np.sign(volume_ratio * volume_acceleration * volume_persistence) * \
                         (np.abs(volume_ratio * volume_acceleration * volume_persistence))**(1/3)
    volume_price_divergence = core_price_strength * volume_confirmation
    
    # Intraday Confirmation Signals
    # Morning Session Analysis
    opening_momentum = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    morning_strength = (df['high'] - df['open']) / (df['open'] + 1e-8)
    morning_support = (df['open'] - df['low']) / (df['open'] + 1e-8)
    
    # Afternoon Session Analysis
    afternoon_momentum = (df['close'] - df['high']) / (df['high'] + 1e-8)
    afternoon_resilience = (df['close'] - df['low']) / (df['low'] + 1e-8)
    closing_conviction = (df['close'] - df['open']) / (df['open'] + 1e-8)
    
    # Combined Signal
    session_alignment = (morning_strength - morning_support) * (afternoon_momentum - afternoon_resilience)
    intraday_consistency = np.sign(opening_momentum * closing_conviction) * \
                          (np.abs(opening_momentum * closing_conviction))**(1/2)
    intraday_confirmation = session_alignment * intraday_consistency
    
    # Multi-Timeframe Convergence
    # Short-term Framework (1-3 days)
    price_momentum_st = np.sign(df['close'] / df['close'].shift(1) * df['close'] / df['close'].shift(2)) * \
                       (np.abs(df['close'] / df['close'].shift(1) * df['close'] / df['close'].shift(2)))**(1/2) - 1
    volume_momentum_st = np.sign(df['volume'] / df['volume'].shift(1) * df['volume'] / df['volume'].shift(2)) * \
                        (np.abs(df['volume'] / df['volume'].shift(1) * df['volume'] / df['volume'].shift(2)))**(1/2)
    
    range_efficiency_st = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)).rolling(window=3).apply(
        lambda x: np.sign(x.prod()) * (np.abs(x.prod()))**(1/3), raw=False
    )
    
    # Medium-term Framework (5-10 days)
    price_trend_mt = np.sign(df['close'] / df['close'].shift(5) * df['close'] / df['close'].shift(10)) * \
                    (np.abs(df['close'] / df['close'].shift(5) * df['close'] / df['close'].shift(10)))**(1/2) - 1
    
    volume_avg_5d = df['volume'].rolling(window=5).mean()
    volume_avg_10d = df['volume'].rolling(window=10).mean()
    volume_trend_mt = np.sign(df['volume'] / volume_avg_5d * df['volume'] / volume_avg_10d) * \
                     (np.abs(df['volume'] / volume_avg_5d * df['volume'] / volume_avg_10d))**(1/2)
    
    daily_ranges = (df['high'] - df['low']) / df['close']
    vol_consistency = daily_ranges.rolling(window=10).apply(
        lambda x: np.sign(x[5:].prod()) * (np.abs(x[5:].prod()))**(1/5), raw=False
    ) / (daily_ranges.rolling(window=5).apply(
        lambda x: np.sign(x.prod()) * (np.abs(x.prod()))**(1/5), raw=False
    ) + 1e-8)
    
    # Combined Signal
    timeframe_convergence = price_momentum_st * price_trend_mt
    volume_alignment = np.sign(volume_momentum_st * volume_trend_mt) * \
                      (np.abs(volume_momentum_st * volume_trend_mt))**(1/2)
    multi_timeframe = timeframe_convergence * volume_alignment / (vol_consistency + 1e-8)
    
    # Final Alpha Factor Combination
    alpha_factor = (
        vol_scaled_momentum.rank(pct=True) + 
        volume_price_divergence.rank(pct=True) + 
        intraday_confirmation.rank(pct=True) + 
        multi_timeframe.rank(pct=True)
    ) / 4
    
    return alpha_factor
