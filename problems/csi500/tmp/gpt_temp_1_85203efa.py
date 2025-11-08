import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multi-timeframe momentum, volume-price efficiency,
    intraday session momentum, and multi-timeframe convergence signals.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Alignment
    # Short-term Momentum (2-day)
    short_price_momentum = data['close'] / data['close'].shift(2) - 1
    intraday_confirmation = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_confirmation = intraday_confirmation.replace([np.inf, -np.inf], np.nan)
    
    # Medium-term Momentum (5-day)
    medium_price_momentum = data['close'] / data['close'].shift(5) - 1
    range_efficiency = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    range_efficiency = range_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Long-term Momentum (10-day)
    long_price_momentum = data['close'] / data['close'].shift(10) - 1
    
    # Trend persistence using geometric mean of daily returns
    daily_returns = data['close'].pct_change()
    trend_persistence = (1 + daily_returns).rolling(window=9).apply(
        lambda x: x.prod() ** (1/9) if not x.isna().any() else np.nan
    ) - 1
    
    # Combined momentum signal
    momentum_alignment = (short_price_momentum * medium_price_momentum * long_price_momentum) ** (1/3)
    intraday_multiplier = (intraday_confirmation * range_efficiency) ** (1/2)
    volatility_scaling = (data['high'] - data['low']) / data['close']
    
    momentum_factor = momentum_alignment * intraday_multiplier / volatility_scaling
    
    # Volume-Price Efficiency Factor
    opening_efficiency = (data['close'] - data['open']) / data['open']
    range_utilization = (data['close'] - data['low']) / (data['high'] - data['low'])
    range_utilization = range_utilization.replace([np.inf, -np.inf], np.nan)
    
    momentum_persistence = (data['close'] / data['close'].shift(1) - 1) * (data['close'].shift(1) / data['close'].shift(2) - 1)
    
    # Volume efficiency signals
    volume_acceleration = data['volume'] / data['volume'].shift(1)
    volume_consistency = data['volume'] / data['volume'].rolling(window=4).apply(
        lambda x: x.prod() ** (1/4) if not x.isna().any() else np.nan
    )
    volume_range_ratio = data['volume'] / ((data['high'] - data['low']) / data['close'])
    
    # Combined volume-price efficiency
    core_efficiency = (opening_efficiency * range_utilization * momentum_persistence) ** (1/3)
    volume_multiplier = (volume_acceleration * volume_consistency) ** (1/2)
    efficiency_adjustment = 1 / volume_range_ratio
    
    volume_efficiency_factor = core_efficiency * volume_multiplier * efficiency_adjustment
    
    # Intraday Session Momentum
    gap_momentum = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    morning_strength = (data['high'] - data['open']) / data['open']
    morning_support = (data['open'] - data['low']) / data['open']
    
    afternoon_push = (data['close'] - data['high']) / data['high']
    afternoon_recovery = (data['close'] - data['low']) / data['low']
    closing_drive = (data['close'] - data['open']) / data['open']
    
    session_alignment = (morning_strength * afternoon_push) ** (1/2)
    support_resistance_efficiency = (morning_support * afternoon_recovery) ** (1/2)
    
    intraday_factor = session_alignment * support_resistance_efficiency * closing_drive
    
    # Multi-Timeframe Convergence Factor
    # Short-term Framework (1-3 days)
    price_acceleration = (data['close'] / data['close'].shift(2) - 1) * (data['close'] / data['close'].shift(1) - 1)
    volume_momentum = data['volume'] / data['volume'].shift(2)
    intraday_efficiency = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Medium-term Framework (5-10 days)
    price_trend = data['close'] / data['close'].shift(7) - 1
    volume_trend = data['volume'] / data['volume'].rolling(window=6).apply(
        lambda x: x.prod() ** (1/6) if not x.isna().any() else np.nan
    )
    
    volatility_context = (data['high'] - data['low']) / data[['high', 'low']].apply(
        lambda x: (x['high'] - x['low']), axis=1
    ).rolling(window=6).apply(
        lambda x: x.prod() ** (1/6) if not x.isna().any() else np.nan
    )
    
    timeframe_alignment = (price_acceleration * price_trend) ** (1/2)
    volume_convergence = (volume_momentum * volume_trend) ** (1/2)
    
    convergence_factor = timeframe_alignment * volume_convergence / volatility_context
    
    # Final combined factor (equal weighted combination)
    final_factor = (momentum_factor + volume_efficiency_factor + intraday_factor + convergence_factor) / 4
    
    return final_factor
