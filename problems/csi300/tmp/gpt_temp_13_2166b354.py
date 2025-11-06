import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Price Momentum Divergence
    data['intraday_momentum'] = (data['high'] - data['low']) / data['close']
    data['prev_intraday_momentum'] = data['intraday_momentum'].shift(1)
    momentum_divergence = data['intraday_momentum'] - data['prev_intraday_momentum']
    
    # Volume-Adjusted Price Acceleration
    data['price_diff1'] = data['close'] - data['close'].shift(1)
    data['price_diff2'] = data['close'].shift(1) - data['close'].shift(2)
    price_acceleration = data['price_diff1'] - data['price_diff2']
    volume_ratio = data['volume'] / data['volume'].shift(1)
    volume_adjusted_acceleration = price_acceleration * volume_ratio
    
    # Relative Strength Breakout with Volume Confirmation
    data['five_day_return'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['five_day_avg_volume'] = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_surge = data['volume'] / data['five_day_avg_volume']
    relative_strength_volume = data['five_day_return'] * volume_surge
    
    # Price-Volume Efficiency Ratio
    data['price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['avg_trade_size'] = data['amount'] / data['volume']
    efficiency_ratio = data['price_efficiency'] * data['avg_trade_size']
    
    # Volatility-Regulated Momentum Persistence
    data['three_day_return'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['five_day_avg_range'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    volatility_ratio = data['daily_range'] / data['five_day_avg_range']
    regulated_momentum = data['three_day_return'] / volatility_ratio
    
    # Turnover-Adjusted Price Reversal
    data['ten_day_high'] = data['high'].rolling(window=10, min_periods=1).max()
    data['ten_day_low'] = data['low'].rolling(window=10, min_periods=1).min()
    price_position = (data['close'] - data['ten_day_low']) / (data['ten_day_high'] - data['ten_day_low'])
    
    # Estimate outstanding shares using average trade size and volume
    data['estimated_shares'] = data['volume'] * data['avg_trade_size']
    turnover_rate = data['volume'] / data['estimated_shares']
    turnover_adjusted_reversal = price_position * turnover_rate
    
    # Multi-timeframe Volume-Price Alignment
    data['price_3d_trend'] = data['close'].rolling(window=3, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    data['volume_3d_trend'] = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    data['price_10d_trend'] = data['close'].rolling(window=10, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    data['volume_10d_trend'] = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    
    short_term_corr = data['price_3d_trend'] * data['volume_3d_trend']
    long_term_corr = data['price_10d_trend'] * data['volume_10d_trend']
    timeframe_alignment = short_term_corr - long_term_corr
    
    # Absolute Price Change with Volume Momentum
    abs_price_change = abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1))
    volume_momentum = data['volume'] - data['volume'].shift(1)
    volume_weighted_change = abs_price_change * volume_momentum
    
    # Gap Persistence Factor
    gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    intraday_change = (data['close'] - data['open']) / data['open']
    gap_persistence = gap * np.sign(intraday_change)
    
    # Volume-Weighted Price Range Efficiency
    normalized_range = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    data['twenty_day_avg_volume'] = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_weight = data['volume'] / data['twenty_day_avg_volume']
    volume_weighted_range = normalized_range * volume_weight
    
    # Combine all factors with equal weights
    factor = (
        momentum_divergence.fillna(0) +
        volume_adjusted_acceleration.fillna(0) +
        relative_strength_volume.fillna(0) +
        efficiency_ratio.fillna(0) +
        regulated_momentum.fillna(0) +
        turnover_adjusted_reversal.fillna(0) +
        timeframe_alignment.fillna(0) +
        volume_weighted_change.fillna(0) +
        gap_persistence.fillna(0) +
        volume_weighted_range.fillna(0)
    ) / 10
    
    return factor
