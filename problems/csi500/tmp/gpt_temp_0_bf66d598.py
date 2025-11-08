import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate multi-timeframe momentum divergence alpha factor with volume-price alignment
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Divergence
    # Short-term (3-day)
    data['short_price_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['short_intraday_conf'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['short_volume_support'] = data['volume'] / data['volume'].shift(3)
    
    # Medium-term (5-day)
    data['medium_price_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_range_eff'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['medium_volume_persist'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Long-term (10-day)
    data['long_price_momentum'] = data['close'] / data['close'].shift(10) - 1
    
    # Calculate trend consistency (geometric mean of daily returns)
    returns = data['close'].pct_change() + 1
    data['long_trend_consistency'] = returns.rolling(window=10).apply(
        lambda x: np.prod(x) ** (1/len(x)) if len(x) == 10 else np.nan
    ) - 1
    
    data['long_volume_trend'] = data['volume'] / data['volume'].rolling(window=10).mean()
    
    # Combined momentum signals
    data['momentum_alignment'] = (
        (1 + data['short_price_momentum']) * 
        (1 + data['medium_price_momentum']) * 
        (1 + data['long_price_momentum'])
    ) ** (1/3) - 1
    
    data['volume_confirmation'] = (
        data['short_volume_support'] * 
        data['medium_volume_persist'] * 
        data['long_volume_trend']
    ) ** (1/3)
    
    data['intraday_efficiency'] = (data['short_intraday_conf'] + data['medium_range_eff']) / 2
    
    # Volume-Price Session Alignment (simplified for daily data)
    data['gap_strength'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['morning_range_capture'] = (data['high'] - data['open']) / data['open']
    data['morning_support_test'] = (data['open'] - data['low']) / data['open']
    
    # Using daily volume as proxy (assuming no intraday data)
    data['afternoon_momentum'] = (data['close'] - (data['high'] + data['low']) / 2) / ((data['high'] + data['low']) / 2)
    data['afternoon_recovery'] = (data['close'] - data['low']) / data['low']
    data['closing_drive'] = (data['close'] - data['open']) / data['open']
    
    # Combined session signals
    data['session_consistency'] = (data['morning_range_capture'] * data['afternoon_momentum']) ** 0.5
    data['support_resistance'] = (data['morning_support_test'] * data['afternoon_recovery']) ** 0.5
    
    # Price-Range Efficiency Factor
    # Short-term (3-day)
    price_move_3d = data['close'] - data['close'].shift(2)
    range_sum_3d = (data['high'] - data['low']).rolling(window=3).sum()
    data['short_efficiency'] = price_move_3d / range_sum_3d
    
    data['short_volume_eff'] = data['volume'] / data['volume'].rolling(window=3).mean()
    data['opening_gap_eff'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Medium-term (7-day)
    price_move_7d = data['close'] - data['close'].shift(6)
    range_sum_7d = (data['high'] - data['low']).rolling(window=7).sum()
    data['medium_efficiency'] = price_move_7d / range_sum_7d
    
    data['medium_volume_persist_eff'] = data['volume'] / data['volume'].rolling(window=7).mean()
    data['range_stability'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=7).mean()
    
    # Combined efficiency signals
    data['multi_timeframe_eff'] = (data['short_efficiency'] * data['medium_efficiency']) ** 0.5
    data['volume_conf_eff'] = (data['short_volume_eff'] * data['medium_volume_persist_eff']) ** 0.5
    
    # Convergence-Divergence Detection
    data['short_divergence'] = (data['close'] / data['close'].shift(2) - 1) * (data['volume'] / data['volume'].shift(2) - 1)
    data['medium_divergence'] = (data['close'] / data['close'].shift(5) - 1) * (data['volume'] / data['volume'].rolling(window=5).mean() - 1)
    
    data['divergence_strength'] = abs(
        (data['close'] / data['close'].shift(2) - 1) - (data['volume'] / data['volume'].shift(2) - 1)
    )
    
    data['morning_afternoon_align'] = (data['morning_range_capture'] * data['afternoon_momentum']) ** 0.5
    data['opening_closing_consist'] = (data['gap_strength'] * data['closing_drive']) ** 0.5
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Final alpha factor construction
    # Core momentum component
    momentum_component = data['momentum_alignment'] * data['volume_confirmation'] * (1 + data['intraday_efficiency'])
    
    # Session alignment component
    session_component = data['session_consistency'] * data['support_resistance'] * data['closing_drive']
    
    # Efficiency component
    efficiency_component = data['multi_timeframe_eff'] * data['volume_conf_eff'] / (data['range_stability'] + 1e-8)
    
    # Convergence component
    convergence_component = (
        data['short_divergence'] * data['medium_divergence'] * 
        data['morning_afternoon_align'] * data['opening_closing_consist'] * 
        data['range_utilization']
    ) ** 0.2
    
    # Final alpha factor - weighted combination
    alpha_factor = (
        0.4 * momentum_component +
        0.3 * session_component +
        0.2 * efficiency_component +
        0.1 * convergence_component
    )
    
    return alpha_factor
