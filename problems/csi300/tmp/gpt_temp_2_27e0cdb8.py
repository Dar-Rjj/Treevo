import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Gap-Driven Momentum Factor
    Combines gap analysis, intraday momentum structure, volume flow, and price efficiency metrics
    """
    data = df.copy()
    
    # Gap Analysis
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = data['open'] / data['prev_close'] - 1
    data['directional_gap'] = np.sign(data['open'] - data['prev_close'])
    
    # Gap Persistence
    data['intraday_direction'] = np.sign(data['close'] - data['open'])
    data['gap_continuation'] = (data['intraday_direction'] == data['directional_gap']).astype(int)
    gap_size = np.abs(data['open'] - data['prev_close'])
    gap_size = np.where(gap_size == 0, 1e-10, gap_size)  # Avoid division by zero
    data['gap_filling_ratio'] = np.abs(data['close'] - data['open']) / gap_size
    
    # Intraday Momentum Structure
    data['morning_momentum'] = (data['high'] - data['open']) / data['open']
    data['afternoon_momentum'] = (data['close'] - data['low']) / data['low']
    data['midday_reversal'] = (data['high'] - data['low']) / (data['high'] + data['low'])
    
    # Momentum Asymmetry
    data['up_day'] = (data['close'] > data['open']).astype(int)
    data['down_day'] = (data['close'] < data['open']).astype(int)
    data['momentum_concentration'] = np.maximum(np.abs(data['morning_momentum']), 
                                               np.abs(data['afternoon_momentum']))
    
    # Volume Flow Analysis (using amount as proxy for volume-weighted metrics)
    data['volume_total'] = data['volume']
    data['early_volume_concentration'] = data['volume'].rolling(window=5, min_periods=1).mean() / data['volume_total']
    data['late_volume_surge'] = data['volume'].rolling(window=3, min_periods=1).mean() / data['volume_total']
    
    # Volume-Momentum Alignment
    data['volume_weighted_morning'] = data['morning_momentum'] * data['early_volume_concentration']
    data['volume_weighted_afternoon'] = data['afternoon_momentum'] * data['late_volume_surge']
    
    # Price Efficiency Metrics
    daily_range = data['high'] - data['low']
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)  # Avoid division by zero
    data['price_path_efficiency'] = np.abs(data['close'] - data['open']) / daily_range
    
    # Trend Consistency (3-day rolling window)
    data['intraday_move'] = np.sign(data['close'] - data['open'])
    data['trend_consistency'] = data['intraday_move'].rolling(window=3, min_periods=1).apply(
        lambda x: len(x[x == x.iloc[-1]]) if len(x) > 0 else 0, raw=False
    )
    
    # Volatility Clustering
    data['daily_range'] = daily_range
    data['range_ma_5'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_regime'] = data['daily_range'] / data['range_ma_5']
    
    # Gap-Induced Volatility (correlation proxy)
    data['gap_volatility_ratio'] = np.abs(data['overnight_gap']) / (data['daily_range'] + 1e-10)
    
    # Signal Construction
    
    # Gap-Driven Momentum Core
    gap_morning_component = data['directional_gap'] * data['morning_momentum']
    gap_afternoon_component = data['gap_continuation'] * data['afternoon_momentum']
    
    # Volume-Enhanced Components
    volume_morning_component = data['early_volume_concentration'] * data['morning_momentum']
    volume_afternoon_component = data['late_volume_surge'] * data['afternoon_momentum']
    
    # Efficiency-Weighted Integration
    efficiency_weighted_morning = gap_morning_component * data['price_path_efficiency']
    efficiency_weighted_afternoon = gap_afternoon_component * data['price_path_efficiency']
    trend_persistence = data['trend_consistency'] / 3.0  # Normalize to [0,1]
    
    # Asymmetric Signal Generation
    # Different weights for gap-up vs gap-down scenarios
    gap_up_mask = data['overnight_gap'] > 0
    gap_down_mask = data['overnight_gap'] < 0
    
    # Base signal components
    morning_signal = efficiency_weighted_morning * trend_persistence
    afternoon_signal = efficiency_weighted_afternoon * trend_persistence
    
    # Apply asymmetric weights
    morning_signal_weighted = np.where(gap_up_mask, morning_signal * 1.2, morning_signal * 0.8)
    afternoon_signal_weighted = np.where(gap_down_mask, afternoon_signal * 1.2, afternoon_signal * 0.8)
    
    # Volume-enhanced components
    volume_morning_weighted = volume_morning_component * data['price_path_efficiency']
    volume_afternoon_weighted = volume_afternoon_component * data['price_path_efficiency']
    
    # Combine all components with volatility adjustment
    volatility_adjustment = 1.0 / (1.0 + data['volatility_regime'])
    
    # Final factor calculation
    factor = (
        morning_signal_weighted * 0.4 +
        afternoon_signal_weighted * 0.4 +
        volume_morning_weighted * 0.1 +
        volume_afternoon_weighted * 0.1 +
        data['gap_filling_ratio'] * data['directional_gap'] * 0.1
    ) * volatility_adjustment
    
    # Clean up and return
    result = pd.Series(factor, index=data.index)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return result
