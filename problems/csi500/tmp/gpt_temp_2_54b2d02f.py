import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate regime-adaptive alpha factor combining volume-confirmed momentum, 
    intraday pressure persistence, amount flow imbalance, and range breakouts.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Volume-Confirmed Momentum with Regime Adaptation
    # Momentum calculations
    data['price_momentum_1d'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['price_momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['volume_momentum_1d'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Volume confirmation
    data['directional_alignment_1d'] = (data['price_momentum_1d'] * data['volume_momentum_1d'] > 0).astype(int)
    data['strength_1d'] = abs(data['price_momentum_1d']) * abs(data['volume_momentum_1d'])
    data['directional_alignment_5d'] = (data['price_momentum_5d'] * data['volume_momentum_5d'] > 0).astype(int)
    data['strength_5d'] = abs(data['price_momentum_5d']) * abs(data['volume_momentum_5d'])
    
    # Volume persistence (consecutive days with same direction volume)
    data['volume_direction'] = np.sign(data['volume_momentum_1d'])
    data['volume_streak'] = (data['volume_direction'] == data['volume_direction'].shift(1)).astype(int)
    data['volume_streak_count'] = data['volume_streak'].groupby((data['volume_streak'] != data['volume_streak'].shift()).cumsum()).cumsum()
    
    # Volume acceleration
    data['volume_acceleration'] = data['volume_momentum_1d'] - data['volume_momentum_1d'].shift(1)
    
    # Volatility regime detection
    data['daily_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['range_median_20d'] = data['daily_range'].rolling(window=20).median()
    data['volatility_regime'] = (data['daily_range'] > data['range_median_20d']).astype(int)  # 1 = high, 0 = low
    
    # Intraday Pressure Persistence
    data['morning_pressure'] = (data['high'] - data['open']) / data['open']
    # Using simplified volume ratio (assuming no intraday data)
    data['morning_volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['afternoon_pressure'] = (data['close'] - data['low']) / data['low']
    data['afternoon_volume_persistence'] = data['volume'] / data['volume'].rolling(window=5).mean()
    
    # Multi-day pressure persistence
    data['pressure_direction'] = np.sign(data['morning_pressure'] + data['afternoon_pressure'])
    data['pressure_streak'] = (data['pressure_direction'] == data['pressure_direction'].shift(1)).astype(int)
    data['pressure_streak_count'] = data['pressure_streak'].groupby((data['pressure_streak'] != data['pressure_streak'].shift()).cumsum()).cumsum()
    
    # Amount Flow Imbalance
    data['up_day'] = (data['close'] > data['open']).astype(int)
    data['down_day'] = (data['close'] < data['open']).astype(int)
    data['up_day_amount'] = data['amount'] * data['up_day']
    data['down_day_amount'] = data['amount'] * data['down_day']
    data['net_flow'] = (data['up_day_amount'] - data['down_day_amount']) / data['amount']
    
    # Flow persistence
    data['flow_direction'] = np.sign(data['net_flow'])
    data['flow_streak'] = (data['flow_direction'] == data['flow_direction'].shift(1)).astype(int)
    data['flow_streak_count'] = data['flow_streak'].groupby((data['flow_streak'] != data['flow_streak'].shift()).cumsum()).cumsum()
    data['flow_acceleration'] = data['net_flow'] - data['net_flow'].shift(3)
    data['flow_intensity'] = abs(data['net_flow']) * data['flow_streak_count']
    
    # Flow-price alignment
    data['flow_price_alignment'] = (np.sign(data['net_flow']) == np.sign(data['price_momentum_1d'])).astype(int)
    data['flow_price_strength'] = data['flow_intensity'] * abs(data['price_momentum_1d'])
    
    # Range Breakout with Volume Validation
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['price_vs_20d_high'] = data['close'] / data['high'].rolling(window=20).max()
    data['price_vs_20d_low'] = data['close'] / data['low'].rolling(window=20).min()
    data['breakout_direction'] = np.where(data['price_vs_20d_high'] > 0.98, 1, 
                                         np.where(data['price_vs_20d_low'] < 1.02, -1, 0))
    
    # Volume validation
    data['breakout_volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_persistence'] = ((data['volume'] > data['volume'].shift(1)) & 
                                 (data['volume'].shift(1) > data['volume'].shift(2))).astype(int)
    data['volume_range_correlation'] = np.sign(data['volume_momentum_1d']) * np.sign(data['range_expansion'])
    
    # Breakout confidence
    data['breakout_strength'] = data['range_expansion'] * data['breakout_volume_ratio']
    data['validation_score'] = data['volume_persistence'] * data['volume_range_correlation']
    data['breakout_signal'] = data['breakout_strength'] * data['validation_score'] * data['breakout_direction']
    
    # Regime-Adaptive Factor Combination
    for i in range(len(data)):
        if i < 20:  # Skip initial period for rolling calculations
            continue
            
        # Volatility-based weighting
        if data['volatility_regime'].iloc[i] == 1:  # High volatility
            # Emphasize mean reversion factors
            momentum_weight = 0.2
            pressure_weight = 0.3
            flow_weight = 0.3
            breakout_weight = 0.2
        else:  # Low volatility
            # Emphasize momentum factors
            momentum_weight = 0.4
            pressure_weight = 0.2
            flow_weight = 0.2
            breakout_weight = 0.2
        
        # Volume persistence multiplier
        volume_multiplier = 1 + (data['volume_streak_count'].iloc[i] * 0.1)
        
        # Component calculations
        # Volume-confirmed momentum
        momentum_component = (
            data['directional_alignment_1d'].iloc[i] * data['strength_1d'].iloc[i] +
            data['directional_alignment_5d'].iloc[i] * data['strength_5d'].iloc[i]
        ) * volume_multiplier
        
        # Intraday pressure
        pressure_component = (
            data['morning_pressure'].iloc[i] * data['morning_volume_ratio'].iloc[i] +
            data['afternoon_pressure'].iloc[i] * data['afternoon_volume_persistence'].iloc[i]
        ) * (1 + data['pressure_streak_count'].iloc[i] * 0.05)
        
        # Amount flow
        flow_component = (
            data['net_flow'].iloc[i] * data['flow_streak_count'].iloc[i] +
            data['flow_price_strength'].iloc[i] * data['flow_price_alignment'].iloc[i]
        )
        
        # Range breakout
        breakout_component = data['breakout_signal'].iloc[i]
        
        # Combine components with regime-adaptive weights
        factor.iloc[i] = (
            momentum_weight * momentum_component +
            pressure_weight * pressure_component +
            flow_weight * flow_component +
            breakout_weight * breakout_component
        )
    
    # Normalize the factor
    factor = (factor - factor.mean()) / factor.std()
    
    return factor
