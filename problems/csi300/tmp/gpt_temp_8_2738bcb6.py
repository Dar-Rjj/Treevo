import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily range
    data['daily_range'] = data['high'] - data['low']
    data['range_utilization'] = (data['close'] - data['open']) / (data['daily_range'] + 1e-8)
    
    # Multi-timeframe efficiency calculations
    # Short-term efficiency (5-day)
    data['ret_5d'] = data['close'].pct_change(5)
    data['avg_range_5d'] = data['daily_range'].rolling(5).mean()
    data['efficiency_5d'] = data['ret_5d'] / (data['avg_range_5d'] + 1e-8)
    
    # Medium-term efficiency (10-day)
    data['ret_10d'] = data['close'].pct_change(10)
    data['avg_range_10d'] = data['daily_range'].rolling(10).mean()
    data['efficiency_10d'] = data['ret_10d'] / (data['avg_range_10d'] + 1e-8)
    
    # Long-term efficiency (20-day)
    data['ret_20d'] = data['close'].pct_change(20)
    data['avg_range_20d'] = data['daily_range'].rolling(20).mean()
    data['efficiency_20d'] = data['ret_20d'] / (data['avg_range_20d'] + 1e-8)
    
    # Efficiency acceleration patterns
    data['primary_acceleration'] = data['efficiency_5d'] - data['efficiency_10d']
    data['secondary_acceleration'] = data['efficiency_10d'] - data['efficiency_20d']
    data['acceleration_decay'] = data['primary_acceleration'] - data['primary_acceleration'].shift(5)
    
    # Volume-amount efficiency divergence
    data['volume_efficiency'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['amount_efficiency'] = data['amount'] / (data['daily_range'] + 1e-8)
    data['volume_amount_divergence'] = data['amount_efficiency'] - data['volume_efficiency']
    data['divergence_momentum'] = data['volume_amount_divergence'] - data['volume_amount_divergence'].shift(5)
    
    # Range efficiency dynamics
    data['range_efficiency_change'] = data['range_utilization'] - data['range_utilization'].shift(5)
    
    # Efficiency acceleration divergence detection
    data['price_volume_divergence'] = np.where(
        (data['primary_acceleration'] > 0) & (data['volume_amount_divergence'] < 0), 
        data['primary_acceleration'] * abs(data['volume_amount_divergence']),
        np.where(
            (data['primary_acceleration'] < 0) & (data['volume_amount_divergence'] > 0),
            data['primary_acceleration'] * abs(data['volume_amount_divergence']),
            0
        )
    )
    
    # Volatility regime context
    data['vol_20d'] = data['close'].pct_change().rolling(20).std()
    data['vol_60d'] = data['close'].pct_change().rolling(60).std()
    data['vol_40d'] = data['close'].pct_change().rolling(40).std()
    
    data['high_vol_regime'] = (data['vol_20d'] > data['vol_60d']).astype(int)
    data['low_vol_regime'] = (data['vol_20d'] < data['vol_40d']).astype(int)
    
    # Efficiency acceleration persistence
    data['acceleration_streak'] = 0
    for i in range(1, len(data)):
        if data['primary_acceleration'].iloc[i] > data['primary_acceleration'].iloc[i-1]:
            data.loc[data.index[i], 'acceleration_streak'] = data['acceleration_streak'].iloc[i-1] + 1
        elif data['primary_acceleration'].iloc[i] < data['primary_acceleration'].iloc[i-1]:
            data.loc[data.index[i], 'acceleration_streak'] = data['acceleration_streak'].iloc[i-1] - 1
        else:
            data.loc[data.index[i], 'acceleration_streak'] = data['acceleration_streak'].iloc[i-1]
    
    # Donchian efficiency analysis
    data['efficiency_high_20d'] = data['efficiency_5d'].rolling(20).max()
    data['efficiency_low_20d'] = data['efficiency_5d'].rolling(20).min()
    data['efficiency_channel_width'] = data['efficiency_high_20d'] - data['efficiency_low_20d']
    data['efficiency_channel_position'] = (data['efficiency_5d'] - data['efficiency_low_20d']) / (data['efficiency_channel_width'] + 1e-8)
    
    # Regime classification
    strong_acceleration = (data['primary_acceleration'] > 0) & (data['secondary_acceleration'] > 0)
    acceleration_decay = (data['primary_acceleration'] > 0) & (data['acceleration_decay'] < 0)
    efficiency_reversal = (data['primary_acceleration'] < 0) & (data['secondary_acceleration'] < 0)
    
    # Adaptive weighting based on regime
    base_signal = np.zeros(len(data))
    
    # Strong acceleration regime
    mask_strong = strong_acceleration
    base_signal[mask_strong] = (
        data['primary_acceleration'][mask_strong] * 0.3 +
        data['volume_amount_divergence'][mask_strong] * 0.25 +
        data['range_utilization'][mask_strong] * 0.2 +
        data['efficiency_channel_position'][mask_strong] * 0.15 +
        np.tanh(data['acceleration_streak'][mask_strong] / 10) * 0.1
    )
    
    # Acceleration decay regime
    mask_decay = acceleration_decay
    base_signal[mask_decay] = (
        data['acceleration_decay'][mask_decay] * 0.35 +
        (-data['volume_amount_divergence'][mask_decay]) * 0.25 +
        (-data['range_efficiency_change'][mask_decay]) * 0.2 +
        (0.5 - data['efficiency_channel_position'][mask_decay]) * 0.1 +
        (-np.tanh(data['acceleration_streak'][mask_decay] / 10)) * 0.1
    )
    
    # Transition regime (default)
    mask_transition = ~(mask_strong | mask_decay)
    base_signal[mask_transition] = (
        data['primary_acceleration'][mask_transition] * data['volume_amount_divergence'][mask_transition] * 0.3 +
        data['volume_amount_divergence'][mask_transition] * data['range_utilization'][mask_transition] * 0.25 +
        (data['efficiency_channel_position'][mask_transition] - 0.5) * 0.2 +
        (data['primary_acceleration'][mask_transition] * data['secondary_acceleration'][mask_transition]) * 0.15 +
        (data['vol_20d'][mask_transition] / (data['vol_60d'][mask_transition] + 1e-8) - 1) * 0.1
    )
    
    # Volume-amount confirmation multiplier
    volume_confirmation = 1 + np.tanh(data['divergence_momentum'] * data['volume_amount_divergence'])
    
    # Donchian context enhancement
    donchian_enhancement = np.where(
        data['efficiency_channel_position'] > 0.8,
        1.2,
        np.where(
            data['efficiency_channel_position'] < 0.2,
            0.8,
            1.0
        )
    )
    
    # Persistence scaling factor
    persistence_scaling = 1 + np.tanh(data['acceleration_streak'] / 20)
    
    # Volatility adjustment
    volatility_adjustment = np.where(
        data['high_vol_regime'] == 1,
        0.7,
        np.where(
            data['low_vol_regime'] == 1,
            1.3,
            1.0
        )
    )
    
    # Final composite alpha factor
    alpha_factor = (
        base_signal * 
        volume_confirmation * 
        donchian_enhancement * 
        persistence_scaling * 
        volatility_adjustment
    )
    
    # Smooth the final factor with a 3-day moving average
    alpha_smoothed = pd.Series(alpha_factor, index=data.index).rolling(3, min_periods=1).mean()
    
    return alpha_smoothed
