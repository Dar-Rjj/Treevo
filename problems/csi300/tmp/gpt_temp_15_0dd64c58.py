import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate returns and basic metrics
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    data['high_low_range'] = data['high'] - data['low']
    data['close_open_range'] = abs(data['close'] - data['open'])
    data['trade_size'] = data['amount'] / data['volume']
    
    # Volatility-Trend Regime Classification
    data['vol_20'] = data['returns'].rolling(window=20, min_periods=10).std()
    data['vol_60'] = data['returns'].rolling(window=60, min_periods=30).std()
    data['vol_ratio'] = data['vol_20'] / data['vol_60']
    data['trend_strength'] = (data['close'] / data['close'].shift(10) - 1) / data['vol_20']
    
    # Volume-Price Regime Assessment
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_surge'] = (data['volume'] > 1.5 * data['volume_ma_5']).astype(int)
    data['volume_drought'] = (data['volume'] < 0.7 * data['volume_ma_5']).astype(int)
    data['volume_price_divergence'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['close'] / data['close'].shift(5) - 1)
    
    # Microstructure State Integration
    data['efficiency'] = data['close_open_range'] / data['high_low_range']
    data['trade_size_state'] = data['trade_size']
    
    # Asymmetric Price-Volume Dynamics
    data['upside_rejection'] = data['high'] - np.maximum(data['open'], data['close'])
    data['downside_rejection'] = np.minimum(data['open'], data['close']) - data['low']
    data['net_asymmetric_rejection'] = data['upside_rejection'] - data['downside_rejection']
    
    # Volume Asymmetry Components
    up_volume = data.apply(lambda x: x['volume'] if x['close'] > x['open'] else 0, axis=1)
    down_volume = data.apply(lambda x: x['volume'] if x['close'] < x['open'] else 0, axis=1)
    
    data['up_volume_10'] = up_volume.rolling(window=10, min_periods=5).sum()
    data['down_volume_10'] = down_volume.rolling(window=10, min_periods=5).sum()
    data['up_down_volume_ratio'] = data['up_volume_10'] / (data['down_volume_10'] + 1e-8)
    
    data['volume_concentration'] = data['volume'].rolling(window=5, min_periods=3).max() / data['volume'].rolling(window=5, min_periods=3).mean()
    data['bid_ask_pressure'] = (data['high'] - data['close']) / (data['close'] - data['low'] + 1e-8)
    
    # Price Gap Dynamics
    data['gap_direction'] = (data['close'] - data['open']) / (abs(data['close'] - data['open']) + 1e-8)
    data['gap_direction_persistence'] = data['gap_direction'].rolling(window=5, min_periods=3).mean()
    data['gap_efficiency'] = data['close_open_range'] / data['high_low_range']
    data['gap_velocity_alignment'] = np.sign(data['close'] - data['open']) * np.sign(data['returns'])
    
    # Velocity-Efficiency Integration
    data['clean_momentum'] = data['returns']
    data['short_term_acceleration'] = (data['close'] / data['close'].shift(3) - 1) - (data['close'].shift(3) / data['close'].shift(6) - 1)
    data['momentum_efficiency_ratio'] = data['clean_momentum'] / (data['high_low_range'] / data['close'].shift(1))
    data['breakout_asymmetry'] = (data['high'] / data['high'].shift(1) - 1) - (data['low'] / data['low'].shift(1) - 1)
    
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(3))**(1/3) - 1
    data['trade_size_momentum'] = (data['trade_size'] / data['trade_size'].shift(1)) - 1
    data['volume_weighted_efficiency'] = data['efficiency'] * data['volume']
    
    # Efficiency Dynamics
    data['efficiency_momentum'] = (data['efficiency'] / data['efficiency'].shift(1)) - 1
    
    # Calculate state persistence
    for col in ['vol_ratio', 'trend_strength', 'volume_surge', 'volume_drought', 'net_asymmetric_rejection']:
        data[f'{col}_persistence'] = data[col].rolling(window=3, min_periods=2).apply(
            lambda x: np.sum(np.diff(x) == 0) / (len(x) - 1) if len(x) > 1 else 0
        )
    
    # Regime classification
    data['volatility_regime'] = np.select(
        [data['vol_ratio'] > 1.2, data['vol_ratio'] < 0.8],
        [2, 0],  # High=2, Low=0
        default=1  # Normal=1
    )
    
    data['trend_regime'] = np.select(
        [data['trend_strength'] > 0.5, data['trend_strength'] < -0.5],
        [2, 0],  # Trending=2, Reversal=0
        default=1  # Mean-reverting=1
    )
    
    # Core Velocity Components
    data['asymmetric_rejection_velocity'] = data['net_asymmetric_rejection'] * data['efficiency_momentum']
    data['volume_momentum_velocity'] = data['clean_momentum'] * data['volume_acceleration']
    data['trade_size_efficiency_velocity'] = (data['trade_size'] * data['efficiency']) * data['short_term_acceleration']
    data['breakout_efficiency_velocity'] = data['breakout_asymmetry'] * data['efficiency']
    
    # State-Specific Enhancement
    regime_multiplier = np.ones(len(data))
    regime_multiplier = np.where(data['volatility_regime'] == 2, 1.3, regime_multiplier)  # High volatility
    regime_multiplier = np.where(data['trend_regime'] == 2, 1.2, regime_multiplier)  # Trending
    regime_multiplier = np.where(data['trend_regime'] == 1, 1.1, regime_multiplier)  # Mean-reverting
    regime_multiplier = np.where(data['volume_surge'] == 1, 1.2, regime_multiplier)  # Volume surge
    
    # Apply state persistence weight
    state_persistence = (data['vol_ratio_persistence'] + data['trend_strength_persistence']) / 2
    regime_multiplier = regime_multiplier * (1 + state_persistence)
    
    # Asymmetry-Confirmed Signals
    data['volume_asymmetry_velocity'] = data['volume_momentum_velocity'] * data['up_down_volume_ratio']
    data['rejection_efficiency_velocity'] = data['asymmetric_rejection_velocity'] * data['gap_efficiency']
    data['institutional_breakout_velocity'] = data['breakout_efficiency_velocity'] * data['trade_size_momentum']
    data['price_gap_momentum'] = data['clean_momentum'] * data['gap_direction_persistence']
    
    # Range and Volatility Dynamics
    data['range_expansion'] = (data['high_low_range'] / data['high_low_range'].shift(1) > 1.2).astype(int)
    data['mean_reversion_strength'] = 1 - abs(data['close'] - data['close'].shift(1)) / data['high_low_range']
    
    # Composite Alpha Synthesis
    primary_factor = data['volume_asymmetry_velocity'] * data['volume_price_divergence']
    secondary_factor = data['rejection_efficiency_velocity'] * data['efficiency_momentum']
    tertiary_factor = data['institutional_breakout_velocity'] * data['trade_size_momentum']
    quaternary_factor = data['price_gap_momentum'] * data['mean_reversion_strength']
    
    # Final regime-adaptive alpha
    final_alpha = (
        primary_factor * 0.4 + 
        secondary_factor * 0.3 + 
        tertiary_factor * 0.2 + 
        quaternary_factor * 0.1
    ) * regime_multiplier
    
    # Apply multi-timeframe validation
    momentum_consistency = data['clean_momentum'].rolling(window=3, min_periods=2).apply(
        lambda x: np.sum(np.diff(np.sign(x)) == 0) / (len(x) - 1) if len(x) > 1 else 0
    )
    
    final_alpha = final_alpha * (1 + momentum_consistency)
    
    return final_alpha
