import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_filling_ratio'] = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    
    # Opening Auction Signals
    data['opening_pressure'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Calculate rolling statistics for auction imbalance
    amount_rolling_mean = data['amount'].rolling(window=5, min_periods=1).mean().shift(1)
    amount_rolling_std = data['amount'].rolling(window=5, min_periods=1).std().shift(1)
    data['auction_imbalance'] = (data['amount'] - amount_rolling_mean) / (amount_rolling_std + 0.001)
    
    # Midday Momentum Shifts
    data['midpoint_reversion'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] - data['low'])/2).replace(0, np.nan)
    
    # Volume Acceleration
    data['volume_ratio_t'] = data['volume'] / data['volume'].shift(1)
    data['volume_ratio_t_minus_1'] = data['volume'].shift(1) / data['volume'].shift(2)
    data['volume_acceleration'] = data['volume_ratio_t'] - data['volume_ratio_t_minus_1']
    
    # Closing Dynamics
    data['end_of_day_pressure'] = (data['close'] - (data['high'] + data['low'])/2) / (data['close'] - data['open']).replace(0, np.nan)
    
    close_rolling_std = data['close'].rolling(window=5, min_periods=1).std().shift(1)
    data['final_hour_momentum'] = (data['close'] - data['close'].shift(1)) / (close_rolling_std + 0.001)
    
    # Multi-Timeframe Gap Analysis
    # Gap Persistence (4-day correlation)
    gap_magnitude_series = data['gap_magnitude']
    gap_persistence = []
    for i in range(len(data)):
        if i >= 5:
            current_window = gap_magnitude_series.iloc[i-4:i+1]
            prev_window = gap_magnitude_series.iloc[i-5:i]
            if len(current_window.dropna()) >= 3 and len(prev_window.dropna()) >= 3:
                corr = current_window.corr(prev_window)
                gap_persistence.append(corr if not np.isnan(corr) else 0)
            else:
                gap_persistence.append(0)
        else:
            gap_persistence.append(0)
    data['gap_persistence'] = gap_persistence
    
    # Consecutive Gap Direction
    data['gap_sign'] = np.sign(data['gap_magnitude'])
    data['consecutive_gap_direction'] = data['gap_sign'].rolling(window=3, min_periods=1).sum()
    
    # Gap Volatility
    data['gap_volatility'] = data['gap_magnitude'].rolling(window=6, min_periods=1).std()
    
    # Gap Clustering
    gap_magnitude_rolling_mean = data['gap_magnitude'].rolling(window=10, min_periods=1).mean().shift(1)
    gap_clustering = []
    for i in range(len(data)):
        if i >= 5:
            window = data['gap_magnitude'].iloc[i-5:i+1]
            threshold = gap_magnitude_rolling_mean.iloc[i]
            count = (window > threshold).sum()
            gap_clustering.append(count)
        else:
            gap_clustering.append(0)
    data['gap_clustering'] = gap_clustering
    
    # Gap Mean Reversion
    abs_gap_rolling_mean = abs(data['gap_magnitude']).rolling(window=10, min_periods=1).mean().shift(1)
    data['gap_mean_reversion'] = data['gap_magnitude'] / (abs_gap_rolling_mean + 0.001)
    
    # Gap Regime Transition
    data['regime_change_signal'] = np.sign(data['gap_magnitude'] * data['gap_magnitude'].shift(1))
    gap_volatility_5d = data['gap_magnitude'].rolling(window=6, min_periods=1).std()
    data['transition_strength'] = abs(data['gap_magnitude'] - data['gap_magnitude'].shift(1)) / (gap_volatility_5d + 0.001)
    
    # Volume-Gap Integration
    volume_rolling_mean = data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    data['gap_volume_multiplier'] = (data['volume'] / (volume_rolling_mean + 0.001)) * data['gap_magnitude']
    
    # Volume Confirmation (4-day correlation)
    volume_confirmation = []
    for i in range(len(data)):
        if i >= 4:
            volume_window = data['volume'].iloc[i-4:i+1]
            gap_window = abs(data['gap_magnitude']).iloc[i-4:i+1]
            if len(volume_window.dropna()) >= 3 and len(gap_window.dropna()) >= 3:
                corr = volume_window.corr(gap_window)
                volume_confirmation.append(np.sign(data['gap_magnitude'].iloc[i]) * (corr if not np.isnan(corr) else 0))
            else:
                volume_confirmation.append(0)
        else:
            volume_confirmation.append(0)
    data['volume_confirmation'] = volume_confirmation
    
    # Amount-Based Gap Quality
    amount_per_volume = data['amount'] / (data['volume'] + 0.001)
    amount_per_volume_rolling_mean = amount_per_volume.rolling(window=5, min_periods=1).mean().shift(1)
    data['large_trade_gap_impact'] = (amount_per_volume / (amount_per_volume_rolling_mean + 0.001)) * data['gap_magnitude']
    
    amount_rolling_mean_10d = data['amount'].rolling(window=10, min_periods=1).mean().shift(1)
    data['institutional_gap_participation'] = (data['amount'] / (amount_rolling_mean_10d + 0.001)) * data['gap_filling_ratio']
    
    # Core Gap Signals
    data['persistent_gap_momentum'] = data['gap_persistence'] * data['gap_magnitude'] * data['volume_acceleration']
    data['gap_reversion_signal'] = -data['gap_mean_reversion'] * data['intraday_range_efficiency']
    data['breakaway_gap_detection'] = data['gap_magnitude'] * data['gap_clustering'] * data['auction_imbalance']
    
    # Microstructure Enhancement
    data['opening_enhancement'] = data['opening_pressure'] * data['gap_volume_multiplier']
    data['midday_enhancement'] = data['midpoint_reversion'] * data['volume_confirmation']
    data['closing_enhancement'] = data['end_of_day_pressure'] * data['final_hour_momentum']
    
    # Regime-Adaptive Weighting
    gap_volatility_rolling_mean = data['gap_volatility'].rolling(window=5, min_periods=1).mean().shift(1)
    data['gap_regime_weight'] = 1 / (1 + abs(data['gap_volatility'] / (gap_volatility_rolling_mean + 0.001) - 1))
    data['transition_weight'] = data['transition_strength'] * data['regime_change_signal']
    
    # Final Alpha Calculation
    alpha = (data['persistent_gap_momentum'] * data['opening_enhancement'] + 
             data['gap_reversion_signal'] * data['midday_enhancement'] * data['gap_regime_weight'] + 
             data['breakaway_gap_detection'] * data['closing_enhancement'] * data['transition_weight'])
    
    return alpha
