import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price Dispersion Dynamics
    # Intraday Dispersion Patterns
    data['relative_range_dispersion'] = (data['high'] - data['low']) / (data['close'].shift(1) + 1e-8)
    data['opening_gap_dispersion'] = np.abs(data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['close_to_close_dispersion'] = np.abs(data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Multi-timeframe Dispersion
    data['short_term_vol_clustering'] = ((data['close'] - data['close'].shift(1)) ** 2) / ((data['close'].shift(2) - data['close'].shift(3)) ** 2 + 1e-8)
    data['dispersion_momentum'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) - 1
    data['range_expansion_signal'] = (data['high'] - data['low']) / ((data['high'].shift(1) - data['low'].shift(1) + 
                                                                     data['high'].shift(2) - data['low'].shift(2) + 
                                                                     data['high'].shift(3) - data['low'].shift(3)) / 3 + 1e-8)
    
    # Asymmetric Dispersion
    data['upside_dispersion_bias'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['close'].shift(1) + 1e-8)
    data['downside_dispersion_bias'] = (np.minimum(data['open'], data['close']) - data['low']) / (data['close'].shift(1) + 1e-8)
    data['net_dispersion_asymmetry'] = (data['upside_dispersion_bias'] - data['downside_dispersion_bias']) / (data['close'].shift(1) + 1e-8)
    
    # Volume-Dispersion Interaction
    # Volume-Weighted Dispersion
    data['volume_adjusted_range'] = (data['high'] - data['low']) * data['volume']
    data['dispersion_per_unit_volume'] = (data['high'] - data['low']) / (data['volume'] + 1e-8)
    data['volume_dispersion_ratio'] = data['volume'] / ((data['high'] - data['low']) + 1e-8)
    
    # Volume Flow Dynamics
    data['relative_volume_surge'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['volume_acceleration'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) / (data['volume'].shift(1) / (data['volume'].shift(2) + 1e-8) + 1e-8)
    
    # Volume persistence calculation
    volume_persistence = []
    for i in range(len(data)):
        if i >= 2:
            count = sum([1 for j in range(i-2, i+1) if data['volume'].iloc[j] > data['volume'].iloc[j-1]])
            volume_persistence.append(count / 3)
        else:
            volume_persistence.append(np.nan)
    data['volume_persistence'] = volume_persistence
    
    # Dispersion-Volume Alignment
    data['high_dispersion_high_volume'] = ((data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1))) & (data['volume'] > data['volume'].shift(1))
    data['low_dispersion_high_volume'] = ((data['high'] - data['low']) < (data['high'].shift(1) - data['low'].shift(1))) & (data['volume'] > data['volume'].shift(1))
    data['dispersion_volume_correlation'] = np.sign(data['high'] - data['low'] - (data['high'].shift(1) - data['low'].shift(1))) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Momentum Convergence Signals
    # Price Momentum Components
    data['raw_price_momentum'] = data['close'] / (data['close'].shift(1) + 1e-8) - 1
    data['momentum_acceleration'] = (data['close'] / (data['close'].shift(1) + 1e-8) - 1) / (data['close'].shift(1) / (data['close'].shift(2) + 1e-8) - 1 + 1e-8)
    
    # Momentum persistence calculation
    momentum_persistence = []
    for i in range(len(data)):
        if i >= 2:
            count = sum([1 for j in range(i-2, i+1) if np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) == np.sign(data['close'].iloc[j-1] - data['close'].iloc[j-2])])
            momentum_persistence.append(count / 3)
        else:
            momentum_persistence.append(np.nan)
    data['momentum_persistence'] = momentum_persistence
    
    # Volume-Momentum Integration
    data['volume_weighted_momentum'] = (data['close'] / (data['close'].shift(1) + 1e-8) - 1) * data['volume']
    data['momentum_per_unit_volume'] = (data['close'] / (data['close'].shift(1) + 1e-8) - 1) / (data['volume'] + 1e-8)
    data['volume_momentum_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Dispersion-Momentum Convergence
    data['high_momentum_low_dispersion'] = (np.abs(data['close'] - data['close'].shift(1)) > np.abs(data['close'].shift(1) - data['close'].shift(2))) & (data['high'] - data['low'] < data['high'].shift(1) - data['low'].shift(1))
    data['momentum_dispersion_ratio'] = np.abs(data['close'] - data['close'].shift(1)) / ((data['high'] - data['low']) + 1e-8)
    data['convergence_efficiency'] = np.abs(data['close'] - data['close'].shift(1)) / ((data['high'] - data['low']) * data['volume'] + 1e-8)
    
    # Trade Size Dynamics
    # Average Trade Size Patterns
    data['raw_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_momentum'] = (data['amount'] / (data['volume'] + 1e-8)) / (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + 1e-8)
    data['trade_size_volatility'] = np.abs((data['amount'] / (data['volume'] + 1e-8)) - (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8))) / (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + 1e-8)
    
    # Trade Size-Dispersion Interaction
    data['large_trade_dispersion'] = (data['amount'] / (data['volume'] + 1e-8)) * (data['high'] - data['low'])
    data['trade_size_per_unit_range'] = (data['amount'] / (data['volume'] + 1e-8)) / ((data['high'] - data['low']) + 1e-8)
    data['trade_size_range_alignment'] = np.sign(data['amount'] / (data['volume'] + 1e-8) - data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8)) * np.sign(data['high'] - data['low'] - (data['high'].shift(1) - data['low'].shift(1)))
    
    # Trade Size-Momentum Integration
    data['large_trade_momentum'] = (data['close'] - data['close'].shift(1)) * (data['amount'] / (data['volume'] + 1e-8))
    data['trade_size_momentum_efficiency'] = (data['close'] - data['close'].shift(1)) / ((data['amount'] / (data['volume'] + 1e-8)) + 1e-8)
    data['trade_size_momentum_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['amount'] / (data['volume'] + 1e-8) - data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8))
    
    # Multi-timeframe Validation
    # Short-term Pattern Consistency
    # 3-day dispersion trend calculation
    dispersion_trend = []
    for i in range(len(data)):
        if i >= 2:
            count = sum([1 for j in range(i-2, i+1) if data['high'].iloc[j] - data['low'].iloc[j] > data['high'].iloc[j-1] - data['low'].iloc[j-1]])
            dispersion_trend.append(count / 3)
        else:
            dispersion_trend.append(np.nan)
    data['dispersion_trend_3d'] = dispersion_trend
    
    # 3-day volume trend calculation
    volume_trend = []
    for i in range(len(data)):
        if i >= 2:
            count = sum([1 for j in range(i-2, i+1) if data['volume'].iloc[j] > data['volume'].iloc[j-1]])
            volume_trend.append(count / 3)
        else:
            volume_trend.append(np.nan)
    data['volume_trend_3d'] = volume_trend
    
    # 3-day momentum consistency calculation
    momentum_consistency = []
    for i in range(len(data)):
        if i >= 2:
            count = sum([1 for j in range(i-2, i+1) if np.sign(data['close'].iloc[j] - data['close'].iloc[j-1]) == np.sign(data['close'].iloc[j-1] - data['close'].iloc[j-2])])
            momentum_consistency.append(count / 3)
        else:
            momentum_consistency.append(np.nan)
    data['momentum_consistency_3d'] = momentum_consistency
    
    # Regime Detection
    data['high_volatility_regime'] = (data['high'] - data['low']) > ((data['high'].shift(1) - data['low'].shift(1)) + 
                                                                     (data['high'].shift(2) - data['low'].shift(2)) + 
                                                                     (data['high'].shift(3) - data['low'].shift(3))) / 3
    data['high_volume_regime'] = data['volume'] > (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3
    data['strong_momentum_regime'] = np.abs(data['close'] - data['close'].shift(1)) > np.abs(data['close'].shift(1) - data['close'].shift(2))
    
    # Signal Confirmation
    data['dispersion_momentum_confirmation'] = ((data['high'] - data['low']) > (data['high'].shift(1) - data['low'].shift(1))) & (np.abs(data['close'] - data['close'].shift(1)) > np.abs(data['close'].shift(1) - data['close'].shift(2)))
    data['volume_momentum_confirmation'] = (data['volume'] > data['volume'].shift(1)) & (np.sign(data['close'] - data['close'].shift(1)) == np.sign(data['volume'] - data['volume'].shift(1)))
    data['trade_size_confirmation'] = ((data['amount'] / (data['volume'] + 1e-8)) > (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8))) & (np.sign(data['close'] - data['close'].shift(1)) == np.sign(data['amount'] / (data['volume'] + 1e-8) - data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8)))
    
    # Alpha Construction
    # Core Convergence Factors
    data['efficient_momentum'] = data['raw_price_momentum'] * data['momentum_dispersion_ratio']
    data['volume_aligned_dispersion'] = data['net_dispersion_asymmetry'] * data['volume_momentum_alignment']
    data['trade_size_convergence'] = data['large_trade_momentum'] * data['trade_size_momentum_alignment']
    
    # Regime-Adaptive Weighting
    data['volatility_weighted_factor'] = data['efficient_momentum'] * (1 + data['high_volatility_regime'])
    data['volume_weighted_factor'] = data['volume_aligned_dispersion'] * (1 + data['high_volume_regime'])
    data['momentum_weighted_factor'] = data['trade_size_convergence'] * (1 + data['strong_momentum_regime'])
    
    # Final Alpha Components
    data['primary_alpha'] = data['volatility_weighted_factor'] * data['dispersion_momentum_confirmation']
    data['secondary_alpha'] = data['volume_weighted_factor'] * data['volume_momentum_confirmation']
    data['composite_alpha'] = data['primary_alpha'] + data['secondary_alpha'] + data['momentum_weighted_factor']
    
    # Return the composite alpha factor
    return data['composite_alpha']
