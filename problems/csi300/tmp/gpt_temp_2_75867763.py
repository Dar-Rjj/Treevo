import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure proper data types and handle potential division by zero
    data = data.astype(float)
    
    # Calculate basic price differences and ratios
    data['close_prev'] = data['close'].shift(1)
    data['high_prev'] = data['high'].shift(1)
    data['low_prev'] = data['low'].shift(1)
    data['volume_prev'] = data['volume'].shift(1)
    data['close_prev2'] = data['close'].shift(2)
    data['close_prev4'] = data['close'].shift(4)
    
    # Handle NaN values from shifts
    data = data.fillna(method='bfill')
    
    # Quantum Asymmetric Price Structure
    # Fractal Amplitude Asymmetry
    data['upward_fractal_amplitude'] = np.where(
        data['close'] > data['close_prev'],
        ((data['high'] - data['low']) / (data['high_prev'] - data['low_prev'])) * 
        ((data['close'] - data['open']) / (data['high'] - data['low'])),
        0
    )
    
    data['downward_fractal_amplitude'] = np.where(
        data['close'] < data['close_prev'],
        ((data['high'] - data['low']) / (data['high_prev'] - data['low_prev'])) * 
        ((data['open'] - data['close']) / (data['high'] - data['low'])),
        0
    )
    
    data['fractal_amplitude_asymmetry'] = (
        data['upward_fractal_amplitude'] - data['downward_fractal_amplitude']
    )
    
    # Price Dimension Asymmetry
    data['high_side_dimension'] = (
        (data['high'] - data['close']) / (data['high'] - data['low']) * 
        (data['close'] - data['close_prev']) / (data['high'] - data['low'])
    )
    
    data['low_side_dimension'] = (
        (data['close'] - data['low']) / (data['high'] - data['low']) * 
        (data['close'] - data['close_prev']) / (data['high'] - data['low'])
    )
    
    data['dimension_asymmetry'] = (
        data['high_side_dimension'] - data['low_side_dimension']
    )
    
    # Fractal Persistence Asymmetry
    data['sign_current'] = np.sign(data['close'] - data['open'])
    data['sign_prev'] = np.sign(data['close_prev'] - data['open'].shift(1))
    
    data['positive_persistence'] = np.where(
        data['sign_current'] == data['sign_prev'],
        (data['volume'] - data['volume_prev']) * (data['close'] - data['close_prev']),
        0
    )
    
    data['negative_persistence'] = np.where(
        data['sign_current'] != data['sign_prev'],
        (data['volume'] - data['volume_prev']) * (data['close'] - data['close_prev']),
        0
    )
    
    data['persistence_asymmetry'] = (
        data['positive_persistence'] - data['negative_persistence']
    )
    
    # Asymmetric Microstructure Patterns
    # Clustering Asymmetry
    mid_point = (data['high'] + data['low']) / 2
    
    data['upper_clustering'] = np.where(
        data['close'] > mid_point,
        (np.abs(data['close'] - mid_point) / (data['high'] - data['low'])) * 
        (data['close'] - data['close_prev']),
        0
    )
    
    data['lower_clustering'] = np.where(
        data['close'] < mid_point,
        (np.abs(data['close'] - mid_point) / (data['high'] - data['low'])) * 
        (data['close'] - data['close_prev']),
        0
    )
    
    data['clustering_asymmetry'] = (
        data['upper_clustering'] - data['lower_clustering']
    )
    
    # Boundary Asymmetry
    data['upper_boundary_strength'] = (
        (data['high'] - data['close']) * (data['close'] - data['low']) / 
        ((data['high'] - data['low']) ** 2) * data['volume']
    )
    
    data['lower_boundary_strength'] = (
        (data['close'] - data['low']) * (data['high'] - data['close']) / 
        ((data['high'] - data['low']) ** 2) * data['volume']
    )
    
    data['boundary_asymmetry'] = (
        data['upper_boundary_strength'] - data['lower_boundary_strength']
    )
    
    # Momentum Asymmetry
    ret_t = data['close'] / data['close_prev'] - 1
    ret_t1 = data['close_prev'] / data['close_prev2'] - 1
    ret_t4 = data['close'] / data['close_prev4'] - 1
    
    data['accelerating_momentum'] = np.where(
        ret_t * ret_t1 > 0,
        (ret_t * ret_t1) / np.where(ret_t4 != 0, ret_t4, 1),
        0
    )
    
    data['decelerating_momentum'] = np.where(
        ret_t * ret_t1 < 0,
        (ret_t * ret_t1) / np.where(ret_t4 != 0, ret_t4, 1),
        0
    )
    
    data['momentum_asymmetry'] = (
        data['accelerating_momentum'] - data['decelerating_momentum']
    )
    
    # Asymmetric Volume Dynamics
    # Volume Scaling Asymmetry
    data['volume_expansion'] = np.where(
        data['volume'] > data['volume_prev'],
        (data['volume'] / data['volume_prev']) * (data['close'] - data['close_prev']),
        0
    )
    
    data['volume_contraction'] = np.where(
        data['volume'] < data['volume_prev'],
        (data['volume'] / data['volume_prev']) * (data['close'] - data['close_prev']),
        0
    )
    
    data['volume_scaling_asymmetry'] = (
        data['volume_expansion'] - data['volume_contraction']
    )
    
    # Volume Persistence Asymmetry
    volume_increase = data['volume'] > data['volume_prev']
    volume_decrease = data['volume'] < data['volume_prev']
    
    # Calculate consecutive counts
    data['pos_volume_persistence'] = volume_increase.astype(int)
    data['neg_volume_persistence'] = volume_decrease.astype(int)
    
    for i in range(1, len(data)):
        if volume_increase.iloc[i]:
            data.iloc[i, data.columns.get_loc('pos_volume_persistence')] = (
                data['pos_volume_persistence'].iloc[i-1] + 1
            )
        if volume_decrease.iloc[i]:
            data.iloc[i, data.columns.get_loc('neg_volume_persistence')] = (
                data['neg_volume_persistence'].iloc[i-1] + 1
            )
    
    data['positive_volume_persistence'] = (
        data['pos_volume_persistence'] * (data['close'] - data['open'])
    )
    data['negative_volume_persistence'] = (
        data['neg_volume_persistence'] * (data['close'] - data['open'])
    )
    
    data['volume_persistence_asymmetry'] = (
        data['positive_volume_persistence'] - data['negative_volume_persistence']
    )
    
    # Volume Field Asymmetry
    data['positive_volume_field'] = np.where(
        data['close'] > data['open'],
        (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low']) * 
        (data['volume'] - data['volume_prev']),
        0
    )
    
    data['negative_volume_field'] = np.where(
        data['close'] < data['open'],
        (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low']) * 
        (data['volume'] - data['volume_prev']),
        0
    )
    
    data['volume_field_asymmetry'] = (
        data['positive_volume_field'] - data['negative_volume_field']
    )
    
    # Asymmetric Entropy Dynamics
    # Price Entropy Asymmetry
    upper_ratio = (data['high'] - data['close']) / (data['high'] - data['low'])
    lower_ratio = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Handle log(0) cases
    upper_ratio_safe = np.where(upper_ratio > 0, upper_ratio, 1e-10)
    lower_ratio_safe = np.where(lower_ratio > 0, lower_ratio, 1e-10)
    
    data['upper_price_entropy'] = (
        -upper_ratio * np.log(upper_ratio_safe) * 
        (data['close'] - data['open']) / (data['high'] - data['low'])
    )
    
    data['lower_price_entropy'] = (
        -lower_ratio * np.log(lower_ratio_safe) * 
        (data['close'] - data['open']) / (data['high'] - data['low'])
    )
    
    data['price_entropy_asymmetry'] = (
        data['upper_price_entropy'] - data['lower_price_entropy']
    )
    
    # Volume Entropy Asymmetry
    volume_change_ratio = (data['volume'] - data['volume_prev']) / data['volume_prev']
    volume_change_ratio_neg = (data['volume_prev'] - data['volume']) / data['volume_prev']
    
    # Handle log cases
    vol_ratio_safe = np.where(np.abs(volume_change_ratio) > 1e-10, volume_change_ratio, 1e-10)
    vol_ratio_neg_safe = np.where(np.abs(volume_change_ratio_neg) > 1e-10, volume_change_ratio_neg, 1e-10)
    
    data['high_volume_entropy'] = (
        -volume_change_ratio * np.log(np.abs(vol_ratio_safe)) * data['volume']
    )
    
    data['low_volume_entropy'] = (
        -volume_change_ratio_neg * np.log(np.abs(vol_ratio_neg_safe)) * data['volume']
    )
    
    data['volume_entropy_asymmetry'] = (
        data['high_volume_entropy'] - data['low_volume_entropy']
    )
    
    # Entropy Convergence Asymmetry
    entropy_product = data['price_entropy_asymmetry'] * data['volume_entropy_asymmetry']
    
    data['positive_entropy_convergence'] = np.where(
        entropy_product > 0,
        entropy_product * (data['close'] - data['close_prev']),
        0
    )
    
    data['negative_entropy_convergence'] = np.where(
        entropy_product < 0,
        entropy_product * (data['close'] - data['close_prev']),
        0
    )
    
    data['convergence_asymmetry'] = (
        data['positive_entropy_convergence'] - data['negative_entropy_convergence']
    )
    
    # Fractal Asymmetry Regimes
    # Efficiency Asymmetry
    data['high_side_efficiency'] = (
        (data['high'] - data['close']) / (data['high'] - data['low']) * 
        data['volume'] / np.abs(data['close'] - data['close_prev']) * 
        (data['close'] - data['open']) / (data['high'] - data['low'])
    )
    
    data['low_side_efficiency'] = (
        (data['close'] - data['low']) / (data['high'] - data['low']) * 
        data['volume'] / np.abs(data['close'] - data['close_prev']) * 
        (data['close'] - data['open']) / (data['high'] - data['low'])
    )
    
    data['efficiency_asymmetry'] = (
        data['high_side_efficiency'] - data['low_side_efficiency']
    )
    
    # Utilization Asymmetry
    data['upper_utilization'] = (
        (data['high'] - data['close']) / (data['high'] - data['low']) * 
        np.abs(data['close'] - data['open']) / (data['high'] - data['low']) * 
        (data['close'] - data['close_prev'])
    )
    
    data['lower_utilization'] = (
        (data['close'] - data['low']) / (data['high'] - data['low']) * 
        np.abs(data['close'] - data['open']) / (data['high'] - data['low']) * 
        (data['close'] - data['close_prev'])
    )
    
    data['utilization_asymmetry'] = (
        data['upper_utilization'] - data['lower_utilization']
    )
    
    # Symmetry Fracture
    symmetry_ratio = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['symmetry_deviation'] = np.abs(symmetry_ratio - 0.5) * data['volume']
    data['symmetry_direction'] = np.sign(symmetry_ratio - 0.5) * data['volume']
    data['symmetry_fracture'] = data['symmetry_deviation'] * data['symmetry_direction']
    
    # Fractal Asymmetry Alpha Synthesis
    # Micro-level Asymmetry
    data['micro_asymmetry_composite'] = (
        (data['fractal_amplitude_asymmetry'] * data['price_entropy_asymmetry'] +
         data['volume_scaling_asymmetry'] * data['volume_entropy_asymmetry']) / 2
    )
    
    # Meso-level Asymmetry
    data['meso_asymmetry_composite'] = (
        (data['clustering_asymmetry'] * data['boundary_asymmetry'] +
         data['momentum_asymmetry'] * data['efficiency_asymmetry']) / 2
    )
    
    # Macro-level Asymmetry
    data['macro_asymmetry_composite'] = (
        (data['persistence_asymmetry'] * data['utilization_asymmetry'] +
         data['convergence_asymmetry'] * data['symmetry_fracture']) / 2
    )
    
    # Final Fractal Asymmetry Alpha
    data['multi_scale_integration'] = (
        (data['micro_asymmetry_composite'] + 
         data['meso_asymmetry_composite'] + 
         data['macro_asymmetry_composite']) / 3
    )
    
    data['volume_confirmation'] = (
        data['multi_scale_integration'] * data['volume_field_asymmetry']
    )
    
    data['final_alpha'] = (
        data['multi_scale_integration'] * data['volume_confirmation']
    )
    
    # Return the final alpha factor series
    return data['final_alpha']
