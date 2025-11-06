import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Momentum with Volatility Scaling and Volume-Price Convergence factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum Framework
    # Multi-timeframe Returns
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ret_8d'] = data['close'] / data['close'].shift(8) - 1
    data['ret_13d'] = data['close'] / data['close'].shift(13) - 1
    
    # Momentum Consistency
    returns_matrix = data[['ret_3d', 'ret_8d', 'ret_13d']]
    data['directional_agreement'] = (returns_matrix > 0).sum(axis=1)
    data['magnitude_consistency'] = returns_matrix.std(axis=1)
    data['momentum_strength'] = returns_matrix.mean(axis=1)
    
    # Momentum Quality Score
    data['momentum_quality_score'] = (
        data['momentum_strength'] * 
        (1 - data['magnitude_consistency']) * 
        (data['directional_agreement'] / 3)
    )
    
    # 2. Enhanced Volatility Scaling
    # Dynamic Volatility Measures
    data['gk_vol'] = np.sqrt(
        0.5 * (np.log(data['high'] / data['low']))**2 - 
        (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']))**2
    )
    data['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(data['high'] / data['low']))**2
    )
    data['rs_vol'] = np.sqrt(
        np.log(data['high'] / data['close']) * np.log(data['high'] / data['open']) + 
        np.log(data['low'] / data['close']) * np.log(data['low'] / data['open'])
    )
    
    # Volatility Regime Detection
    data['vol_ratio'] = data['gk_vol'] / data['parkinson_vol']
    
    # Volatility persistence (5-day rolling correlation)
    vol_window = 5
    gk_vol_series = data['gk_vol']
    vol_persistence = []
    for i in range(len(data)):
        if i >= vol_window:
            window1 = gk_vol_series.iloc[i-vol_window:i-1]
            window2 = gk_vol_series.iloc[i-vol_window+1:i]
            if len(window1) > 1 and len(window2) > 1:
                corr = np.corrcoef(window1, window2)[0, 1]
                vol_persistence.append(corr if not np.isnan(corr) else 0)
            else:
                vol_persistence.append(0)
        else:
            vol_persistence.append(0)
    data['vol_persistence'] = vol_persistence
    
    data['regime_score'] = data['vol_ratio'] * (1 + data['vol_persistence'])
    
    # Adaptive Volatility Scaling
    data['core_momentum'] = (
        data['momentum_quality_score'] / 
        (data['gk_vol'] * data['regime_score'])
    ).replace([np.inf, -np.inf], 0).fillna(0)
    
    # 3. Volume-Price Convergence Analysis
    # Volume Dynamics
    data['volume_velocity'] = (
        (data['volume'] / data['volume'].shift(3) - 1) - 
        (data['volume'].shift(3) / data['volume'].shift(6) - 1)
    )
    
    # Volume persistence (5-day rolling correlation)
    volume_series = data['volume']
    vol_persistence_list = []
    for i in range(len(data)):
        if i >= vol_window:
            window1 = volume_series.iloc[i-vol_window:i-1]
            window2 = volume_series.iloc[i-vol_window+1:i]
            if len(window1) > 1 and len(window2) > 1:
                corr = np.corrcoef(window1, window2)[0, 1]
                vol_persistence_list.append(corr if not np.isnan(corr) else 0)
            else:
                vol_persistence_list.append(0)
        else:
            vol_persistence_list.append(0)
    data['volume_persistence'] = vol_persistence_list
    
    # Volume stability (5-day rolling inverse std)
    data['volume_stability'] = 1 / data['volume'].rolling(window=5).std()
    data['volume_stability'] = data['volume_stability'].replace([np.inf, -np.inf], 1).fillna(1)
    
    # Price-Volume Relationship
    # Volume-return correlation (5-day rolling)
    returns_5d = data['close'].pct_change()
    volume_return_corr = []
    for i in range(len(data)):
        if i >= vol_window:
            ret_window = returns_5d.iloc[i-vol_window:i]
            vol_window_data = volume_series.iloc[i-vol_window:i]
            if len(ret_window) > 1 and len(vol_window_data) > 1:
                corr = np.corrcoef(ret_window, vol_window_data)[0, 1]
                volume_return_corr.append(corr if not np.isnan(corr) else 0)
            else:
                volume_return_corr.append(0)
        else:
            volume_return_corr.append(0)
    data['volume_return_correlation'] = volume_return_corr
    
    data['volume_leadership'] = (
        np.sign(data['volume_velocity']) * 
        np.sign(data['momentum_strength'])
    )
    data['convergence_strength'] = (
        np.abs(data['volume_return_correlation']) * 
        data['volume_leadership']
    )
    
    # Volume Confirmation Score
    data['volume_confirmation_score'] = (
        data['convergence_strength'] * 
        data['volume_persistence'] * 
        data['volume_stability']
    )
    
    # 4. Factor Synthesis
    data['volume_enhancement'] = 1 + (0.25 * data['volume_confirmation_score'])
    data['final_alpha'] = data['core_momentum'] * data['volume_enhancement']
    
    # Return the final alpha factor series
    return data['final_alpha']
