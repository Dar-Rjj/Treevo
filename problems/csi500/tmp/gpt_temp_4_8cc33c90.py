import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    # Calculate rolling average of TR for regime detection
    data['TR_avg_10'] = data['TR'].rolling(window=10, min_periods=1).mean()
    
    # Regime Detection
    conditions = [
        data['TR'] > (1.5 * data['TR_avg_10']),  # High volatility
        data['TR'] < (0.7 * data['TR_avg_10']),  # Low volatility
    ]
    choices = [2, 0]  # 2=High, 1=Normal, 0=Low
    data['regime'] = np.select(conditions, choices, default=1)
    
    # Microstructure Momentum Components
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_momentum'] = (data['open'] - data['prev_close']) / data['prev_close'].replace(0, np.nan)
    data['directional_bias'] = ((data['high'] - data['open']) - (data['open'] - data['low'])) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume-Efficiency Dynamics
    data['volume_efficiency'] = np.abs(data['close'] - data['open']) / data['volume'].replace(0, np.nan)
    data['volume_flow'] = np.sign(data['close'] - data['open']) * data['volume']
    data['volume_alignment'] = np.sign(data['directional_bias']) * np.sign(data['volume_flow'])
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Complexity Analysis
    # Price Fractal Dimension
    data['price_change_sum'] = np.abs(data['close'].diff()).rolling(window=5, min_periods=1).sum()
    data['range_sum'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).sum()
    data['PFD'] = 1 + np.log(data['price_change_sum'].replace(0, np.nan)) / np.log(data['range_sum'].replace(0, np.nan))
    
    # Range Compression
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['range_compression'] = (data['prev_high'] - data['prev_low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Complexity Momentum
    data['complexity_momentum'] = data['PFD'] / data['PFD'].shift(3).replace(0, np.nan) - 1
    
    # Regime-Specific Integration
    high_vol_signal = data['volume_efficiency'] * data['range_compression'] * data['complexity_momentum']
    normal_vol_signal = data['intraday_momentum'] * data['volume_ratio'] * data['directional_bias']
    low_vol_signal = data['gap_momentum'] * data['volume_alignment'] * data['PFD']
    
    conditions = [
        data['regime'] == 2,  # High volatility
        data['regime'] == 0,  # Low volatility
    ]
    choices = [high_vol_signal, low_vol_signal]
    data['core_signal'] = np.select(conditions, choices, default=normal_vol_signal)
    
    # Quality Assessment - count matching signs across momentum components
    momentum_signs = pd.DataFrame({
        'intraday': np.sign(data['intraday_momentum']),
        'gap': np.sign(data['gap_momentum']),
        'directional': np.sign(data['directional_bias'])
    })
    data['quality_score'] = momentum_signs.apply(lambda x: sum(x == x.iloc[0]) if not x.isna().all() else 1, axis=1)
    
    # Final Factor
    data['factor'] = data['core_signal'] * data['quality_score']
    
    return data['factor']
