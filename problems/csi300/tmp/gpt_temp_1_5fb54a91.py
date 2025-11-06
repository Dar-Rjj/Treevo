import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Market Regime Classification
    # Opening efficiency
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['open_efficiency'] = np.abs(data['open'] - data['prev_close']) / (data['prev_high'] - data['prev_low'] + 1e-8)
    
    # Closing efficiency
    data['hl_midpoint'] = (data['high'] + data['low']) / 2
    data['close_efficiency'] = np.abs(data['close'] - data['hl_midpoint']) / (data['high'] - data['low'] + 1e-8)
    
    # Regime classification based on efficiency patterns
    data['efficiency_ratio'] = data['open_efficiency'] / (data['close_efficiency'] + 1e-8)
    data['regime'] = np.where(data['efficiency_ratio'] > 1.2, 2,  # High opening efficiency regime
                     np.where(data['efficiency_ratio'] < 0.8, 0,  # High closing efficiency regime
                              1))  # Balanced regime
    
    # Volume Flow Analysis
    # Volume-weighted directional flow
    data['price_change'] = data['close'] - data['open']
    data['volume_flow'] = data['price_change'] * data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Volume flow persistence over 3 days
    data['volume_flow_ma3'] = data['volume_flow'].rolling(window=3, min_periods=1).mean()
    data['volume_persistence'] = data['volume_flow'] / (data['volume_flow_ma3'] + 1e-8)
    
    # Price-volume divergence detection
    data['price_trend'] = data['close'].rolling(window=5, min_periods=1).mean()
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['pv_divergence'] = np.where(
        (data['price_trend'] > data['price_trend'].shift(1)) & (data['volume_trend'] < data['volume_trend'].shift(1)), -1,
        np.where((data['price_trend'] < data['price_trend'].shift(1)) & (data['volume_trend'] > data['volume_trend'].shift(1)), 1, 0)
    )
    
    # Multi-timeframe Order Imbalance
    # Opening gap and volume intensity
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['opening_volume_intensity'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    
    # Intraday volume distribution
    data['intraday_volume_ratio'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=1).mean() + 1e-8)
    
    # Closing auction pressure
    data['closing_pressure'] = (data['close'] - data['hl_midpoint']) * data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Price Discovery Latency
    # Price adjustment delay metrics
    data['price_adjustment'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['discovery_latency'] = 1 - data['price_adjustment']
    
    # Adaptive Composite Factor
    # Regime-classified signal weighting
    regime_weights = {
        0: {'volume_flow': 0.3, 'pv_divergence': 0.4, 'closing_pressure': 0.3},  # Closing efficiency regime
        1: {'volume_flow': 0.4, 'opening_gap': 0.3, 'discovery_latency': 0.3},   # Balanced regime
        2: {'opening_gap': 0.5, 'volume_persistence': 0.3, 'intraday_volume_ratio': 0.2}  # Opening efficiency regime
    }
    
    # Calculate regime-specific factors
    factors = []
    for idx, row in data.iterrows():
        regime = row['regime']
        weights = regime_weights[regime]
        
        factor_value = 0
        for component, weight in weights.items():
            if component == 'pv_divergence':
                factor_value += row[component] * weight
            else:
                # Normalize other components
                normalized_component = (row[component] - data[component].mean()) / (data[component].std() + 1e-8)
                factor_value += normalized_component * weight
        
        factors.append(factor_value)
    
    # Create final factor series
    factor_series = pd.Series(factors, index=data.index, name='regime_aware_efficiency_factor')
    
    # Clean up intermediate columns
    drop_cols = ['prev_close', 'prev_high', 'prev_low', 'hl_midpoint', 'price_change', 
                'volume_flow_ma3', 'price_trend', 'volume_trend']
    data.drop(columns=drop_cols, inplace=True)
    
    return factor_series
