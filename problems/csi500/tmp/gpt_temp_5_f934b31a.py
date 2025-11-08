import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor based on dynamic momentum-volume divergence with volatility regime adaptation,
    volume anomaly detection, intraday efficiency scoring, and gap behavior analysis.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Dynamic Momentum-Volume Divergence
    # Multi-timeframe Momentum Analysis
    data['mom_3d_price'] = data['close'] / data['close'].shift(3)
    data['mom_10d_price'] = data['close'] / data['close'].shift(10)
    data['mom_3d_volume'] = data['volume'] / data['volume'].shift(3)
    data['mom_10d_volume'] = data['volume'] / data['volume'].shift(10)
    
    # Adaptive Divergence Calculation
    data['div_short'] = data['mom_3d_price'] - data['mom_3d_volume']
    data['div_medium'] = data['mom_10d_price'] - data['mom_10d_volume']
    data['div_persistence'] = np.sign(data['div_short']) * np.sign(data['div_medium'])
    
    # Volatility-Regime Adaptive Weighting
    # Volatility Assessment
    data['returns'] = np.log(data['close'] / data['close'].shift(1))
    data['vol_recent'] = data['returns'].rolling(window=5).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    data['vol_baseline'] = data['returns'].rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    
    # Dynamic Component Selection
    def select_primary_component(row):
        if row['vol_recent'] > 2 * row['vol_baseline']:
            return row['div_short']
        elif row['vol_recent'] >= 0.5 * row['vol_baseline'] and row['vol_recent'] <= 2 * row['vol_baseline']:
            return (row['div_short'] + row['div_medium']) / 2
        else:
            return row['div_medium']
    
    data['primary_component'] = data.apply(select_primary_component, axis=1)
    
    # Robust Volume Anomaly Detection
    # Volume Distribution Analysis
    data['volume_median'] = data['volume'].rolling(window=20).median()
    data['volume_mad'] = data['volume'].rolling(window=20).apply(lambda x: np.median(np.abs(x - np.median(x))))
    data['volume_zscore'] = (data['volume'] - data['volume_median']) / data['volume_mad']
    
    # Dynamic Spike Classification
    def classify_volume_anomaly(zscore):
        if zscore > 4:
            return 'extreme'
        elif zscore > 2:
            return 'strong'
        elif zscore > 1:
            return 'moderate'
        else:
            return 'normal'
    
    data['volume_class'] = data['volume_zscore'].apply(classify_volume_anomaly)
    
    # Volume-Enhanced Component
    def apply_volume_multiplier(row):
        if row['volume_class'] == 'extreme':
            return row['primary_component'] * 2.5
        elif row['volume_class'] == 'strong':
            return row['primary_component'] * 1.8
        elif row['volume_class'] == 'moderate':
            return row['primary_component'] * 1.3
        else:
            return row['primary_component'] * 1.0
    
    data['volume_enhanced'] = data.apply(apply_volume_multiplier, axis=1)
    
    # Intraday Price Efficiency Scoring
    # Range-Based Metrics
    data['normalized_range'] = (data['high'] - data['low']) / data['close']
    data['close_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Efficiency Classification
    def classify_efficiency(row):
        if row['close_efficiency'] > 0.8 and row['normalized_range'] > 0.015:
            return 'high'
        elif row['close_efficiency'] > 0.6 and row['normalized_range'] > 0.01:
            return 'medium'
        elif row['close_efficiency'] < 0.4 or row['normalized_range'] < 0.005:
            return 'low'
        else:
            return 'neutral'
    
    data['efficiency_class'] = data.apply(classify_efficiency, axis=1)
    
    # Efficiency-Adjusted Component
    def apply_efficiency_multiplier(row):
        if row['efficiency_class'] == 'high':
            return row['volume_enhanced'] * 1.4
        elif row['efficiency_class'] == 'medium':
            return row['volume_enhanced'] * 1.2
        elif row['efficiency_class'] == 'low':
            return row['volume_enhanced'] * 0.8
        else:
            return row['volume_enhanced'] * 1.0
    
    data['efficiency_adjusted'] = data.apply(apply_efficiency_multiplier, axis=1)
    
    # Gap Behavior Analysis
    # Gap Metrics
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_fill'] = (data['close'] - data['open']) / data['open']
    data['gap_persistence'] = np.sign(data['overnight_gap']) * np.sign(data['intraday_fill'])
    
    # Gap Pattern Classification
    def classify_gap_pattern(row):
        if abs(row['overnight_gap']) > 0.03 and np.sign(row['overnight_gap']) == np.sign(row['intraday_fill']):
            return 'breakaway'
        elif abs(row['overnight_gap']) > 0.03 and np.sign(row['overnight_gap']) != np.sign(row['intraday_fill']):
            return 'exhaustion'
        else:
            return 'common'
    
    data['gap_class'] = data.apply(classify_gap_pattern, axis=1)
    
    # Final Alpha Factor
    def calculate_final_alpha(row):
        base_component = row['efficiency_adjusted']
        
        # Apply gap pattern multiplier
        if row['gap_class'] == 'breakaway':
            base_component *= 1.5
        elif row['gap_class'] == 'exhaustion':
            base_component *= 0.7
        # common gap: no multiplier (1.0)
        
        # Apply persistence multiplier
        persistence_multiplier = 1 + 0.2 * row['gap_persistence']
        base_component *= persistence_multiplier
        
        return base_component
    
    data['alpha_factor'] = data.apply(calculate_final_alpha, axis=1)
    
    return data['alpha_factor']
