import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Asymmetry Momentum factor
    """
    data = df.copy()
    
    # True Range Calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Regime Detection
    data['tr_10day_avg'] = data['tr'].rolling(window=10, min_periods=1).mean()
    data['tr_ratio'] = data['tr'] / data['tr_10day_avg']
    
    def get_vol_regime(ratio):
        if ratio > 1.5:
            return 2  # High Volatility
        elif ratio < 0.7:
            return 0  # Low Volatility
        else:
            return 1  # Normal Volatility
    
    data['vol_regime'] = data['tr_ratio'].apply(get_vol_regime)
    
    # Price Asymmetry Decomposition
    data['hl_range'] = data['high'] - data['low']
    data['hl_range'] = np.where(data['hl_range'] == 0, 1e-6, data['hl_range'])  # Avoid division by zero
    
    # Directional Pressure Components
    data['upward_pressure'] = (data['high'] - data['open']) / data['hl_range']
    data['downward_pressure'] = (data['open'] - data['low']) / data['hl_range']
    data['net_directional_bias'] = (data['upward_pressure'] - data['downward_pressure']) / (data['upward_pressure'] + data['downward_pressure'] + 1e-6)
    
    # Intraday Price Path Analysis
    data['price_path_efficiency'] = (data['close'] - data['open']) / data['hl_range']
    data['opening_gap_persistence'] = np.sign(data['open'] - data['prev_close']) * np.sign(data['close'] - data['open'])
    data['midday_reversal'] = ((data['high'] + data['low']) / 2 - data['open']) / data['hl_range']
    
    # Multi-Timeframe Asymmetry
    data['overnight_intraday_ratio'] = (data['open'] - data['prev_close']) / (data['close'] - data['open'] + 1e-6)
    data['early_late_bias'] = (data['high'] - data['open']) / (data['close'] - data['low'] + 1e-6)
    
    # Asymmetry persistence (rolling 5-day)
    def calc_asymmetry_persistence(series):
        if len(series) < 5:
            return 0
        current_sign = np.sign(series.iloc[-1])
        matches = sum(np.sign(series.iloc[i]) == current_sign for i in range(len(series)-1))
        return matches / 4.0
    
    data['asymmetry_persistence'] = data['net_directional_bias'].rolling(window=5, min_periods=1).apply(
        calc_asymmetry_persistence, raw=False
    )
    
    # Volume Asymmetry Dynamics
    # Volume Distribution Analysis
    data['volume_max_5d'] = data['volume'].rolling(window=5, min_periods=1).max()
    data['volume_concentration'] = data['volume'] / (data['volume_max_5d'] + 1e-6)
    
    data['volume_avg_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_flow_directionality'] = data['volume'] * np.sign(data['close'] - data['open']) / (data['volume_avg_5d'] + 1e-6)
    
    data['volume_median_10d'] = data['volume'].rolling(window=10, min_periods=1).median()
    data['volume_max_10d'] = data['volume'].rolling(window=10, min_periods=1).max()
    data['volume_min_10d'] = data['volume'].rolling(window=10, min_periods=1).min()
    data['volume_burst_intensity'] = (data['volume'] - data['volume_median_10d']) / (data['volume_max_10d'] - data['volume_min_10d'] + 1e-6)
    
    # Volume-Price Alignment
    data['confirmed_volume_pressure'] = data['volume'] * (data['close'] - data['open']) / (data['hl_range'] + 1e-6)
    data['contrarian_volume_signals'] = data['volume'] * np.sign(data['open'] - data['prev_close']) / (data['hl_range'] + 1e-6)
    data['volume_path_consistency'] = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Volume Regime Patterns
    data['high_volume_directional_bias'] = (data['volume'] > 1.5 * data['volume_median_10d']).astype(int) * data['net_directional_bias']
    data['low_volume_compression'] = (data['volume'] < 0.7 * data['volume_median_10d']).astype(int) * data['price_path_efficiency']
    data['volume_transition'] = abs(data['volume_concentration'] - data['volume_concentration'].shift(2)).fillna(0)
    
    # Asymmetry Convergence Analysis
    # Price-Volume Alignment Strength
    data['direction_consistency'] = np.sign(data['net_directional_bias']) * np.sign(data['volume_flow_directionality'])
    data['magnitude_correlation'] = data['net_directional_bias'] * data['volume_flow_directionality']
    data['path_volume_alignment'] = data['price_path_efficiency'] * data['volume_path_consistency']
    
    # Multi-timeframe Asymmetry Convergence
    def calc_fractal_alignment(row):
        components = [
            np.sign(row['net_directional_bias']),
            np.sign(row['price_path_efficiency']),
            np.sign(row['volume_flow_directionality']),
            np.sign(row['opening_gap_persistence'])
        ]
        if len(components) < 2:
            return 0
        matches = sum(c1 == c2 for i, c1 in enumerate(components) for j, c2 in enumerate(components) if i < j)
        return matches / 6.0  # 6 possible pairs
    
    data['fractal_alignment_score'] = data.apply(calc_fractal_alignment, axis=1)
    
    data['convergence_strength'] = (
        abs(data['net_directional_bias']) * 
        abs(data['price_path_efficiency']) * 
        abs(data['volume_flow_directionality'])
    )
    
    data['asymmetry_volatility'] = data['net_directional_bias'].rolling(window=5, min_periods=1).std()
    
    # Range Compression Breakout Signals
    data['prev_hl_range'] = data['hl_range'].shift(1)
    data['range_ratio'] = data['hl_range'] / (data['prev_hl_range'] + 1e-6)
    data['range_compression'] = 1 / (data['range_ratio'] + 1e-6)
    
    data['volume_5day_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_significance'] = data['volume'] / (data['volume_5day_avg'] + 1e-6)
    
    data['intraday_position'] = (data['close'] - data['low']) / (data['hl_range'] + 1e-6)
    data['breakout_strength'] = data['range_compression'] * data['volume_significance']
    
    # Regime-Adaptive Signal Integration
    def calculate_regime_factor(row):
        if row['vol_regime'] == 2:  # High Volatility
            primary = row['magnitude_correlation']
            secondary = row['volume_burst_intensity'] * row['breakout_strength']
            weight = row['convergence_strength'] * row['volume_significance']
            
        elif row['vol_regime'] == 1:  # Normal Volatility
            primary = row['fractal_alignment_score']
            secondary = row['price_path_efficiency'] * row['volume_path_consistency']
            weight = row['fractal_alignment_score'] * row['asymmetry_persistence']
            
        else:  # Low Volatility
            primary = row['breakout_strength']
            secondary = row['volume_transition'] * row['asymmetry_volatility']
            weight = row['breakout_strength'] * abs(row['direction_consistency'])
        
        return (primary + 0.5 * secondary) * weight
    
    data['regime_factor'] = data.apply(calculate_regime_factor, axis=1)
    
    # Quality-Adjusted Factor Generation
    data['convergence_quality'] = data['fractal_alignment_score'] * data['convergence_strength']
    data['volume_confirmation'] = data['volume_significance'] * abs(data['direction_consistency'])
    data['position_strength'] = data['intraday_position'] * data['breakout_strength']
    
    def calculate_quality_score(row):
        quality_components = [
            row['convergence_quality'],
            row['volume_confirmation'],
            row['position_strength'],
            row['asymmetry_persistence']
        ]
        return np.mean([c for c in quality_components if not np.isnan(c)])
    
    data['quality_score'] = data.apply(calculate_quality_score, axis=1)
    
    # Final Composite Factor with Quality Adjustment
    data['final_factor'] = data['regime_factor'] * data['quality_score']
    
    # Clean up and return
    result = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
