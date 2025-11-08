import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Fractal Decomposition
    # Multi-Scale Volatility Components
    data['micro_vol'] = (data['high'] - data['low']) / data['close']
    data['intraday_mom_vol'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['overnight_gap_vol'] = np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Volatility Persistence Patterns
    data['vol_momentum'] = (data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2))
    
    # Calculate volatility regime stability
    vol_ranges = data['high'] - data['low']
    vol_ma_5 = vol_ranges.rolling(window=5, min_periods=1).mean()
    vol_stability = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            count = sum((vol_ranges.iloc[i-j] > vol_ma_5.iloc[i-j]) for j in range(5))
            vol_stability.iloc[i] = count
        else:
            vol_stability.iloc[i] = np.nan
    data['vol_regime_stability'] = vol_stability
    
    # Volatility clustering detection
    vol_clustering = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            signs_sum = sum(np.sign((vol_ranges.iloc[i-j] - vol_ranges.iloc[i-j-1])) for j in range(4))
            vol_clustering.iloc[i] = signs_sum
        else:
            vol_clustering.iloc[i] = np.nan
    data['vol_clustering'] = vol_clustering
    
    # Volatility Asymmetry Analysis
    data['upside_vol_intensity'] = (data['high'] - data['open']) / (data['high'] - data['low'])
    data['downside_vol_intensity'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['vol_asymmetry_ratio'] = data['upside_vol_intensity'] / data['downside_vol_intensity']
    
    # Volume Flow Fractal Dynamics
    # Volume Distribution Analysis
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).sum()
    
    # Volume flow persistence
    vol_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            count = sum(data['volume'].iloc[i-j] > data['volume'].iloc[i-j-1] for j in range(4))
            vol_persistence.iloc[i] = count
        else:
            vol_persistence.iloc[i] = np.nan
    data['volume_flow_persistence'] = vol_persistence
    
    data['volume_burst'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).median()
    
    # Volume-Volatility Interaction
    data['volume_efficiency'] = (data['close'] - data['close'].shift(1)) / (data['volume'] * (data['high'] - data['low']))
    data['vol_adj_volume_flow'] = data['volume'] / (data['high'] - data['low'])
    
    vol_range_change = (data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))
    volume_change = data['volume'] - data['volume'].shift(1)
    data['volume_volatility_alignment'] = np.sign(volume_change) * np.sign(vol_range_change)
    
    # Volume Regime Classification
    vol_median_10 = data['volume'].rolling(window=10, min_periods=1).median()
    data['high_flow_regime'] = (data['volume'] > 2 * vol_median_10).astype(float)
    data['low_flow_regime'] = (data['volume'] < 0.5 * vol_median_10).astype(float)
    data['transition_flow'] = ((data['volume_concentration'] < 0.3) & (data['volume_flow_persistence'] > 3)).astype(float)
    
    # Fractal Regime Momentum Signals
    # Multi-Scale Momentum Integration
    data['micro_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Short-term momentum
    vol_sum_3 = (data['high'] - data['low']).rolling(window=3, min_periods=1).sum()
    data['short_term_momentum'] = (data['close'] - data['close'].shift(3)) / vol_sum_3
    
    # Momentum volatility adjustment
    vol_avg_5 = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['momentum_vol_adjustment'] = (data['close'] - data['close'].shift(5)) / vol_avg_5
    
    # Regime Transition Detection
    data['vol_regime_shift'] = np.abs(data['vol_asymmetry_ratio'] - data['vol_asymmetry_ratio'].shift(3))
    data['volume_flow_transition'] = data['volume_concentration'] / data['volume_concentration'].shift(2)
    data['momentum_regime_confirmation'] = data['micro_momentum'] * np.sign(data['short_term_momentum'])
    
    # Composite Fractal Factor Synthesis
    # Regime-Weighted Signal Generation
    high_flow_signal = data['volume_efficiency'] * data['vol_asymmetry_ratio']
    low_flow_signal = data['volume_concentration'] * data['momentum_vol_adjustment']
    transition_signal = data['volume_volatility_alignment'] * data['vol_regime_shift']
    
    # Apply regime weights
    regime_weighted = (data['high_flow_regime'] * high_flow_signal + 
                      data['low_flow_regime'] * low_flow_signal + 
                      data['transition_flow'] * transition_signal)
    
    # Multi-Scale Signal Integration
    volatility_persistence_weighted = data['vol_regime_stability'] * data['short_term_momentum']
    volume_flow_adjusted = data['volume_flow_persistence'] * data['micro_vol']
    regime_stability_enhanced = data['vol_clustering'] * data['micro_momentum']
    
    # Final composite factor
    composite_factor = (regime_weighted + 
                       volatility_persistence_weighted + 
                       volume_flow_adjusted + 
                       regime_stability_enhanced)
    
    # Normalize and clean
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan)
    composite_factor = (composite_factor - composite_factor.rolling(window=20, min_periods=1).mean()) / composite_factor.rolling(window=20, min_periods=1).std()
    
    return composite_factor
