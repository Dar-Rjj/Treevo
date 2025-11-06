import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Asymmetry Factor
    Combines volume distribution asymmetry with price structure patterns
    across multiple timeframes to predict future returns
    """
    data = df.copy()
    
    # Calculate Directional Volume Asymmetry Components
    # Multi-Scale Volume Distribution Analysis
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['up_day'] = (data['price_change'] > 0).astype(int)
    data['down_day'] = (data['price_change'] < 0).astype(int)
    
    # 5-day directional volume ratio
    data['up_volume_5d'] = data['volume'].rolling(window=5).apply(
        lambda x: x[data['up_day'].iloc[-5:].values.astype(bool)].sum() if any(data['up_day'].iloc[-5:]) else 0, 
        raw=False
    )
    data['down_volume_5d'] = data['volume'].rolling(window=5).apply(
        lambda x: x[data['down_day'].iloc[-5:].values.astype(bool)].sum() if any(data['down_day'].iloc[-5:]) else 0, 
        raw=False
    )
    data['volume_ratio_5d'] = data['up_volume_5d'] / (data['down_volume_5d'] + 1e-8)
    
    # 20-day volume concentration
    data['up_volume_20d'] = data['volume'].rolling(window=20).apply(
        lambda x: x[data['up_day'].iloc[-20:].values.astype(bool)].sum() if any(data['up_day'].iloc[-20:]) else 0, 
        raw=False
    )
    data['down_volume_20d'] = data['volume'].rolling(window=20).apply(
        lambda x: x[data['down_day'].iloc[-20:].values.astype(bool)].sum() if any(data['down_day'].iloc[-20:]) else 0, 
        raw=False
    )
    data['volume_concentration_20d'] = (data['up_volume_20d'] - data['down_volume_20d']) / (data['up_volume_20d'] + data['down_volume_20d'] + 1e-8)
    
    # Dynamic volume skewness
    data['volume_skew_10d'] = data['volume'].rolling(window=10).skew()
    
    # Price-Volume Divergence Detection
    data['price_volume_corr_5d'] = data['close'].rolling(window=5).corr(data['volume'])
    data['price_volume_corr_20d'] = data['close'].rolling(window=20).corr(data['volume'])
    data['correlation_divergence'] = data['price_volume_corr_5d'] - data['price_volume_corr_20d']
    
    # Volume acceleration during consolidation
    data['price_range_5d'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close'].rolling(window=5).mean()
    data['volume_acceleration'] = data['volume'].pct_change(3) / (data['price_range_5d'] + 1e-8)
    
    # Analyze Multi-Timeframe Price Structure Asymmetry
    # Intraday Price Distribution Patterns
    data['daily_range'] = data['high'] - data['low']
    data['opening_efficiency'] = (data['open'] - data['low']) / (data['daily_range'] + 1e-8)
    data['close_location'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8) - 0.5
    
    # Final hour momentum (using close relative to day's range)
    data['final_hour_momentum'] = data['close_location'].rolling(window=3).mean()
    
    # Price Range Efficiency Analysis
    data['true_range_utilization'] = (data['close'] - data['open']) / (data['daily_range'] + 1e-8)
    
    # Range compression/expansion
    data['range_ratio'] = data['daily_range'] / data['daily_range'].rolling(window=10).mean()
    
    # Price efficiency during volume spikes
    data['volume_spike'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['efficiency_volume_spike'] = data['true_range_utilization'] * data['volume_spike']
    
    # Range asymmetry during directional moves
    data['up_move_efficiency'] = data['true_range_utilization'].where(data['price_change'] > 0, 0)
    data['down_move_efficiency'] = -data['true_range_utilization'].where(data['price_change'] < 0, 0)
    data['directional_efficiency'] = data['up_move_efficiency'] + data['down_move_efficiency']
    
    # Construct Asymmetry-Convergence Signals
    # Volume-Price Asymmetry Alignment
    data['volume_skew_efficiency_alignment'] = data['volume_skew_10d'] * data['true_range_utilization']
    
    # Volume concentration with price confirmation
    data['volume_concentration_confirmation'] = data['volume_concentration_20d'] * np.sign(data['price_change'].rolling(window=5).mean())
    
    # Multi-Timeframe Asymmetry Detection
    data['volume_distribution_divergence'] = data['volume_ratio_5d'] - data['volume_concentration_20d'].rolling(window=5).mean()
    
    # Hidden asymmetry detection
    data['hidden_asymmetry'] = (data['volume_skew_10d'] * data['correlation_divergence']).rolling(window=5).std()
    
    # Generate Dynamic Asymmetry Integration
    # Volume-Weighted Asymmetry Combination
    volume_weight = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Volume-weighted asymmetry signals
    data['weighted_volume_asymmetry'] = (data['volume_ratio_5d'] * volume_weight).rolling(window=5).mean()
    data['weighted_price_asymmetry'] = (data['directional_efficiency'] * volume_weight).rolling(window=5).mean()
    
    # Price-Structure Confirmed Pattern Recognition
    data['confirmed_asymmetry'] = (
        data['volume_concentration_confirmation'] * 
        data['true_range_utilization'] * 
        np.sign(data['price_change'])
    )
    
    # Timeframe Asymmetry Assessment
    data['intraday_asymmetry'] = (data['opening_efficiency'] + data['close_location'] + data['final_hour_momentum']) / 3
    data['daily_asymmetry'] = data['directional_efficiency'].rolling(window=5).mean()
    
    # Cross-timeframe confirmation
    data['timeframe_confirmation'] = np.sign(data['intraday_asymmetry']) * np.sign(data['daily_asymmetry'])
    
    # Factor Synthesis and Output
    # Multi-Dimensional Asymmetry Integration
    volume_asymmetry_components = (
        data['volume_ratio_5d'].fillna(0) + 
        data['volume_concentration_20d'].fillna(0) + 
        data['volume_skew_10d'].fillna(0)
    ) / 3
    
    price_structure_asymmetry = (
        data['opening_efficiency'].fillna(0) + 
        data['close_location'].fillna(0) + 
        data['true_range_utilization'].fillna(0) + 
        data['directional_efficiency'].fillna(0)
    ) / 4
    
    timeframe_alignment = (
        data['timeframe_confirmation'].fillna(0) + 
        data['volume_distribution_divergence'].fillna(0) + 
        data['hidden_asymmetry'].fillna(0)
    ) / 3
    
    # Final Alpha Generation
    asymmetry_composite = (
        0.4 * volume_asymmetry_components + 
        0.4 * price_structure_asymmetry + 
        0.2 * timeframe_alignment
    )
    
    # Apply volume-price efficiency scaling
    volume_efficiency_scale = data['efficiency_volume_spike'].rolling(window=10).mean()
    final_factor = asymmetry_composite * (1 + volume_efficiency_scale.fillna(0))
    
    # Dynamic Signal Validation - Adaptive refinement
    recent_performance = final_factor.rolling(window=10).corr(data['price_change'].shift(-1).fillna(0))
    adaptive_weight = 1 + np.tanh(recent_performance.fillna(0))
    
    # Final refined factor
    refined_factor = final_factor * adaptive_weight
    
    return refined_factor
