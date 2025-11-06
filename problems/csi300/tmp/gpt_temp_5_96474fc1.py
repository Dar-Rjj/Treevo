import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    epsilon = 1e-8
    
    # Multi-Timeframe Flow Asymmetry Framework
    df['morning_flow_pressure'] = (df['high'] - df['open']) * df['volume']
    df['afternoon_flow_pressure'] = (df['open'] - df['low']) * df['volume']
    df['net_flow_asymmetry'] = df['morning_flow_pressure'] - df['afternoon_flow_pressure']
    
    # Multi-Period Flow Momentum
    df['short_term_flow_momentum'] = df['net_flow_asymmetry'] / (df['net_flow_asymmetry'].shift(1).abs() + epsilon)
    
    df['net_flow_sum_3'] = df['net_flow_asymmetry'].rolling(window=3, min_periods=1).sum()
    df['net_flow_abs_sum_3'] = df['net_flow_asymmetry'].abs().rolling(window=3, min_periods=1).sum()
    df['medium_term_flow_momentum'] = df['net_flow_sum_3'] / (df['net_flow_abs_sum_3'] + epsilon)
    
    df['net_flow_sum_6'] = df['net_flow_asymmetry'].rolling(window=6, min_periods=1).sum()
    df['net_flow_abs_sum_6'] = df['net_flow_asymmetry'].abs().rolling(window=6, min_periods=1).sum()
    df['long_term_flow_momentum'] = df['net_flow_sum_6'] / (df['net_flow_abs_sum_6'] + epsilon)
    
    # Flow Asymmetry Regime
    df['flow_momentum_divergence'] = (df['short_term_flow_momentum'] - df['medium_term_flow_momentum']) * (df['medium_term_flow_momentum'] - df['long_term_flow_momentum'])
    df['flow_acceleration'] = df['short_term_flow_momentum'] - df['long_term_flow_momentum']
    df['flow_asymmetry_score'] = df['flow_momentum_divergence'] * df['flow_acceleration']
    
    # Nonlinear Price Impact Dynamics
    df['relative_trade_size'] = df['amount'] / (df['amount'].shift(1) + epsilon)
    df['size_weighted_return'] = (df['close'] - df['open']) * df['relative_trade_size']
    df['nonlinear_size_impact'] = (df['size_weighted_return'] ** 3) / (df['size_weighted_return'].abs() ** 2 + epsilon)
    
    df['volume_return_efficiency'] = ((df['close'] - df['open']) ** 2) / (df['volume'] + 1)
    df['volume_pressure_divergence'] = df['volume_return_efficiency'] / (df['volume_return_efficiency'].shift(1) + epsilon)
    df['nonlinear_volume_impact'] = df['volume_pressure_divergence'] * ((df['close'] - df['close'].shift(2)) ** 2)
    
    df['size_volume_interaction'] = df['nonlinear_size_impact'] * df['nonlinear_volume_impact']
    df['impact_momentum'] = df['size_volume_interaction'] / (df['size_volume_interaction'].shift(1).abs() + epsilon)
    df['price_impact_core'] = df['impact_momentum'] * df['flow_asymmetry_score']
    
    # Gap-Driven Momentum System
    df['overnight_gap'] = df['open'] - df['close'].shift(1)
    df['intraday_gap'] = df['close'] - df['open']
    df['gap_ratio'] = df['overnight_gap'] / (df['intraday_gap'] + epsilon)
    
    df['gap_momentum'] = df['gap_ratio'] * (df['volume'] - df['volume'].shift(1))
    df['gap_acceleration'] = (df['gap_ratio'] - df['gap_ratio'].shift(1)) * (df['close'] - df['close'].shift(2))
    
    def count_gap_persistence(series):
        current_sign = np.sign(series.iloc[-1]) if len(series) > 0 else 0
        count = 0
        for i in range(1, min(3, len(series))):
            if np.sign(series.iloc[-i-1]) == current_sign:
                count += 1
        return count
    
    df['gap_persistence'] = df['gap_ratio'].rolling(window=3, min_periods=1).apply(count_gap_persistence, raw=False)
    
    df['enhanced_gap_signal'] = df['gap_momentum'] * df['gap_acceleration']
    df['gap_quality_score'] = df['enhanced_gap_signal'] * df['gap_persistence']
    df['gap_momentum_core'] = df['gap_quality_score'] * df['price_impact_core']
    
    # Volatility-Flow Convergence Framework
    df['micro_range'] = df['high'] - df['low']
    df['short_term_range'] = df['high'].rolling(window=3, min_periods=1).max() - df['low'].rolling(window=3, min_periods=1).min()
    df['medium_term_range'] = df['high'].rolling(window=6, min_periods=1).max() - df['low'].rolling(window=6, min_periods=1).min()
    df['volatility_cascade'] = (df['micro_range'] * df['short_term_range']) / (df['medium_term_range'] + epsilon)
    
    df['flow_volatility_efficiency'] = df['net_flow_asymmetry'] / (df['micro_range'] + epsilon)
    df['volatility_flow_momentum'] = df['flow_volatility_efficiency'] / (df['flow_volatility_efficiency'].shift(1) + epsilon)
    df['nonlinear_volatility_flow'] = df['volatility_flow_momentum'] * df['volatility_cascade']
    
    df['multi_timeframe_convergence'] = df['nonlinear_volatility_flow'] * df['gap_momentum_core']
    df['convergence_momentum'] = df['multi_timeframe_convergence'] / (df['multi_timeframe_convergence'].shift(1).abs() + epsilon)
    df['volatility_flow_core'] = df['convergence_momentum'] * df['flow_asymmetry_score']
    
    # Persistence-Enhanced Momentum
    def count_persistence(series):
        current_sign = np.sign(series.iloc[-1]) if len(series) > 0 else 0
        count = 0
        for i in range(1, min(3, len(series))):
            if np.sign(series.iloc[-i-1]) == current_sign:
                count += 1
        return count
    
    df['flow_asymmetry_persistence'] = df['net_flow_asymmetry'].rolling(window=3, min_periods=1).apply(count_persistence, raw=False)
    df['price_impact_persistence'] = df['price_impact_core'].rolling(window=3, min_periods=1).apply(count_persistence, raw=False)
    df['gap_momentum_persistence'] = df['gap_momentum_core'].rolling(window=3, min_periods=1).apply(count_persistence, raw=False)
    df['volatility_flow_persistence'] = df['volatility_flow_core'].rolling(window=3, min_periods=1).apply(count_persistence, raw=False)
    
    df['core_persistence_score'] = df['flow_asymmetry_persistence'] * df['price_impact_persistence']
    df['momentum_persistence_score'] = df['gap_momentum_persistence'] * df['volatility_flow_persistence']
    df['overall_persistence_quality'] = df['core_persistence_score'] * df['momentum_persistence_score']
    
    # Final Alpha Construction
    df['flow_impact_base'] = df['price_impact_core'] * df['flow_asymmetry_score']
    df['gap_volatility_base'] = df['gap_momentum_core'] * df['volatility_flow_core']
    df['core_alpha_base'] = df['flow_impact_base'] * df['gap_volatility_base']
    
    df['persistence_weighted_base'] = df['core_alpha_base'] * df['overall_persistence_quality']
    df['persistence_momentum'] = df['persistence_weighted_base'] / (df['persistence_weighted_base'].shift(1).abs() + epsilon)
    df['enhanced_alpha'] = df['persistence_momentum'] * df['flow_asymmetry_score']
    
    return df['enhanced_alpha']
