import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Asymmetry Framework
    # Directional Volatility Components
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['upside_move'] = np.maximum(0, data['close_ret'])
    data['downside_move'] = np.maximum(0, -data['close_ret'])
    
    data['upside_vol'] = data['upside_move'].rolling(window=5).std()
    data['downside_vol'] = data['downside_move'].rolling(window=5).std()
    data['volatility_asymmetry_ratio'] = data['upside_vol'] / data['downside_vol']
    
    # Intraday Volatility Structure
    data['open_gap'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_move'] = data['close'] / data['open'] - 1
    
    data['opening_vol'] = data['open_gap'].rolling(window=5).std()
    data['closing_vol'] = data['intraday_move'].rolling(window=5).std()
    data['intraday_volatility_bias'] = data['opening_vol'] / data['closing_vol']
    
    # Multi-Timeframe Volatility Alignment
    data['short_term_vol'] = data['close'].pct_change().rolling(window=5).std()
    data['medium_term_vol'] = data['close'].pct_change().shift(5).rolling(window=5).std()
    data['volatility_regime_shift'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Bidirectional Flow Analysis
    # Price-Volume Flow Dynamics
    data['up_volume'] = np.where(data['close'] > data['close'].shift(1), data['volume'], 0)
    data['down_volume'] = np.where(data['close'] < data['close'].shift(1), data['volume'], 0)
    
    data['upward_flow_intensity'] = data['up_volume'].rolling(window=5).sum()
    data['downward_flow_intensity'] = data['down_volume'].rolling(window=5).sum()
    data['net_flow_direction'] = (data['upward_flow_intensity'] - data['downward_flow_intensity']) / \
                                (data['upward_flow_intensity'] + data['downward_flow_intensity'] + 1e-8)
    
    # Amount-Based Flow Patterns
    data['amount_ma_5'] = data['amount'].rolling(window=5).mean()
    data['large_trade'] = np.where(data['amount'] > data['amount_ma_5'], data['amount'], 0)
    data['small_trade'] = np.where(data['amount'] < data['amount_ma_5'], data['amount'], 0)
    
    data['large_trade_flow'] = data['large_trade'].rolling(window=5).sum()
    data['small_trade_flow'] = data['small_trade'].rolling(window=5).sum()
    data['institutional_flow_bias'] = data['large_trade_flow'] / (data['small_trade_flow'] + 1e-8)
    
    # Flow-Volatility Coupling
    data['abs_return'] = data['close'].pct_change().abs()
    
    def rolling_corr(x, y, window):
        return x.rolling(window=window).corr(y)
    
    data['flow_induced_volatility'] = rolling_corr(data['volume'], data['abs_return'], 5)
    data['daily_range'] = data['high'] - data['low']
    data['amount_volatility_sensitivity'] = rolling_corr(data['amount'], data['daily_range'], 5)
    data['flow_volatility_alignment'] = data['flow_induced_volatility'] * data['amount_volatility_sensitivity']
    
    # Asymmetric Regime Classification
    data['high_asymmetry_regime'] = (data['volatility_asymmetry_ratio'] > 1.5) | (data['volatility_asymmetry_ratio'] < 0.67)
    data['balanced_volatility_regime'] = (data['volatility_asymmetry_ratio'] > 0.8) & (data['volatility_asymmetry_ratio'] < 1.25)
    data['transitional_volatility_regime'] = (data['volatility_regime_shift'] > 1.3) | (data['volatility_regime_shift'] < 0.77)
    
    data['strong_upward_flow'] = (data['net_flow_direction'] > 0.3) & (data['institutional_flow_bias'] > 1.2)
    data['strong_downward_flow'] = (data['net_flow_direction'] < -0.3) & (data['institutional_flow_bias'] > 1.2)
    data['mixed_flow_regime'] = (data['net_flow_direction'].abs() < 0.2) & (data['institutional_flow_bias'] < 0.8)
    
    data['high_asymmetry_strong_flow'] = data['high_asymmetry_regime'] & (data['strong_upward_flow'] | data['strong_downward_flow'])
    data['balanced_mixed_flow'] = data['balanced_volatility_regime'] & data['mixed_flow_regime']
    data['transitional_institutional_flow'] = data['transitional_volatility_regime'] & (data['institutional_flow_bias'] > 1.5)
    
    # Regime-Specific Alpha Signals
    # High Asymmetry Momentum
    data['asymmetric_trend_following'] = (data['close'] / data['close'].shift(3) - 1) * data['volatility_asymmetry_ratio']
    data['flow_enhanced_asymmetry'] = data['asymmetric_trend_following'] * data['net_flow_direction']
    data['high_asymmetry_momentum'] = data['flow_enhanced_asymmetry'] * data['short_term_vol']
    
    # Balanced Regime Mean Reversion
    data['close_ma_5'] = data['close'].rolling(window=5).mean()
    data['volatility_normalized_reversion'] = (data['close'] - data['close_ma_5']) / (data['short_term_vol'] + 1e-8)
    data['flow_confirmed_reversion'] = data['volatility_normalized_reversion'] * (1 - data['net_flow_direction'].abs())
    data['balanced_regime_alpha'] = data['flow_confirmed_reversion'] * data['intraday_volatility_bias']
    
    # Transitional Regime Breakout
    data['volatility_breakout_signal'] = data['daily_range'] / (data['medium_term_vol'] + 1e-8)
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['flow_acceleration'] = (data['volume'] / data['volume_ma_5'] - 1) * data['institutional_flow_bias']
    data['transitional_alpha'] = data['volatility_breakout_signal'] * data['flow_acceleration']
    
    # Hierarchical Signal Integration
    # Primary Regime Selection
    regime_signals = pd.DataFrame({
        'high_asymmetry': data['high_asymmetry_momentum'],
        'balanced': data['balanced_regime_alpha'],
        'transitional': data['transitional_alpha']
    })
    
    data['dominant_regime_score'] = regime_signals.max(axis=1)
    second_highest = regime_signals.apply(lambda x: x.nlargest(2).iloc[-1], axis=1)
    data['regime_confidence'] = 1 - (second_highest / (data['dominant_regime_score'] + 1e-8))
    data['selected_regime_signal'] = data['dominant_regime_score'] * data['regime_confidence']
    
    # Cross-Regime Validation
    data['volatility_flow_consistency'] = rolling_corr(data['volume'], data['daily_range'], 5)
    data['amount_volatility_alignment'] = rolling_corr(data['amount'], data['abs_return'], 5)
    data['multi_dimensional_validation'] = data['volatility_flow_consistency'] * data['amount_volatility_alignment']
    
    # Final Alpha Construction
    data['base_alpha'] = data['selected_regime_signal'] * data['multi_dimensional_validation']
    data['risk_adjusted_alpha'] = data['base_alpha'] / (data['short_term_vol'] + 1e-8)
    data['final_flow_enhanced_alpha'] = data['risk_adjusted_alpha'] * (1 + data['flow_volatility_alignment'])
    
    # Dynamic Regime Transition Monitoring
    # Volatility Regime Stability
    data['asymmetry_persistence'] = rolling_corr(data['volatility_asymmetry_ratio'], 
                                               data['volatility_asymmetry_ratio'].shift(5), 5)
    data['volatility_shift_momentum'] = data['volatility_regime_shift'] / data['volatility_regime_shift'].shift(1)
    data['volatility_regime_confidence'] = data['asymmetry_persistence'].abs() * data['volatility_shift_momentum']
    
    # Flow Regime Persistence
    data['net_flow_consistency'] = rolling_corr(data['net_flow_direction'], 
                                              data['net_flow_direction'].shift(5), 5)
    data['institutional_flow_stability'] = rolling_corr(data['institutional_flow_bias'], 
                                                      data['institutional_flow_bias'].shift(5), 5)
    data['flow_regime_confidence'] = data['net_flow_consistency'] * data['institutional_flow_stability']
    
    # Adaptive Alpha Output
    data['overall_regime_stability'] = data['volatility_regime_confidence'] * data['flow_regime_confidence']
    data['stability_weighted_alpha'] = data['final_flow_enhanced_alpha'] * data['overall_regime_stability']
    data['dynamic_alpha'] = data['stability_weighted_alpha'] * (1 + data['volatility_regime_shift'])
    
    return data['dynamic_alpha']
