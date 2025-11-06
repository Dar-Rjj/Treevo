import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Helper functions
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Price-Volume Divergence Dynamics
    data['price_change'] = data['close'].pct_change()
    data['up_volume_momentum'] = np.where(data['close'] > data['close'].shift(1), 
                                         data['price_change'] * data['volume'], 0)
    data['down_volume_momentum'] = np.where(data['close'] < data['close'].shift(1), 
                                           data['price_change'] * data['volume'], 0)
    data['volume_price_divergence'] = data['up_volume_momentum'] - data['down_volume_momentum']
    
    # Gap-Volume Interaction
    data['gap_efficiency'] = ((data['open'] - data['close'].shift(1)) / data['close'].shift(1)) * data['volume']
    data['intraday_efficiency'] = ((data['close'] - data['open']) / data['open']) * data['volume']
    data['gap_intraday_alignment'] = np.sign(data['gap_efficiency']) * np.sign(data['intraday_efficiency'])
    
    # Volume Concentration Patterns
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['amount_ma_5'] = data['amount'].rolling(window=5, min_periods=1).mean()
    data['high_volume_concentration'] = ((data['volume'] > 1.5 * data['volume_ma_5'].shift(1)) & 
                                        (data['amount'] > data['amount_ma_5'].shift(1))).astype(int)
    data['low_volume_concentration'] = ((data['volume'] < 0.5 * data['volume_ma_5'].shift(1)) & 
                                       (data['amount'] < data['amount_ma_5'].shift(1))).astype(int)
    data['volume_concentration_score'] = (data['high_volume_concentration'].rolling(window=10, min_periods=1).sum() - 
                                         data['low_volume_concentration'].rolling(window=10, min_periods=1).sum())
    
    # Multi-Timeframe Volatility Regimes
    # Short-term Volatility (5-day)
    data['range_volatility'] = ((data['high'] - data['low']).rolling(window=5, min_periods=1).mean() / 
                               data['close'].rolling(window=5, min_periods=1).mean())
    data['gap_volatility'] = (np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)).rolling(window=5, min_periods=1).mean()
    data['short_term_vol_momentum'] = data['range_volatility'] - data['range_volatility'].shift(5)
    
    # Medium-term Volatility (20-day)
    data['true_range_val'] = true_range(data['high'], data['low'], data['close'].shift(1))
    data['true_range_volatility'] = (data['true_range_val'].rolling(window=20, min_periods=1).mean() / 
                                   data['close'].rolling(window=20, min_periods=1).mean())
    data['close_volatility'] = data['price_change'].rolling(window=20, min_periods=1).std()
    data['volatility_regime_shift'] = data['true_range_volatility'] / data['true_range_volatility'].shift(20) - 1
    
    # Volatility Scale Divergence
    data['volatility_scale_divergence'] = data['short_term_vol_momentum'] - data['volatility_regime_shift']
    data['volatility_persistence'] = (data['true_range_val'] > data['true_range_val'].rolling(window=10, min_periods=1).mean().shift(1)).astype(int)
    data['volatility_persistence'] = data.groupby(data.index)['volatility_persistence'].transform(
        lambda x: x * (x.groupby((x != x.shift(1)).cumsum()).cumcount() + 1))
    data['scale_alignment'] = data['volatility_persistence'] * data['volatility_regime_shift']
    
    # Trade Efficiency Dynamics
    data['overnight_efficiency'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['intraday_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['total_day_efficiency'] = (np.abs(data['open'] - data['close'].shift(1)) + np.abs(data['close'] - data['open'])) / (data['high'] - data['low'])
    
    data['volume_weighted_overnight'] = data['overnight_efficiency'] * data['volume']
    data['volume_weighted_intraday'] = data['intraday_efficiency'] * data['volume']
    data['volume_efficiency_ratio'] = data['volume_weighted_intraday'] / (data['volume_weighted_overnight'] + 0.001)
    
    data['overnight_eff_momentum'] = data['overnight_efficiency'] - data['overnight_efficiency'].shift(5)
    data['intraday_eff_momentum'] = data['intraday_efficiency'] - data['intraday_efficiency'].shift(5)
    data['efficiency_divergence'] = data['intraday_eff_momentum'] - data['overnight_eff_momentum']
    
    # Regime-Dependent Volume Patterns
    data['high_vol_volume_sensitivity'] = data['volume'] / data['true_range_val']
    data['high_vol_trade_size'] = data['amount'] / data['volume']
    data['high_vol_efficiency'] = data['total_day_efficiency'] * data['high_vol_volume_sensitivity']
    
    data['low_vol_volume_concentration'] = data['volume'] / data['volume_ma_5'].shift(1)
    data['low_vol_price_impact'] = data['price_change'] / data['volume']
    data['low_vol_efficiency'] = data['intraday_efficiency'] * data['low_vol_volume_concentration']
    
    data['volatility_breakout'] = (data['true_range_val'] > 2 * data['true_range_val'].rolling(window=10, min_periods=1).mean().shift(1)).astype(int)
    data['volume_surge'] = (data['volume'] > 2 * data['volume'].rolling(window=10, min_periods=1).mean().shift(1)).astype(int)
    data['transition_signal'] = data['volatility_breakout'] * data['volume_surge']
    
    # Multi-Scale Volume Momentum
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['amount_momentum'] = data['amount'] / data['amount'].shift(5) - 1
    data['volume_amount_divergence'] = data['volume_momentum'] - data['amount_momentum']
    
    data['volume_trend'] = data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean() - 1
    data['amount_trend'] = data['amount'] / data['amount'].rolling(window=20, min_periods=1).mean() - 1
    data['volume_amount_trend_alignment'] = np.sign(data['volume_trend']) * np.sign(data['amount_trend'])
    
    data['short_medium_volume_spread'] = data['volume_momentum'] - data['volume_trend']
    data['volume_persistence'] = (data['volume'] > data['volume_ma_5'].shift(1)).astype(int)
    data['volume_persistence'] = data.groupby(data.index)['volume_persistence'].transform(
        lambda x: x * (x.groupby((x != x.shift(1)).cumsum()).cumcount() + 1))
    data['multi_scale_volume_score'] = data['volume_persistence'] * data['volume_amount_trend_alignment']
    
    # Regime-Adaptive Factor Construction
    # High Volatility Strategy
    data['vol_volume_amplification'] = data['volume_price_divergence'] * data['true_range_volatility']
    data['high_vol_momentum'] = data['short_term_vol_momentum'] * data['volume_momentum']
    data['high_vol_composite'] = data['vol_volume_amplification'] * data['high_vol_efficiency']
    
    # Low Volatility Strategy
    data['concentration_sensitivity'] = data['volume_concentration_score'] * data['low_vol_price_impact']
    data['low_vol_persistence'] = data['volume_persistence'] * data['efficiency_divergence']
    data['low_vol_composite'] = data['concentration_sensitivity'] * data['low_vol_efficiency']
    
    # Transition Regime Strategy
    data['volume_breakout_momentum'] = data['volume_momentum'] * data['transition_signal']
    data['volatility_breakout_momentum'] = data['short_term_vol_momentum'] * data['transition_signal']
    data['breakout_composite'] = data['volume_breakout_momentum'] * data['volatility_breakout_momentum']
    data['breakout_efficiency'] = data['total_day_efficiency'] * data['transition_signal']
    data['transition_factor'] = data['breakout_composite'] * data['breakout_efficiency']
    
    # Composite Factor Synthesis
    data['short_term_component'] = data['volume_price_divergence'] * data['short_term_vol_momentum']
    data['medium_term_component'] = data['volatility_regime_shift'] * data['volume_trend']
    data['scale_alignment_component'] = data['volatility_scale_divergence'] * data['multi_scale_volume_score']
    
    # Regime Detection and Selection
    data['true_range_vol_ma_20'] = data['true_range_volatility'].rolling(window=20, min_periods=1).mean()
    data['high_vol_regime'] = (data['true_range_volatility'] > data['true_range_vol_ma_20']).astype(int)
    data['low_vol_regime'] = (data['true_range_volatility'] < 0.5 * data['true_range_vol_ma_20']).astype(int)
    
    # Regime-Selected Base Factor
    data['base_factor'] = np.where(data['transition_signal'] == 1, data['transition_factor'],
                                  np.where(data['high_vol_regime'] == 1, data['high_vol_composite'],
                                          np.where(data['low_vol_regime'] == 1, data['low_vol_composite'],
                                                  (data['short_term_component'] + data['medium_term_component'] + data['scale_alignment_component']) / 3)))
    
    # Final Factor Construction
    data['efficiency_adjusted'] = data['base_factor'] * data['total_day_efficiency']
    data['volume_confirmed'] = data['efficiency_adjusted'] * data['volume_amount_trend_alignment']
    data['final_factor'] = data['volume_confirmed'] * (1 + np.abs(data['efficiency_divergence']))
    
    return data['final_factor']
