import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate Volume-Weighted Price Acceleration
    # Short-term volume-weighted return (3-day)
    data['vw_ret_short'] = (data['close'] * data['volume']) / (data['close'].shift(3) * data['volume'].shift(3)) - 1
    
    # Medium-term volume-weighted return (8-day)
    data['vw_ret_medium'] = (data['close'] * data['volume']) / (data['close'].shift(8) * data['volume'].shift(8)) - 1
    
    # Acceleration signals
    data['accel_positive'] = ((data['vw_ret_short'] > data['vw_ret_medium']) & 
                             (data['vw_ret_short'] > 0) & 
                             (data['vw_ret_medium'] > 0) & 
                             ((data['vw_ret_short'] - data['vw_ret_medium']) > 0.1))
    
    data['accel_negative'] = ((data['vw_ret_short'] < data['vw_ret_medium']) & 
                             (data['vw_ret_short'] < 0) & 
                             (data['vw_ret_medium'] < 0) & 
                             ((data['vw_ret_medium'] - data['vw_ret_short']) > 0.1))
    
    # Calculate Price Efficiency Ratio
    # Sum of absolute price changes over 10 days
    abs_changes = []
    for i in range(10):
        abs_changes.append(abs(data['close'] - data['close'].shift(i+1)))
    
    data['sum_abs_changes'] = sum(abs_changes)
    data['net_price_change'] = abs(data['close'] - data['close'].shift(10))
    data['price_efficiency'] = data['net_price_change'] / data['sum_abs_changes']
    
    # Determine Market Regime
    data['regime_efficient'] = data['price_efficiency'] > 0.8
    data['regime_inefficient'] = data['price_efficiency'] < 0.4
    data['regime_transitional'] = (data['price_efficiency'] >= 0.4) & (data['price_efficiency'] <= 0.8)
    
    # Detect Regime-Specific Reversal Patterns
    # Efficient market reversal signals
    data['high_5d'] = data['high'].rolling(window=5, min_periods=1).max()
    data['overextension'] = ((data['close'] > data['high_5d']) & 
                            data['accel_negative'] & 
                            (data['price_efficiency'] > 0.9))
    
    # Calculate consecutive up days
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['up_day'] = data['price_change'] > 0
    data['consecutive_up'] = data['up_day'].rolling(window=8, min_periods=1).apply(
        lambda x: max(sum(1 for _ in group) if x.iloc[0] else 0 for _, group in pd.Series(x).groupby(x.ne(x.shift()).cumsum())), 
        raw=False
    )
    
    data['trend_exhaustion'] = ((data['consecutive_up'] > 7) & 
                               (data['vw_ret_short'] < data['vw_ret_medium']) & 
                               (data['price_efficiency'] > 0.8))
    
    # Inefficient market reversal signals
    data['range'] = data['high'] - data['low']
    data['range_avg'] = data['range'].rolling(window=10, min_periods=1).mean()
    data['volume_avg'] = data['volume'].rolling(window=10, min_periods=1).mean()
    
    data['breakout_signal'] = ((data['range'] > 1.5 * data['range_avg']) & 
                              (data['volume'] > 2.0 * data['volume_avg']) & 
                              (data['price_efficiency'] > 0.6))
    
    data['false_breakout'] = ((data['range'] > 1.2 * data['range_avg']) & 
                             (data['volume'] < 0.7 * data['volume_avg']) & 
                             (data['price_efficiency'] < 0.5))
    
    # Transitional market signals
    data['accel_divergence'] = abs(data['vw_ret_short'] - data['vw_ret_medium']) > 0.15
    data['efficiency_cross'] = (data['price_efficiency'] > 0.6) & (data['price_efficiency'].shift(1) <= 0.6)
    
    # Generate Multi-Timeframe Alpha Signal
    # Base score from volume-weighted acceleration
    data['base_score'] = np.where(data['accel_positive'], 0.8, 
                                 np.where(data['accel_negative'], -0.8, 0))
    
    # Regime multipliers
    data['regime_multiplier'] = np.where(data['regime_efficient'], 1.0,
                                       np.where(data['regime_inefficient'], 0.7, 0.5))
    
    # Reversal pattern adjustments
    data['reversal_adjustment'] = 0
    # Efficient market reversals (negative adjustment)
    efficient_reversal = data['overextension'] | data['trend_exhaustion']
    data.loc[efficient_reversal & (data['base_score'] > 0), 'reversal_adjustment'] = -0.3
    data.loc[efficient_reversal & (data['base_score'] < 0), 'reversal_adjustment'] = 0.3
    
    # Inefficient market breakouts (positive adjustment)
    data.loc[data['breakout_signal'] & (data['base_score'] > 0), 'reversal_adjustment'] = 0.4
    data.loc[data['breakout_signal'] & (data['base_score'] < 0), 'reversal_adjustment'] = -0.4
    
    # Transitional market signals
    transitional_positive = data['accel_divergence'] & data['efficiency_cross'] & (data['base_score'] > 0)
    transitional_negative = data['accel_divergence'] & data['efficiency_cross'] & (data['base_score'] < 0)
    
    data.loc[transitional_positive, 'reversal_adjustment'] = 0.2
    data.loc[transitional_negative, 'reversal_adjustment'] = -0.2
    
    # Calculate final signal score
    data['final_score'] = (data['base_score'] + data['reversal_adjustment']) * data['regime_multiplier']
    
    # Generate trading signal
    data['alpha_signal'] = np.where(data['final_score'] > 0.6, 1,
                                  np.where(data['final_score'] < -0.6, -1, 0))
    
    return data['alpha_signal']
