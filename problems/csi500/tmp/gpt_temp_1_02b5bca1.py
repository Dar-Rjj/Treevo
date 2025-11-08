import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Decay-Adjusted Intraday Momentum
    data['price_momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['combined_momentum'] = data['price_momentum'] * data['intraday_efficiency']
    
    # Find days since peak momentum
    data['momentum_peak'] = data['combined_momentum'].expanding().max()
    data['days_since_peak'] = 0
    current_peak = data['combined_momentum'].iloc[0]
    peak_day = 0
    
    for i in range(1, len(data)):
        if data['combined_momentum'].iloc[i] > current_peak:
            current_peak = data['combined_momentum'].iloc[i]
            peak_day = i
        data.iloc[i, data.columns.get_loc('days_since_peak')] = i - peak_day
    
    data['decay_factor'] = np.exp(-0.1386 * data['days_since_peak'])
    data['decayed_momentum'] = data['combined_momentum'] * data['decay_factor']
    
    # Volume persistence integration
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['volume_direction'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['decay_momentum_factor'] = data['decayed_momentum'] * (1 + data['volume_direction'] * np.log(np.abs(data['volume_ratio'])))
    
    # Regime-Adaptive Volatility Factor
    data['returns'] = data['close'] / data['close'].shift(1) - 1
    data['short_vol'] = data['returns'].rolling(window=5).std()
    data['long_vol'] = data['returns'].rolling(window=20).std()
    data['regime_ratio'] = data['short_vol'] / data['long_vol']
    
    data['high_vol_signal'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['low_vol_signal'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * (data['volume'] / data['volume'].shift(1) - 1)
    data['transition_weight'] = 1 / (1 + np.exp(-10 * (data['regime_ratio'] - 1)))
    
    data['high_vol_contrib'] = data['high_vol_signal'] * data['transition_weight']
    data['low_vol_contrib'] = data['low_vol_signal'] * (1 - data['transition_weight'])
    data['regime_factor'] = data['high_vol_contrib'] + data['low_vol_contrib']
    
    # Multi-Timeframe Convergence Alignment
    data['price_momentum_st'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['volume_trend_st'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3)
    data['st_convergence'] = data['price_momentum_st'] * np.sign(data['volume_trend_st'] - 1) * np.log(np.abs(data['volume_trend_st']))
    
    data['price_momentum_mt'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['volume_trend_mt'] = data['volume'] / data['volume'].rolling(window=10).mean()
    data['mt_convergence'] = data['price_momentum_mt'] * np.sign(data['volume_trend_mt'] - 1) * np.log(np.abs(data['volume_trend_mt']))
    
    data['alignment_score'] = np.sign(data['st_convergence']) * np.sign(data['mt_convergence'])
    data['strength_score'] = np.abs(data['st_convergence']) + np.abs(data['mt_convergence'])
    data['convergence_factor'] = data['alignment_score'] * data['strength_score'] * (1 + np.abs(data['st_convergence'] - data['mt_convergence']))
    
    # Price-Volume Efficiency with Persistence
    data['price_change_efficiency'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['volume_efficiency'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2)) / 2)
    data['combined_efficiency'] = data['price_change_efficiency'] * np.sign(data['volume_efficiency'] - 1)
    
    # Calculate efficiency streak
    data['efficiency_sign'] = np.sign(data['combined_efficiency'])
    data['efficiency_streak'] = 1
    for i in range(1, len(data)):
        if data['efficiency_sign'].iloc[i] == data['efficiency_sign'].iloc[i-1]:
            data.iloc[i, data.columns.get_loc('efficiency_streak')] = data['efficiency_streak'].iloc[i-1] + 1
        else:
            data.iloc[i, data.columns.get_loc('efficiency_streak')] = 1
    
    data['streak_weight'] = np.log(1 + data['efficiency_streak'])
    data['weighted_efficiency'] = data['combined_efficiency'] * data['streak_weight']
    
    data['direction'] = np.sign(data['close'] - data['close'].shift(1))
    data['magnitude'] = data['weighted_efficiency'] * np.abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['efficiency_factor'] = data['direction'] * data['magnitude']
    
    # Amount Flow Persistence with Decay
    data['up_day_amount'] = np.where(data['close'] > data['open'], data['amount'], 0)
    data['down_day_amount'] = np.where(data['close'] < data['open'], data['amount'], 0)
    
    # Calculate net flow using rolling windows
    data['net_flow'] = (data['up_day_amount'].rolling(window=5).sum() - data['down_day_amount'].rolling(window=5).sum()) / \
                       (data['up_day_amount'].rolling(window=5).sum() + data['down_day_amount'].rolling(window=5).sum())
    
    # Calculate flow direction streak
    data['flow_sign'] = np.sign(data['net_flow'])
    data['flow_streak'] = 1
    for i in range(1, len(data)):
        if data['flow_sign'].iloc[i] == data['flow_sign'].iloc[i-1]:
            data.iloc[i, data.columns.get_loc('flow_streak')] = data['flow_streak'].iloc[i-1] + 1
        else:
            data.iloc[i, data.columns.get_loc('flow_streak')] = 1
    
    # Calculate cumulative flow over current streak
    data['cumulative_flow'] = 0.0
    for i in range(len(data)):
        streak_length = data['flow_streak'].iloc[i]
        if streak_length > 0:
            start_idx = max(0, i - streak_length + 1)
            data.iloc[i, data.columns.get_loc('cumulative_flow')] = data['net_flow'].iloc[start_idx:i+1].sum()
    
    data['persistence_score'] = data['cumulative_flow'] * np.exp(-0.1386 * data['flow_streak'])
    data['momentum_component'] = data['persistence_score'] * data['net_flow']
    data['reversal_indicator'] = 1 / (1 + np.exp(data['flow_streak'] - 5))
    data['amount_flow_factor'] = data['momentum_component'] * data['reversal_indicator']
    
    # Combine all factors with equal weights
    factors = ['decay_momentum_factor', 'regime_factor', 'convergence_factor', 'efficiency_factor', 'amount_flow_factor']
    
    for factor in factors:
        data[factor] = data[factor].fillna(0)
    
    # Z-score normalization for each factor
    for factor in factors:
        mean_val = data[factor].mean()
        std_val = data[factor].std()
        if std_val > 0:
            data[f'{factor}_norm'] = (data[factor] - mean_val) / std_val
        else:
            data[f'{factor}_norm'] = 0
    
    # Combine normalized factors
    result = (data['decay_momentum_factor_norm'] + data['regime_factor_norm'] + 
              data['convergence_factor_norm'] + data['efficiency_factor_norm'] + 
              data['amount_flow_factor_norm']) / 5
    
    return result
