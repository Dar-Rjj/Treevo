import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Acceleration Calculation
    data['short_acc'] = (data['close'] / data['close'].shift(6) - 1) - (data['close'] / data['close'].shift(3) - 1)
    data['medium_acc'] = (data['close'] / data['close'].shift(12) - 1) - (data['close'] / data['close'].shift(6) - 1)
    data['long_acc'] = (data['close'] / data['close'].shift(20) - 1) - (data['close'] / data['close'].shift(10) - 1)
    
    # Price Efficiency Calculation
    data['price_change_abs'] = data['close'].diff().abs()
    data['short_price_eff'] = (data['close'] / data['close'].shift(3) - 1) / data['price_change_abs'].rolling(window=3, min_periods=3).sum()
    data['medium_price_eff'] = (data['close'] / data['close'].shift(6) - 1) / data['price_change_abs'].rolling(window=6, min_periods=6).sum()
    data['long_price_eff'] = (data['close'] / data['close'].shift(10) - 1) / data['price_change_abs'].rolling(window=10, min_periods=10).sum()
    
    # Range Efficiency Calculation
    data['daily_range'] = data['high'] - data['low']
    data['short_range_eff'] = (data['close'] / data['close'].shift(3) - 1) / data['daily_range'].rolling(window=3, min_periods=3).sum()
    data['medium_range_eff'] = (data['close'] / data['close'].shift(6) - 1) / data['daily_range'].rolling(window=6, min_periods=6).sum()
    data['long_range_eff'] = (data['close'] / data['close'].shift(10) - 1) / data['daily_range'].rolling(window=10, min_periods=10).sum()
    
    # Intraday Efficiency
    data['intraday_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_eff'] = data['intraday_eff'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Flow Momentum Integration
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['money_flow'] = data['typical_price'] * data['volume'] * np.sign(data['close'] - data['close'].shift(1))
    
    data['short_flow'] = data['money_flow'] - data['money_flow'].shift(3)
    data['medium_flow'] = data['money_flow'] - data['money_flow'].shift(6)
    data['long_flow'] = data['money_flow'] - data['money_flow'].shift(10)
    
    # Volume Acceleration
    data['short_vol_acc'] = (data['volume'] / data['volume'].shift(6) - 1) - (data['volume'] / data['volume'].shift(3) - 1)
    data['medium_vol_acc'] = (data['volume'] / data['volume'].shift(12) - 1) - (data['volume'] / data['volume'].shift(6) - 1)
    data['long_vol_acc'] = (data['volume'] / data['volume'].shift(20) - 1) - (data['volume'] / data['volume'].shift(10) - 1)
    
    # Efficiency-Weighted Acceleration
    data['short_eff_weighted_acc'] = data['short_price_eff'] * data['short_acc'] * data['short_flow']
    data['medium_eff_weighted_acc'] = data['medium_price_eff'] * data['medium_acc'] * data['medium_flow']
    data['long_eff_weighted_acc'] = data['long_price_eff'] * data['long_acc'] * data['long_flow']
    
    # Divergence Analysis
    data['eff_acc_div_sm'] = data['short_eff_weighted_acc'] - data['medium_eff_weighted_acc']
    data['eff_acc_div_ml'] = data['medium_eff_weighted_acc'] - data['long_eff_weighted_acc']
    data['cross_timeframe_eff_acc'] = data['short_eff_weighted_acc'] + data['medium_eff_weighted_acc'] + data['long_eff_weighted_acc']
    
    data['vol_acc_div_sm'] = data['medium_vol_acc'] - data['short_vol_acc']
    data['vol_acc_div_ml'] = data['long_vol_acc'] - data['medium_vol_acc']
    
    data['range_eff_div_sm'] = data['short_range_eff'] - data['medium_range_eff']
    data['range_eff_div_ml'] = data['medium_range_eff'] - data['long_range_eff']
    
    # Intraday Strength & Liquidity Assessment
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['close_position'] = data['close_position'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['range_utilization'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_utilization'] = data['range_utilization'].replace([np.inf, -np.inf], 0).fillna(0)
    
    data['opening_gap'] = data['open'] / data['close'].shift(1) - 1
    
    data['strength_score'] = data['close_position'] * data['range_utilization'] * data['opening_gap'].abs()
    
    data['amount_ratio'] = data['amount'] / (data['amount'].rolling(window=3, min_periods=3).mean())
    
    # Volume persistence calculation
    vol_ma_3 = data['volume'].rolling(window=3, min_periods=3).mean()
    data['volume_persistence'] = 0
    for i in range(3, len(data)):
        count = 0
        for j in range(i, max(i-5, -1), -1):
            if data['volume'].iloc[j] > vol_ma_3.iloc[j]:
                count += 1
            else:
                break
        data.loc[data.index[i], 'volume_persistence'] = count
    
    data['volume_surge'] = data['volume'] > (1.3 * data['volume'].rolling(window=3, min_periods=3).mean())
    data['liquidity_score'] = data['amount_ratio'] * data['volume_persistence']
    
    # Volatility Context
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        )
    )
    data['volatility_regime'] = data['true_range'] > data['true_range'].rolling(window=5, min_periods=5).mean()
    data['volatility_weight'] = data['true_range'] / data['true_range'].rolling(window=20, min_periods=20).mean()
    
    # Adaptive Alpha Generation
    # High Efficiency Regime Strategy
    high_eff_condition = (data['short_price_eff'] > 0.7) & (data['medium_price_eff'] > 0.5)
    high_eff_signal = data['eff_acc_div_sm']
    high_eff_confirmation = (data['vol_acc_div_sm'] > 0) & (data['strength_score'] > 0.25)
    high_eff_factor = high_eff_signal * high_eff_confirmation.astype(int) * data['volatility_weight']
    
    # Flow Momentum Strategy
    flow_momentum_condition = (data['short_flow'] > data['medium_flow']) & (data['medium_flow'] > data['long_flow'])
    flow_momentum_signal = data['cross_timeframe_eff_acc']
    flow_momentum_confirmation = (data['range_utilization'] > 0.6) & data['volume_surge']
    flow_momentum_factor = flow_momentum_signal * flow_momentum_confirmation.astype(int) * data['liquidity_score']
    
    # Volatility-Adaptive Strategy
    volatility_adaptive_signal = (data['eff_acc_div_sm'] + data['eff_acc_div_ml']) * data['strength_score']
    volatility_adaptive_filter = data['vol_acc_div_sm'] > 0
    volatility_adaptive_factor = volatility_adaptive_signal * volatility_adaptive_filter.astype(int) * data['volatility_weight']
    
    # Range Efficiency Strategy
    range_eff_condition = (data['short_range_eff'] > data['medium_range_eff']) & (data['medium_range_eff'] > data['long_range_eff'])
    range_eff_signal = (data['range_eff_div_sm'] + data['range_eff_div_ml']) * data['intraday_eff']
    range_eff_confirmation = (data['volume_persistence'] > 2) & (data['amount_ratio'] > 1.1)
    range_eff_factor = range_eff_signal * range_eff_confirmation.astype(int) * data['strength_score']
    
    # Combine strategies with conditions
    final_factor = pd.Series(index=data.index, dtype=float)
    
    # Apply each strategy only when its condition is met
    final_factor[high_eff_condition] = high_eff_factor[high_eff_condition]
    final_factor[flow_momentum_condition] = flow_momentum_factor[flow_momentum_condition]
    final_factor[data['volatility_regime']] = volatility_adaptive_factor[data['volatility_regime']]
    final_factor[range_eff_condition] = range_eff_factor[range_eff_condition]
    
    # Fill remaining values with weighted average
    mask = final_factor.isna()
    weights = [0.4, 0.3, 0.2, 0.1]  # Weighting for each strategy
    weighted_avg = (
        weights[0] * high_eff_factor + 
        weights[1] * flow_momentum_factor + 
        weights[2] * volatility_adaptive_factor + 
        weights[3] * range_eff_factor
    )
    final_factor[mask] = weighted_avg[mask]
    
    return final_factor
