import pandas as pd
import numpy as np
def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required periods
    periods = [3, 5, 6, 10, 12, 19, 20]
    max_period = max(periods)
    
    # Calculate basic price differences and ranges
    data['close_diff'] = data['close'].diff()
    data['abs_close_diff'] = np.abs(data['close_diff'])
    data['daily_range'] = data['high'] - data['low']
    
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = np.abs(data['high'] - data['close'].shift(1))
    data['tr3'] = np.abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Typical Price and Money Flow
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['money_flow'] = data['typical_price'] * data['volume'] * np.sign(data['close_diff'])
    
    # Initialize all intermediate columns
    for col in ['short_price_eff', 'short_range_eff', 'short_price_acc', 'short_vol_acc',
                'medium_price_eff', 'medium_range_eff', 'medium_price_acc', 'medium_vol_acc',
                'long_price_eff', 'long_range_eff', 'long_price_acc', 'long_vol_acc',
                'short_flow', 'medium_flow', 'long_flow',
                'short_eff_weighted_acc', 'medium_eff_weighted_acc', 'long_eff_weighted_acc',
                'range_efficiency', 'close_position', 'gap_consistency',
                'amount_ratio', 'volume_persistence', 'volume_surge',
                'strength_score', 'liquidity_score',
                'trend_direction', 'volatility_regime', 'volatility_weight',
                'short_momentum', 'medium_momentum']:
        data[col] = np.nan
    
    # Calculate factors for each day
    for i in range(max_period, len(data)):
        # Multi-Timeframe Efficiency & Acceleration Framework
        # Short-term (3-6 days)
        if i >= 6:
            # Price Efficiency
            price_changes_sum = data['abs_close_diff'].iloc[i-2:i+1].sum()
            if price_changes_sum != 0:
                data.loc[data.index[i], 'short_price_eff'] = (data['close'].iloc[i] - data['close'].iloc[i-3]) / price_changes_sum
            
            # Range Efficiency
            range_sum = data['daily_range'].iloc[i-2:i+1].sum()
            if range_sum != 0:
                data.loc[data.index[i], 'short_range_eff'] = (data['close'].iloc[i] - data['close'].iloc[i-3]) / range_sum
            
            # Price Acceleration
            ret_6d = data['close'].iloc[i] / data['close'].iloc[i-6] - 1
            ret_3d = data['close'].iloc[i] / data['close'].iloc[i-3] - 1
            data.loc[data.index[i], 'short_price_acc'] = ret_6d - ret_3d
            
            # Volume Acceleration
            vol_ret_6d = data['volume'].iloc[i] / data['volume'].iloc[i-6] - 1
            vol_ret_3d = data['volume'].iloc[i] / data['volume'].iloc[i-3] - 1
            data.loc[data.index[i], 'short_vol_acc'] = vol_ret_6d - vol_ret_3d
        
        # Medium-term (6-12 days)
        if i >= 12:
            # Price Efficiency
            price_changes_sum = data['abs_close_diff'].iloc[i-5:i+1].sum()
            if price_changes_sum != 0:
                data.loc[data.index[i], 'medium_price_eff'] = (data['close'].iloc[i] - data['close'].iloc[i-6]) / price_changes_sum
            
            # Range Efficiency
            range_sum = data['daily_range'].iloc[i-5:i+1].sum()
            if range_sum != 0:
                data.loc[data.index[i], 'medium_range_eff'] = (data['close'].iloc[i] - data['close'].iloc[i-6]) / range_sum
            
            # Price Acceleration
            ret_12d = data['close'].iloc[i] / data['close'].iloc[i-12] - 1
            ret_6d = data['close'].iloc[i] / data['close'].iloc[i-6] - 1
            data.loc[data.index[i], 'medium_price_acc'] = ret_12d - ret_6d
            
            # Volume Acceleration
            vol_ret_12d = data['volume'].iloc[i] / data['volume'].iloc[i-12] - 1
            vol_ret_6d = data['volume'].iloc[i] / data['volume'].iloc[i-6] - 1
            data.loc[data.index[i], 'medium_vol_acc'] = vol_ret_12d - vol_ret_6d
        
        # Long-term (10-20 days)
        if i >= 20:
            # Price Efficiency
            price_changes_sum = data['abs_close_diff'].iloc[i-9:i+1].sum()
            if price_changes_sum != 0:
                data.loc[data.index[i], 'long_price_eff'] = (data['close'].iloc[i] - data['close'].iloc[i-10]) / price_changes_sum
            
            # Range Efficiency
            range_sum = data['daily_range'].iloc[i-9:i+1].sum()
            if range_sum != 0:
                data.loc[data.index[i], 'long_range_eff'] = (data['close'].iloc[i] - data['close'].iloc[i-10]) / range_sum
            
            # Price Acceleration
            ret_20d = data['close'].iloc[i] / data['close'].iloc[i-20] - 1
            ret_10d = data['close'].iloc[i] / data['close'].iloc[i-10] - 1
            data.loc[data.index[i], 'long_price_acc'] = ret_20d - ret_10d
            
            # Volume Acceleration
            vol_ret_20d = data['volume'].iloc[i] / data['volume'].iloc[i-20] - 1
            vol_ret_10d = data['volume'].iloc[i] / data['volume'].iloc[i-10] - 1
            data.loc[data.index[i], 'long_vol_acc'] = vol_ret_20d - vol_ret_10d
        
        # Flow Momentum & Divergence Analysis
        if i >= 10:
            # Multi-Timeframe Flow Momentum
            data.loc[data.index[i], 'short_flow'] = data['money_flow'].iloc[i] - data['money_flow'].iloc[i-3]
            data.loc[data.index[i], 'medium_flow'] = data['money_flow'].iloc[i] - data['money_flow'].iloc[i-6]
            data.loc[data.index[i], 'long_flow'] = data['money_flow'].iloc[i] - data['money_flow'].iloc[i-10]
            
            # Efficiency-Weighted Acceleration
            if pd.notna(data['short_price_eff'].iloc[i]) and pd.notna(data['short_price_acc'].iloc[i]):
                data.loc[data.index[i], 'short_eff_weighted_acc'] = data['short_price_eff'].iloc[i] * data['short_price_acc'].iloc[i]
            
            if pd.notna(data['medium_price_eff'].iloc[i]) and pd.notna(data['medium_price_acc'].iloc[i]):
                data.loc[data.index[i], 'medium_eff_weighted_acc'] = data['medium_price_eff'].iloc[i] * data['medium_price_acc'].iloc[i]
            
            if pd.notna(data['long_price_eff'].iloc[i]) and pd.notna(data['long_price_acc'].iloc[i]):
                data.loc[data.index[i], 'long_eff_weighted_acc'] = data['long_price_eff'].iloc[i] * data['long_price_acc'].iloc[i]
        
        # Intraday Behavior & Quality Assessment
        # Price Behavior Quality
        daily_range = data['high'].iloc[i] - data['low'].iloc[i]
        if daily_range != 0:
            data.loc[data.index[i], 'range_efficiency'] = (data['close'].iloc[i] - data['low'].iloc[i]) / daily_range
            data.loc[data.index[i], 'close_position'] = (data['close'].iloc[i] - data['low'].iloc[i]) / daily_range
        
        if i >= 1:
            gap_today = data['close'].iloc[i] / data['open'].iloc[i]
            gap_yesterday = data['close'].iloc[i-1] / data['open'].iloc[i-1]
            data.loc[data.index[i], 'gap_consistency'] = gap_today - gap_yesterday
        
        # Liquidity Assessment
        if i >= 2:
            avg_amount = (data['amount'].iloc[i-2:i+1].sum() / 3)
            if avg_amount != 0:
                data.loc[data.index[i], 'amount_ratio'] = data['amount'].iloc[i] / avg_amount
            
            # Volume persistence (simplified)
            vol_avg = (data['volume'].iloc[i-3:i].sum() / 3)
            data.loc[data.index[i], 'volume_persistence'] = 1 if data['volume'].iloc[i] > vol_avg else 0
            data.loc[data.index[i], 'volume_surge'] = 1 if data['volume'].iloc[i] > 1.3 * vol_avg else 0
        
        # Combined Quality Score
        if pd.notna(data['close_position'].iloc[i]) and pd.notna(data['range_efficiency'].iloc[i]) and pd.notna(data['gap_consistency'].iloc[i]):
            data.loc[data.index[i], 'strength_score'] = data['close_position'].iloc[i] * data['range_efficiency'].iloc[i] * np.abs(data['gap_consistency'].iloc[i])
        
        if pd.notna(data['amount_ratio'].iloc[i]) and pd.notna(data['volume_persistence'].iloc[i]):
            data.loc[data.index[i], 'liquidity_score'] = data['amount_ratio'].iloc[i] * data['volume_persistence'].iloc[i]
        
        # Market Regime & Volatility Context
        if i >= 20:
            data.loc[data.index[i], 'trend_direction'] = np.sign(data['close'].iloc[i] - data['close'].iloc[i-20])
            
            # Volatility State
            avg_tr_5d = data['true_range'].iloc[i-4:i+1].mean()
            data.loc[data.index[i], 'volatility_regime'] = np.sign(data['true_range'].iloc[i] - avg_tr_5d)
            
            # Volatility Weight
            avg_tr_20d = data['true_range'].iloc[i-19:i+1].mean()
            if avg_tr_20d != 0:
                data.loc[data.index[i], 'volatility_weight'] = data['true_range'].iloc[i] / avg_tr_20d
            
            # Multi-Timeframe Alignment
            data.loc[data.index[i], 'short_momentum'] = data['close'].iloc[i] / data['close'].iloc[i-5] - 1
            data.loc[data.index[i], 'medium_momentum'] = data['close'].iloc[i] / data['close'].iloc[i-10] - 1
        
        # Regime-Adaptive Alpha Construction
        alpha_components = []
        
        # High Efficiency Trending Alpha
        if (pd.notna(data['short_price_eff'].iloc[i]) and pd.notna(data['medium_price_eff'].iloc[i]) and
            data['short_price_eff'].iloc[i] > 0.7 and data['medium_price_eff'].iloc[i] > 0.5 and
            pd.notna(data['short_eff_weighted_acc'].iloc[i]) and pd.notna(data['medium_eff_weighted_acc'].iloc[i]) and
            pd.notna(data['short_flow'].iloc[i])):
            
            signal = (data['short_eff_weighted_acc'].iloc[i] - data['medium_eff_weighted_acc'].iloc[i]) * data['short_flow'].iloc[i]
            
            # Volume Acceleration Divergence condition
            vol_div_condition = (pd.notna(data['short_vol_acc'].iloc[i]) and pd.notna(data['medium_vol_acc'].iloc[i]) and
                               data['short_vol_acc'].iloc[i] < 0 and data['medium_vol_acc'].iloc[i] > 0)
            
            if (vol_div_condition and pd.notna(data['strength_score'].iloc[i]) and 
                data['strength_score'].iloc[i] > 0.25 and pd.notna(data['volatility_weight'].iloc[i]) and
                pd.notna(data['trend_direction'].iloc[i])):
                
                alpha1 = signal * data['volatility_weight'].iloc[i] * (data['trend_direction'].iloc[i] > 0)
                alpha_components.append(alpha1)
        
        # Flow-Enhanced Breakout Alpha
        if (pd.notna(data['short_flow'].iloc[i]) and pd.notna(data['medium_flow'].iloc[i]) and pd.notna(data['long_flow'].iloc[i]) and
            data['short_flow'].iloc[i] > data['medium_flow'].iloc[i] and data['medium_flow'].iloc[i] > data['long_flow'].iloc[i] and
            pd.notna(data['short_eff_weighted_acc'].iloc[i]) and pd.notna(data['medium_eff_weighted_acc'].iloc[i]) and 
            pd.notna(data['long_eff_weighted_acc'].iloc[i])):
            
            signal = (data['short_eff_weighted_acc'].iloc[i] + data['medium_eff_weighted_acc'].iloc[i] + 
                     data['long_eff_weighted_acc'].iloc[i]) * (data['short_flow'].iloc[i] + data['medium_flow'].iloc[i] + data['long_flow'].iloc[i])
            
            if (pd.notna(data['range_efficiency'].iloc[i]) and data['range_efficiency'].iloc[i] > 0.6 and
                pd.notna(data['volume_surge'].iloc[i]) and data['volume_surge'].iloc[i] and
                pd.notna(data['liquidity_score'].iloc[i]) and pd.notna(data['short_momentum'].iloc[i]) and
                pd.notna(data['medium_momentum'].iloc[i])):
                
                momentum_aligned = np.sign(data['short_momentum'].iloc[i]) == np.sign(data['medium_momentum'].iloc[i])
                alpha2 = signal * data['liquidity_score'].iloc[i] * momentum_aligned
                alpha_components.append(alpha2)
        
        # Volatility-Regime Adaptive Divergence Alpha
        if (pd.notna(data['volatility_regime'].iloc[i]) and data['volatility_regime'].iloc[i] > 0 and
            pd.notna(data['short_eff_weighted_acc'].iloc[i]) and pd.notna(data['medium_eff_weighted_acc'].iloc[i]) and
            pd.notna(data['long_eff_weighted_acc'].iloc[i]) and pd.notna(data['short_flow'].iloc[i]) and
            pd.notna(data['medium_flow'].iloc[i]) and pd.notna(data['strength_score'].iloc[i])):
            
            signal1 = (data['short_eff_weighted_acc'].iloc[i] - data['medium_eff_weighted_acc'].iloc[i]) * data['short_flow'].iloc[i]
            signal2 = (data['medium_eff_weighted_acc'].iloc[i] - data['long_eff_weighted_acc'].iloc[i]) * data['medium_flow'].iloc[i]
            signal = (signal1 + signal2) * data['strength_score'].iloc[i]
            
            # Volume Divergence Score
            if (pd.notna(data['short_vol_acc'].iloc[i]) and pd.notna(data['medium_vol_acc'].iloc[i]) and
                pd.notna(data['long_vol_acc'].iloc[i])):
                
                vol_div_score = (data['medium_vol_acc'].iloc[i] - data['short_vol_acc'].iloc[i]) * (data['long_vol_acc'].iloc[i] - data['medium_vol_acc'].iloc[i])
                
                if (vol_div_score > 0 and pd.notna(data['volatility_weight'].iloc[i]) and
                    pd.notna(data['trend_direction'].iloc[i]) and pd.notna(data['volatility_regime'].iloc[i])):
                    
                    alpha3 = signal * data['volatility_weight'].iloc[i] * (data['trend_direction'].iloc[i] * data['volatility_regime'].iloc[i] > 0)
                    alpha_components.append(alpha3)
        
        # Final factor value
        if alpha_components:
            result.iloc[i] = np.mean(alpha_components)
        else:
            result.iloc[i] = 0
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
