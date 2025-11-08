import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Detection
    data['prev_close'] = data['close'].shift(1)
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['TR_10d_avg'] = data['TR'].rolling(window=10).mean()
    
    def classify_volatility(row):
        if row['TR'] > 1.5 * row['TR_10d_avg']:
            return 2  # High volatility
        elif row['TR'] < 0.7 * row['TR_10d_avg']:
            return 0  # Low volatility
        else:
            return 1  # Normal volatility
    
    data['volatility_regime'] = data.apply(classify_volatility, axis=1)
    
    # Multi-Timeframe Price Complexity Analysis
    for window in [3, 5]:
        price_changes_sum = 0
        daily_ranges_sum = 0
        
        for i in range(window):
            price_changes_sum += abs(data['close'].shift(i) - data['close'].shift(i+1))
            daily_ranges_sum += (data['high'].shift(i) - data['low'].shift(i))
        
        # Avoid division by zero and log of zero
        daily_ranges_sum = daily_ranges_sum.replace(0, np.nan)
        price_changes_sum = price_changes_sum.replace(0, np.nan)
        
        pfd = 1 + np.log(price_changes_sum) / np.log(daily_ranges_sum)
        data[f'PFD_{window}d'] = pfd
    
    data['PFD_ratio'] = data['PFD_3d'] / data['PFD_5d']
    data['PFD_trend'] = data['PFD_3d'] - data['PFD_5d']
    
    # Multi-Timeframe Volume Complexity Analysis
    for window in [3, 5]:
        volume_changes_sum = 0
        
        for i in range(window):
            volume_changes_sum += abs(data['volume'].shift(i) - data['volume'].shift(i+1))
        
        # Avoid division by zero and log of zero
        volume_changes_sum = volume_changes_sum.replace(0, np.nan)
        
        vfd = 1 + np.log(volume_changes_sum) / np.log(volume_changes_sum)
        data[f'VFD_{window}d'] = vfd
    
    data['VFD_ratio'] = data['VFD_3d'] / data['VFD_5d']
    data['VFD_trend'] = data['VFD_3d'] - data['VFD_5d']
    
    # Momentum Quality Assessment
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum consistency score
    def calculate_consistency(row):
        signs = [np.sign(row['momentum_5d']), np.sign(row['momentum_10d']), np.sign(row['momentum_20d'])]
        positive_count = sum(1 for s in signs if s > 0)
        negative_count = sum(1 for s in signs if s < 0)
        return max(positive_count, negative_count)
    
    data['momentum_consistency'] = data.apply(calculate_consistency, axis=1)
    
    # Momentum strength
    data['momentum_strength'] = (
        abs(data['momentum_5d']) + 
        abs(data['momentum_10d']) + 
        abs(data['momentum_20d'])
    ) / 3
    
    # Intraday Price-Position Analysis
    data['intraday_move'] = data['close'] - data['open']
    data['intraday_strength'] = data['intraday_move'] / (data['high'] - data['low'])
    data['intraday_strength'] = data['intraday_strength'].replace([np.inf, -np.inf], np.nan)
    
    # 5-day price range and position
    data['high_5d'] = data['high'].rolling(window=5).max()
    data['low_5d'] = data['low'].rolling(window=5).min()
    data['position'] = (data['close'] - data['low_5d']) / (data['high_5d'] - data['low_5d'])
    data['position'] = data['position'].replace([np.inf, -np.inf], np.nan)
    data['position_strength'] = 1 - abs(data['position'] - 0.5)
    
    # Complexity-Momentum Divergence Analysis
    data['PFD_VFD_divergence'] = data['PFD_ratio'] - data['VFD_ratio']
    data['complexity_trend_divergence'] = data['PFD_trend'] - data['VFD_trend']
    
    # Volatility-Adaptive Integration
    def volatility_adjustment(row):
        if row['volatility_regime'] == 2:  # High volatility
            # Emphasize complexity divergence, weight volume complexity more
            complexity_weight = 1.2
            volume_weight = 1.3
            momentum_threshold = 0.02
            position_weight = 1.5
        elif row['volatility_regime'] == 0:  # Low volatility
            # Reduce complexity sensitivity, focus on momentum consistency
            complexity_weight = 0.7
            volume_weight = 0.8
            momentum_threshold = 0.01
            position_weight = 1.0
        else:  # Normal volatility
            complexity_weight = 1.0
            volume_weight = 1.0
            momentum_threshold = 0.015
            position_weight = 1.2
        
        return complexity_weight, volume_weight, momentum_threshold, position_weight
    
    # Apply volatility adjustments
    adjustments = data.apply(volatility_adjustment, axis=1, result_type='expand')
    data[['complexity_weight', 'volume_weight', 'momentum_threshold', 'position_weight']] = adjustments
    
    # Quality-Adjusted Factor Generation
    def calculate_quality_metric(row):
        # Divergence quality components
        complexity_div_strength = abs(row['PFD_VFD_divergence']) + abs(row['complexity_trend_divergence'])
        momentum_consistency_weight = row['momentum_consistency'] / 3.0
        volatility_adjustment = 1.0 if row['volatility_regime'] == 1 else 0.8
        
        quality_metric = (
            complexity_div_strength * 
            momentum_consistency_weight * 
            volatility_adjustment * 
            row['position_strength']
        )
        return quality_metric
    
    data['quality_metric'] = data.apply(calculate_quality_metric, axis=1)
    
    # Final factor calculation
    def calculate_final_factor(row):
        # Raw divergence strength
        raw_divergence = (
            row['PFD_VFD_divergence'] * row['complexity_weight'] +
            row['complexity_trend_divergence'] * row['volume_weight']
        )
        
        # Momentum confirmation
        momentum_confirmation = 1.0
        avg_momentum = (row['momentum_5d'] + row['momentum_10d'] + row['momentum_20d']) / 3
        if abs(avg_momentum) > row['momentum_threshold']:
            momentum_confirmation = 1.5 if np.sign(avg_momentum) == np.sign(raw_divergence) else 0.5
        
        # Intraday momentum enhancement
        intraday_enhancement = 1.0 + abs(row['intraday_strength']) * 0.5
        
        # Position strength integration
        position_factor = row['position_strength'] * row['position_weight']
        
        # Final quality-adjusted factor
        final_factor = (
            raw_divergence * 
            row['quality_metric'] * 
            momentum_confirmation * 
            intraday_enhancement * 
            position_factor
        )
        
        return final_factor
    
    data['factor'] = data.apply(calculate_final_factor, axis=1)
    
    return data['factor']
