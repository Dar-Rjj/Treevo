import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Momentum Dynamics
    # Multi-Scale Momentum Efficiency
    data['short_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_acceleration'] = (data['short_term_momentum'] - data['medium_term_momentum']) / np.where(data['short_term_momentum'] != 0, data['short_term_momentum'], np.nan)
    data['price_efficiency'] = (data['close'] - data['open']) / np.where((data['high'] - data['low']) != 0, (data['high'] - data['low']), np.nan)
    
    # Range Fractality Momentum
    data['short_term_range_scaling'] = (data['high'] - data['low']) / ((data['high'] - data['low']).rolling(window=5).mean())
    data['medium_term_range_scaling'] = (data['high'] - data['low']) / ((data['high'] - data['low']).rolling(window=20).mean())
    data['range_fractality_ratio'] = data['short_term_range_scaling'] / np.where(data['medium_term_range_scaling'] != 0, data['medium_term_range_scaling'], np.nan)
    
    # Momentum-Fractality Synthesis
    data['efficiency_momentum'] = data['momentum_acceleration'] * data['price_efficiency']
    data['range_momentum'] = data['range_fractality_ratio'] * data['price_efficiency']
    data['fractal_momentum_alignment'] = data['efficiency_momentum'] * data['range_momentum']
    
    # Microstructure Asymmetry Patterns
    # Order Flow Asymmetry
    data['opening_order_flow_efficiency'] = (data['open'] - data['close'].shift(1)) * (data['amount'] / np.where(data['volume'] != 0, data['volume'], np.nan))
    data['closing_order_pressure'] = (data['close'] - (data['high'] + data['low'])/2) * ((data['amount'] - data['amount'].shift(1)) / np.where(data['amount'].shift(1) != 0, data['amount'].shift(1), np.nan))
    data['opening_closing_order_asymmetry'] = data['opening_order_flow_efficiency'] - data['closing_order_pressure']
    data['order_flow_concentration'] = (data['volume'] / np.where(data['amount'] != 0, data['amount'], np.nan)) - (data['volume'].shift(1) / np.where(data['amount'].shift(1) != 0, data['amount'].shift(1), np.nan))
    
    # Volume-Intensity Asymmetry
    data['volume_scaling'] = data['volume'] / data['volume'].rolling(window=5).mean()
    volume_sign = np.sign(data['volume_scaling'] - 1)
    amount_sign = np.sign(data['amount'] / data['amount'].rolling(window=5).mean() - 1)
    data['amount_volume_coordination'] = volume_sign * amount_sign
    data['volume_fractality_momentum'] = data['volume_scaling'] * data['amount_volume_coordination']
    data['volume_price_order_imbalance'] = ((data['close'] - data['close'].shift(1)) / np.where(data['close'].shift(1) != 0, data['close'].shift(1), np.nan)) - ((data['volume'] - data['volume'].shift(1)) / np.where(data['volume'].shift(1) != 0, data['volume'].shift(1), np.nan))
    
    # Asymmetry Integration
    data['order_flow_asymmetry_score'] = data['opening_closing_order_asymmetry'] * data['order_flow_concentration']
    data['volume_asymmetry_score'] = data['volume_fractality_momentum'] * data['volume_price_order_imbalance']
    data['cross_asymmetry_alignment'] = data['order_flow_asymmetry_score'] * data['volume_asymmetry_score']
    
    # Structural Break Detection
    # Volume-Confirmed Price Breaks
    data['price_break_intensity'] = abs(data['close'] - data['close'].shift(1)) / np.where((data['high'].shift(1) - data['low'].shift(1)) != 0, (data['high'].shift(1) - data['low'].shift(1)), np.nan)
    data['volume_confirmation'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['break_fractality_signal'] = data['price_break_intensity'] * data['volume_confirmation']
    
    # Range Asymmetry Patterns
    # Calculate up/down days and their ranges
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    # Calculate rolling average ranges for up and down days
    up_day_ranges = []
    down_day_ranges = []
    
    for i in range(len(data)):
        if i >= 4:
            up_mask = up_days.iloc[i-4:i+1]
            down_mask = down_days.iloc[i-4:i+1]
            
            up_range = (data['high'].iloc[i-4:i+1] - data['low'].iloc[i-4:i+1])[up_mask]
            down_range = (data['high'].iloc[i-4:i+1] - data['low'].iloc[i-4:i+1])[down_mask]
            
            up_day_ranges.append(up_range.mean() if len(up_range) > 0 else np.nan)
            down_day_ranges.append(down_range.mean() if len(down_range) > 0 else np.nan)
        else:
            up_day_ranges.append(np.nan)
            down_day_ranges.append(np.nan)
    
    data['up_day_range_expansion'] = (data['high'] - data['low']) / pd.Series(up_day_ranges, index=data.index)
    data['down_day_range_compression'] = (data['high'] - data['low']) / pd.Series(down_day_ranges, index=data.index)
    data['range_asymmetry_index'] = data['up_day_range_expansion'] / np.where(data['down_day_range_compression'] != 0, data['down_day_range_compression'], np.nan)
    
    # Break Persistence
    threshold = data['price_break_intensity'].quantile(0.7)
    data['consecutive_break_days'] = 0
    data['volume_persistence'] = 0.0
    
    for i in range(1, len(data)):
        if i >= 5:
            break_mask = data['price_break_intensity'].iloc[i-4:i+1] > threshold
            data.loc[data.index[i], 'consecutive_break_days'] = break_mask.sum()
            data.loc[data.index[i], 'volume_persistence'] = data['volume_confirmation'].iloc[i-4:i+1][break_mask].mean()
    
    data['break_persistence'] = data['consecutive_break_days'] * data['volume_persistence']
    
    # Regime-Adaptive Synthesis
    # Volatility Regime Classification
    data['short_term_volatility'] = ((data['high'] - data['low']) / data['close']).rolling(window=5).mean()
    data['medium_term_volatility'] = ((data['high'] - data['low']) / data['close']).rolling(window=20).mean()
    data['volatility_regime'] = data['short_term_volatility'] / np.where(data['medium_term_volatility'] != 0, data['medium_term_volatility'], np.nan)
    
    # Flow Regime Detection
    high_flow = (data['volume'] / data['volume'].rolling(window=5).mean() > 1.2) & (data['amount'] / data['amount'].rolling(window=5).mean() > 1.1)
    low_flow = (data['volume'] / data['volume'].rolling(window=5).mean() < 0.8) & (data['amount'] / data['amount'].rolling(window=5).mean() < 0.9)
    data['flow_regime_score'] = high_flow.astype(int) - low_flow.astype(int)
    
    # Regime-Weighted Integration
    data['volatility_adjusted_momentum'] = data['fractal_momentum_alignment'] * data['volatility_regime']
    data['flow_enhanced_asymmetry'] = data['cross_asymmetry_alignment'] * data['flow_regime_score']
    data['regime_coherent_signal'] = data['volatility_adjusted_momentum'] * data['flow_enhanced_asymmetry']
    
    # Efficiency Confirmation
    # Trade Efficiency Metrics
    data['gap_efficiency_alignment'] = ((data['open'] - data['close'].shift(1)) / np.where((data['high'].shift(1) - data['low'].shift(1)) != 0, (data['high'].shift(1) - data['low'].shift(1)), np.nan)) * ((data['close'] - data['open']) / np.where((data['high'] - data['low']) != 0, (data['high'] - data['low']), np.nan))
    data['microstructure_timing'] = (data['amount'] / np.where((data['high'] - data['low']) != 0, (data['high'] - data['low']), np.nan)) * data['amount_volume_coordination']
    data['price_impact_efficiency'] = ((data['close'] - data['open']) / np.where((data['high'] - data['low']) != 0, (data['high'] - data['low']), np.nan)) * (data['amount'] / np.where(data['volume'] != 0, data['volume'], np.nan))
    
    # Flow Consistency
    data['amount_confirmation'] = data['amount'] / data['amount'].rolling(window=5).mean()
    data['flow_consistency'] = np.sign(data['volume_confirmation']) * np.sign(data['amount_confirmation'])
    
    # Efficiency Integration
    data['core_efficiency'] = data['gap_efficiency_alignment'] * data['microstructure_timing']
    data['flow_confirmed_efficiency'] = data['core_efficiency'] * data['flow_consistency']
    data['final_efficiency_score'] = data['flow_confirmed_efficiency'] * data['price_impact_efficiency']
    
    # Composite Alpha Construction
    # Core Factor Integration
    data['momentum_foundation'] = data['regime_coherent_signal'] * data['fractal_momentum_alignment']
    data['asymmetry_enhancement'] = data['momentum_foundation'] * data['cross_asymmetry_alignment']
    data['break_confirmation'] = data['asymmetry_enhancement'] * data['break_fractality_signal']
    data['efficiency_finalization'] = data['break_confirmation'] * data['final_efficiency_score']
    
    # Persistence Weighting
    # Calculate consecutive days with same sign of Cross-Asymmetry Alignment
    data['asymmetry_duration'] = 0
    current_sign = 0
    current_streak = 0
    
    for i in range(len(data)):
        if not np.isnan(data['cross_asymmetry_alignment'].iloc[i]):
            sign = np.sign(data['cross_asymmetry_alignment'].iloc[i])
            if sign == current_sign:
                current_streak += 1
            else:
                current_streak = 1
                current_sign = sign
            data.loc[data.index[i], 'asymmetry_duration'] = current_streak
        else:
            current_streak = 0
            current_sign = 0
    
    data['persistence_multiplier'] = data['asymmetry_duration'] * data['break_persistence']
    
    # Final Alpha Output
    data['raw_alpha'] = data['efficiency_finalization'] * data['range_asymmetry_index']
    data['persistence_enhanced_alpha'] = data['raw_alpha'] * data['persistence_multiplier']
    
    # Clean up and return final alpha
    alpha = data['persistence_enhanced_alpha'].replace([np.inf, -np.inf], np.nan)
    
    return alpha
