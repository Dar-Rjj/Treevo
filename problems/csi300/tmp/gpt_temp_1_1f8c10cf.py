import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Acceleration Alpha
    """
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Initialize the factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Volatility Component Calculation
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['gap_vol'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['close_to_close_vol'] = abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['composite_vol_score'] = (data['daily_range_vol'] + data['gap_vol'] + data['close_to_close_vol']) / 3
    
    # Multi-Timeframe Baseline Establishment
    data['short_term_baseline'] = data['composite_vol_score'].rolling(window=21, min_periods=10).median()
    data['medium_term_baseline'] = data['composite_vol_score'].rolling(window=63, min_periods=30).median()
    data['long_term_baseline'] = data['composite_vol_score'].rolling(window=120, min_periods=60).median()
    
    # Hierarchical Regime Classification
    conditions = [
        data['composite_vol_score'] > (2.5 * data['short_term_baseline']),  # Extreme High Volatility
        (data['composite_vol_score'] > (1.8 * data['medium_term_baseline'])) & 
        (data['composite_vol_score'] <= (2.5 * data['short_term_baseline'])),  # High Volatility
        (data['composite_vol_score'] > (1.3 * data['medium_term_baseline'])) & 
        (data['composite_vol_score'] <= (1.8 * data['medium_term_baseline'])),  # Elevated Volatility
        (data['composite_vol_score'] >= (0.7 * data['long_term_baseline'])) & 
        (data['composite_vol_score'] <= (1.3 * data['medium_term_baseline'])),  # Normal Volatility
        (data['composite_vol_score'] >= (0.4 * data['long_term_baseline'])) & 
        (data['composite_vol_score'] < (0.7 * data['long_term_baseline'])),  # Low Volatility
        data['composite_vol_score'] < (0.4 * data['long_term_baseline'])  # Extreme Low Volatility
    ]
    
    choices = ['extreme_high', 'high', 'elevated', 'normal', 'low', 'extreme_low']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Advanced Volume Persistence Analysis
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_15'] = data['volume'].rolling(window=15, min_periods=8).mean()
    data['volume_ma_30'] = data['volume'].rolling(window=30, min_periods=15).mean()
    data['volume_trend_hierarchy'] = (data['volume_ma_5'] / data['volume_ma_15']) * (data['volume_ma_15'] / data['volume_ma_30'])
    
    # Volume Persistence Metrics
    data['volume_dir_persistence'] = data['volume'].rolling(window=8, min_periods=5).apply(
        lambda x: sum(x[i] > x[i-1] for i in range(1, len(x)) if not pd.isna(x[i]) and not pd.isna(x[i-1])), raw=False
    )
    data['volume_mag_persistence'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: sum(x[i] > (1.2 * x[i-1]) for i in range(1, len(x)) if not pd.isna(x[i]) and not pd.isna(x[i-1])), raw=False
    )
    data['volume_trend_persistence'] = data['volume_ma_5'].rolling(window=10, min_periods=6).apply(
        lambda x: sum(x[i] > data['volume_ma_15'].iloc[i] for i in range(len(x))), raw=False
    )
    data['composite_persistence_score'] = (data['volume_dir_persistence'] + data['volume_mag_persistence'] + data['volume_trend_persistence']) / 23
    
    # Volume Regime Classification
    vol_persistence_conditions = [
        data['composite_persistence_score'] >= 0.7,  # High Persistence
        (data['composite_persistence_score'] >= 0.4) & (data['composite_persistence_score'] < 0.7),  # Medium Persistence
        (data['composite_persistence_score'] >= 0.2) & (data['composite_persistence_score'] < 0.4),  # Low Persistence
        data['composite_persistence_score'] < 0.2  # No Persistence
    ]
    vol_persistence_choices = ['high_persistence', 'medium_persistence', 'low_persistence', 'no_persistence']
    data['volume_regime'] = np.select(vol_persistence_conditions, vol_persistence_choices, default='medium_persistence')
    
    # Multi-Scale Momentum Integration
    data['return_1d'] = data['close'] / data['close'].shift(1) - 1
    data['return_2d'] = data['close'] / data['close'].shift(2) - 1
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['ultra_short_score'] = 0.5 * data['return_1d'] + 0.3 * data['return_2d'] + 0.2 * data['return_3d']
    
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['return_7d'] = data['close'] / data['close'].shift(7) - 1
    data['return_10d'] = data['close'] / data['close'].shift(10) - 1
    data['short_term_score'] = 0.4 * data['return_5d'] + 0.35 * data['return_7d'] + 0.25 * data['return_10d']
    
    data['return_15d'] = data['close'] / data['close'].shift(15) - 1
    data['return_20d'] = data['close'] / data['close'].shift(20) - 1
    data['return_25d'] = data['close'] / data['close'].shift(25) - 1
    data['medium_term_score'] = 0.35 * data['return_15d'] + 0.35 * data['return_20d'] + 0.3 * data['return_25d']
    
    # Momentum Regime Classification
    momentum_conditions = [
        (data['ultra_short_score'] > 0.05) & (data['short_term_score'] > 0.05) & (data['medium_term_score'] > 0.05),  # Strong Momentum
        ((data['ultra_short_score'] > 0.02) + (data['short_term_score'] > 0.02) + (data['medium_term_score'] > 0.02)) >= 2,  # Moderate Momentum
        (data['ultra_short_score'].abs() <= 0.02) & (data['short_term_score'].abs() <= 0.02) & (data['medium_term_score'].abs() <= 0.02),  # Neutral Momentum
        ((data['ultra_short_score'] < -0.02) + (data['short_term_score'] < -0.02) + (data['medium_term_score'] < -0.02)) >= 2,  # Moderate Reversion
        (data['ultra_short_score'] < -0.05) & (data['short_term_score'] < -0.05) & (data['medium_term_score'] < -0.05)  # Strong Reversion
    ]
    momentum_choices = ['strong_momentum', 'moderate_momentum', 'neutral', 'moderate_reversion', 'strong_reversion']
    data['momentum_regime'] = np.select(momentum_conditions, momentum_choices, default='neutral')
    
    # Adaptive Integration Engine
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['composite_vol_score']) or pd.isna(data.iloc[i]['composite_persistence_score']):
            factor.iloc[i] = 0
            continue
            
        vol_regime = data.iloc[i]['vol_regime']
        volume_regime = data.iloc[i]['volume_regime']
        momentum_regime = data.iloc[i]['momentum_regime']
        
        base_signal = data.iloc[i]['medium_term_score']  # Default to medium-term momentum
        
        # Extreme High Volatility Regime
        if vol_regime == 'extreme_high':
            if volume_regime in ['high_persistence', 'medium_persistence']:
                multipliers = {
                    'strong_momentum': 0.2,
                    'moderate_momentum': 0.4,
                    'neutral': 1.5,
                    'moderate_reversion': 2.0,
                    'strong_reversion': 2.5
                }
                multiplier = multipliers.get(momentum_regime, 1.0)
            else:
                multipliers = {
                    'strong_momentum': 0.1,
                    'moderate_momentum': 0.2,
                    'neutral': 0.75,
                    'moderate_reversion': 1.0,
                    'strong_reversion': 1.25
                }
                multiplier = multipliers.get(momentum_regime, 0.5)
                base_signal = data.iloc[i]['ultra_short_score']  # Focus on ultra-short-term
        
        # High Volatility Regime
        elif vol_regime == 'high':
            if volume_regime in ['high_persistence', 'medium_persistence']:
                multipliers = {
                    'strong_momentum': 0.8,
                    'moderate_momentum': 0.6,
                    'neutral': 0.4,
                    'moderate_reversion': 0.5,
                    'strong_reversion': 0.3
                }
                multiplier = multipliers.get(momentum_regime, 0.5) * 1.3  # Volume persistence enhancement
            else:
                multipliers = {
                    'strong_momentum': 0.6,
                    'moderate_momentum': 0.4,
                    'neutral': 0.3,
                    'moderate_reversion': 0.2,
                    'strong_reversion': 0.2
                }
                multiplier = multipliers.get(momentum_regime, 0.3)
                base_signal = data.iloc[i]['short_term_score']  # Focus on short-term
        
        # Elevated Volatility Regime
        elif vol_regime == 'elevated':
            if volume_regime == 'high_persistence':
                multipliers = {
                    'strong_momentum': 1.2,
                    'moderate_momentum': 0.9,
                    'neutral': 0.7,
                    'moderate_reversion': 0.8,
                    'strong_reversion': 0.7
                }
            elif volume_regime == 'medium_persistence':
                multipliers = {
                    'strong_momentum': 0.9,
                    'moderate_momentum': 0.7,
                    'neutral': 0.5,
                    'moderate_reversion': 0.6,
                    'strong_reversion': 0.5
                }
            elif volume_regime == 'low_persistence':
                multipliers = {
                    'strong_momentum': 0.8,
                    'moderate_momentum': 1.0,
                    'neutral': 1.2,
                    'moderate_reversion': 1.5,
                    'strong_reversion': 1.3
                }
            else:  # no_persistence
                multipliers = {
                    'strong_momentum': 0.6,
                    'moderate_momentum': 0.5,
                    'neutral': 0.4,
                    'moderate_reversion': 0.3,
                    'strong_reversion': 0.3
                }
            multiplier = multipliers.get(momentum_regime, 0.7)
            
            # Multi-timeframe momentum integration
            momentum_signs = [np.sign(data.iloc[i]['ultra_short_score']), 
                            np.sign(data.iloc[i]['short_term_score']), 
                            np.sign(data.iloc[i]['medium_term_score'])]
            if len(set(momentum_signs)) == 1:  # Strong agreement
                multiplier *= 1.3
            elif len(set(momentum_signs)) == 2:  # Mixed signals
                multiplier *= 0.7
            else:  # Contradictory signals
                multiplier *= 0.2
        
        # Normal Volatility Regime
        elif vol_regime == 'normal':
            if volume_regime == 'high_persistence':
                base_signal = data.iloc[i]['medium_term_score']
                multipliers = {
                    'strong_momentum': 1.4,
                    'moderate_momentum': 1.1,
                    'neutral': 0.9,
                    'moderate_reversion': 1.0,
                    'strong_reversion': 0.9
                }
            elif volume_regime == 'medium_persistence':
                base_signal = data.iloc[i]['short_term_score']
                multipliers = {
                    'strong_momentum': 1.1,
                    'moderate_momentum': 0.9,
                    'neutral': 0.7,
                    'moderate_reversion': 0.8,
                    'strong_reversion': 0.7
                }
            elif volume_regime == 'low_persistence':
                base_signal = data.iloc[i]['ultra_short_score']
                multipliers = {
                    'strong_momentum': 0.8,
                    'moderate_momentum': 0.6,
                    'neutral': 0.5,
                    'moderate_reversion': 0.6,
                    'strong_reversion': 0.5
                }
            else:  # no_persistence
                base_signal = -data.iloc[i]['ultra_short_score']  # Mean reversion
                multipliers = {
                    'strong_momentum': 0.6,
                    'moderate_momentum': 0.8,
                    'neutral': 1.0,
                    'moderate_reversion': 0.9,
                    'strong_reversion': 0.7
                }
            multiplier = multipliers.get(momentum_regime, 0.8)
            
            # Momentum consistency bonus
            momentum_signs = [np.sign(data.iloc[i]['ultra_short_score']), 
                            np.sign(data.iloc[i]['short_term_score']), 
                            np.sign(data.iloc[i]['medium_term_score'])]
            if len(set(momentum_signs)) == 1:
                multiplier *= 1.4
            elif len(set(momentum_signs)) == 2:
                multiplier *= 1.2
            else:
                multiplier *= 0.7
        
        # Low Volatility Regime
        elif vol_regime == 'low':
            volume_breakout = data.iloc[i]['volume'] > (2.0 * data.iloc[i]['volume_ma_30'])
            price_breakout = abs(data.iloc[i]['ultra_short_score']) > 0.03
            
            if volume_breakout and price_breakout:
                multiplier = 1.8
            elif volume_breakout:
                multiplier = 1.2
            elif price_breakout:
                multiplier = 0.8
            else:
                multiplier = 0.3
                base_signal = -data.iloc[i]['ultra_short_score']  # Mean reversion
        
        # Extreme Low Volatility Regime
        else:  # extreme_low
            micro_volume_spike = data.iloc[i]['volume'] > (1.5 * data.iloc[i-1]['volume']) if i > 0 else False
            price_movement = abs(data.iloc[i]['return_1d']) > 0.01
            
            if micro_volume_spike and price_movement:
                multiplier = 1.5
            elif data.iloc[i]['composite_persistence_score'] > 0.5:
                multiplier = 0.8
            else:
                multiplier = 0.1
                base_signal = -data.iloc[i]['ultra_short_score']  # Ultra-short reversion
            
            # Position size limitation
            multiplier *= 0.5
        
        factor.iloc[i] = base_signal * multiplier
    
    # Fill NaN values with 0
    factor = factor.fillna(0)
    
    return factor
