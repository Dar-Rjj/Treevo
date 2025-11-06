import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Rejection Efficiency Factor
    Combines multi-scale price rejection with efficiency metrics and market regime detection
    """
    data = df.copy()
    
    # Multi-Scale Price Rejection Calculations
    # Intraday Rejection (Daily)
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['net_rejection'] = data['upper_shadow'] - data['lower_shadow']
    
    # Short-term Rejection (3-day)
    data['close_3d_max'] = data['close'].rolling(window=3, min_periods=1).max()
    data['close_3d_min'] = data['close'].rolling(window=3, min_periods=1).min()
    data['high_rejection_3d'] = data['high'] - data['close_3d_max']
    data['low_rejection_3d'] = data['close_3d_min'] - data['low']
    data['net_rejection_3d'] = data['high_rejection_3d'] - data['low_rejection_3d']
    
    # Medium-term Rejection (10-day)
    data['close_10d_max'] = data['close'].rolling(window=10, min_periods=1).max()
    data['close_10d_min'] = data['close'].rolling(window=10, min_periods=1).min()
    data['high_rejection_10d'] = data['high'] - data['close_10d_max']
    data['low_rejection_10d'] = data['close_10d_min'] - data['low']
    data['net_rejection_10d'] = data['high_rejection_10d'] - data['low_rejection_10d']
    
    # Multi-Dimensional Efficiency Metrics
    # Price Efficiency Analysis
    daily_range = data['high'] - data['low']
    daily_range = daily_range.replace(0, np.nan)  # Avoid division by zero
    data['intraday_efficiency'] = np.abs(data['close'] - data['open']) / daily_range
    data['volume_weighted_efficiency'] = data['intraday_efficiency'] * data['volume']
    
    # Efficiency Momentum
    data['efficiency_3d_ago'] = data['intraday_efficiency'].shift(3)
    data['efficiency_momentum'] = data['intraday_efficiency'] / data['efficiency_3d_ago'] - 1
    
    # Amount Flow Analysis
    # Estimate large transaction concentration (top 20% by amount)
    data['amount_rank'] = data['amount'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['large_amount_pct'] = (data['amount_rank'] > 0.8).astype(float)
    
    # Amount Flow Direction
    data['bullish_flow'] = data['large_amount_pct'] * (data['close'] > data['open']).astype(float)
    data['bearish_flow'] = data['large_amount_pct'] * (data['close'] < data['open']).astype(float)
    data['net_amount_flow'] = data['bullish_flow'] - data['bearish_flow']
    
    # Range Dynamics Analysis
    data['daily_range'] = daily_range
    data['range_5d_ago'] = data['daily_range'].shift(5)
    data['range_momentum'] = (data['daily_range'] - data['range_5d_ago']) / data['range_5d_ago']
    data['range_expansion'] = data['daily_range'] / data['daily_range'].shift(1)
    
    # Adaptive Market Regime Detection
    # Volatility-Efficiency Regime
    efficiency_ma = data['intraday_efficiency'].rolling(window=10, min_periods=1).mean()
    efficiency_std = data['intraday_efficiency'].rolling(window=10, min_periods=1).std()
    data['efficiency_zscore'] = (data['intraday_efficiency'] - efficiency_ma) / efficiency_std
    
    # High Efficiency: z-score > 1, Low Efficiency: z-score < -1
    data['high_efficiency_regime'] = (data['efficiency_zscore'] > 1).astype(float)
    data['low_efficiency_regime'] = (data['efficiency_zscore'] < -1).astype(float)
    data['transition_efficiency'] = ((data['efficiency_zscore'] >= -1) & (data['efficiency_zscore'] <= 1)).astype(float)
    
    # Volume-Amount Regime
    amount_concentration_ma = data['large_amount_pct'].rolling(window=10, min_periods=1).mean()
    data['high_concentration_regime'] = (data['large_amount_pct'] > amount_concentration_ma + 0.1).astype(float)
    data['low_concentration_regime'] = (data['large_amount_pct'] < amount_concentration_ma - 0.1).astype(float)
    data['mixed_flow_regime'] = ((data['large_amount_pct'] >= amount_concentration_ma - 0.1) & 
                                (data['large_amount_pct'] <= amount_concentration_ma + 0.1)).astype(float)
    
    # Rejection-Confirmation Regime
    rejection_strength = (data['net_rejection'].abs() + data['net_rejection_3d'].abs() + 
                         data['net_rejection_10d'].abs()) / 3
    rejection_strength_ma = rejection_strength.rolling(window=10, min_periods=1).mean()
    
    data['strong_rejection_regime'] = (rejection_strength > rejection_strength_ma + rejection_strength_ma.std()).astype(float)
    data['weak_rejection_regime'] = (rejection_strength < rejection_strength_ma - rejection_strength_ma.std()).astype(float)
    data['mixed_rejection_regime'] = ((rejection_strength >= rejection_strength_ma - rejection_strength_ma.std()) & 
                                     (rejection_strength <= rejection_strength_ma + rejection_strength_ma.std())).astype(float)
    
    # Regime-Adaptive Interactions
    # Calculate regime-weighted rejection scores
    data['regime_weighted_rejection'] = 0.0
    
    # High Efficiency + Strong Rejection → Breakout Confirmation
    breakout_condition = (data['high_efficiency_regime'] == 1) & (data['strong_rejection_regime'] == 1)
    data.loc[breakout_condition, 'regime_weighted_rejection'] = (
        data['net_rejection'] * 1.5 + data['net_rejection_3d'] * 1.2 + data['net_rejection_10d'] * 1.0
    )
    
    # Low Efficiency + Strong Rejection → Mean-Reversion Setup
    mean_reversion_condition = (data['low_efficiency_regime'] == 1) & (data['strong_rejection_regime'] == 1)
    data.loc[mean_reversion_condition, 'regime_weighted_rejection'] = (
        -data['net_rejection'] * 1.3 - data['net_rejection_3d'] * 1.1 - data['net_rejection_10d'] * 0.9
    )
    
    # Transition Phase + Mixed Rejection → Cautious Signal
    cautious_condition = (data['transition_efficiency'] == 1) & (data['mixed_rejection_regime'] == 1)
    data.loc[cautious_condition, 'regime_weighted_rejection'] = (
        data['net_rejection'] * 0.5 + data['net_rejection_3d'] * 0.3 + data['net_rejection_10d'] * 0.2
    )
    
    # Default case for other regime combinations
    default_condition = ~(breakout_condition | mean_reversion_condition | cautious_condition)
    data.loc[default_condition, 'regime_weighted_rejection'] = (
        data['net_rejection'] * 0.8 + data['net_rejection_3d'] * 0.6 + data['net_rejection_10d'] * 0.4
    )
    
    # Final Composite Factor Construction
    # Integrate Efficiency Momentum Enhancement
    efficiency_momentum_enhancement = 1 + data['efficiency_momentum'].clip(-1, 1)
    
    # Scale by amount flow concentration and direction
    amount_flow_adjustment = 1 + data['net_amount_flow'] * 0.5
    
    # Apply range dynamics as volatility adjustment
    range_adjustment = 1 / (1 + data['range_expansion'].abs().clip(upper=2))
    
    # Generate Final Alpha Output
    alpha_factor = (
        data['regime_weighted_rejection'] * 
        efficiency_momentum_enhancement * 
        amount_flow_adjustment * 
        range_adjustment
    )
    
    # Clean up intermediate columns
    columns_to_drop = [
        'upper_shadow', 'lower_shadow', 'net_rejection',
        'close_3d_max', 'close_3d_min', 'high_rejection_3d', 'low_rejection_3d', 'net_rejection_3d',
        'close_10d_max', 'close_10d_min', 'high_rejection_10d', 'low_rejection_10d', 'net_rejection_10d',
        'intraday_efficiency', 'volume_weighted_efficiency', 'efficiency_3d_ago', 'efficiency_momentum',
        'amount_rank', 'large_amount_pct', 'bullish_flow', 'bearish_flow', 'net_amount_flow',
        'daily_range', 'range_5d_ago', 'range_momentum', 'range_expansion',
        'efficiency_zscore', 'high_efficiency_regime', 'low_efficiency_regime', 'transition_efficiency',
        'high_concentration_regime', 'low_concentration_regime', 'mixed_flow_regime',
        'strong_rejection_regime', 'weak_rejection_regime', 'mixed_rejection_regime',
        'regime_weighted_rejection'
    ]
    
    # Only drop columns that exist
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=existing_columns)
    
    return alpha_factor
