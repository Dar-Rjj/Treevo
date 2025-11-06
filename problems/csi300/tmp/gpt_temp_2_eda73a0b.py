import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Regime-Adaptive Momentum-Liquidity Convergence Factor
    """
    data = df.copy()
    
    # Multi-timeframe Volatility & Range Regime Classification
    data['current_range'] = data['high'] - data['low']
    data['median_range_5d'] = data['current_range'].rolling(window=5, min_periods=3).median()
    data['range_ratio'] = data['current_range'] / data['median_range_5d']
    
    # Volatility regime classification
    high_vol_condition = (data['range_ratio'] > 1.2) & (data['current_range'] > (1.5 * data['median_range_5d']))
    low_vol_condition = (data['range_ratio'] < 0.8) & (data['current_range'] < (0.7 * data['median_range_5d']))
    data['vol_regime'] = np.where(high_vol_condition, 'high', 
                                 np.where(low_vol_condition, 'low', 'normal'))
    
    # Momentum Divergence with Efficiency Weighting
    data['momentum_short'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_divergence'] = data['momentum_short'] / (data['momentum_medium'] + 1e-8)
    
    # Price movement efficiency measurement
    data['efficiency_short'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    rolling_high_5d = data['high'].rolling(window=5, min_periods=3).max()
    rolling_low_5d = data['low'].rolling(window=5, min_periods=3).min()
    data['efficiency_medium'] = np.abs(data['close'] - data['close'].shift(5)) / (rolling_high_5d - rolling_low_5d + 1e-8)
    data['efficiency_ratio'] = data['efficiency_short'] / (data['efficiency_medium'] + 1e-8)
    
    # Efficiency-weighted momentum divergence
    data['eff_weighted_momentum'] = data['momentum_divergence'] * data['efficiency_ratio']
    
    # Volume-Liquidity Pressure Integration
    data['liquidity'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['median_liquidity_5d'] = data['liquidity'].rolling(window=5, min_periods=3).median()
    data['liquidity_deviation'] = data['liquidity'] / (data['median_liquidity_5d'] + 1e-8)
    
    # Volume momentum analysis
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_concentration'] = data['volume'] / (data['volume_3d_avg'] + 1e-8)
    data['directional_volume'] = data['volume'] * np.sign(data['close'] - data['close'].shift(1))
    data['volume_pressure'] = data['volume_concentration'] * data['liquidity_deviation']
    
    # Volume-momentum convergence
    data['momentum_volume_convergence'] = data['eff_weighted_momentum'] * data['volume_pressure'] * np.sign(data['directional_volume'])
    
    # Support/Resistance Break Efficiency Enhancement
    data['support_level'] = data['low'].rolling(window=20, min_periods=15).min()
    data['resistance_level'] = data['high'].rolling(window=20, min_periods=15).max()
    
    # Break detection and efficiency
    above_resistance = data['close'] > data['resistance_level'].shift(1)
    below_support = data['close'] < data['support_level'].shift(1)
    
    data['break_level'] = np.where(above_resistance, data['resistance_level'].shift(1),
                                  np.where(below_support, data['support_level'].shift(1), np.nan))
    
    data['penetration_depth'] = np.abs(data['close'] - data['break_level']) / (data['high'] - data['low'] + 1e-8)
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_surge'] = data['volume'] / (data['volume_5d_avg'] + 1e-8)
    data['break_efficiency'] = data['penetration_depth'] * data['volume_surge']
    
    # Apply break efficiency only during confirmed break periods
    data['break_efficiency_adj'] = np.where((above_resistance | below_support), data['break_efficiency'], 0)
    
    # Regime-Adaptive Signal Synthesis
    factor_values = []
    
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['vol_regime']):
            factor_values.append(np.nan)
            continue
            
        regime = data.iloc[i]['vol_regime']
        momentum_vol_conv = data.iloc[i]['momentum_volume_convergence']
        break_eff = data.iloc[i]['break_efficiency_adj']
        range_ratio = data.iloc[i]['range_ratio']
        
        if regime == 'high':
            # High volatility regime: emphasize efficiency, apply volatility dampening
            base_factor = momentum_vol_conv * 1.2  # Emphasize momentum divergence
            volatility_adjusted = base_factor / (range_ratio + 1e-8)  # Dampen by volatility
            break_weighted = break_eff * 1.5  # Enhanced break efficiency
            final_factor = volatility_adjusted + break_weighted
            
        elif regime == 'low':
            # Low volatility regime: focus on volume-liquidity, amplify momentum
            base_factor = momentum_vol_conv * 1.5  # Amplify momentum signals
            volume_weighted = base_factor * data.iloc[i]['volume_pressure']  # Focus on volume-liquidity
            range_expansion = volume_weighted * (1 + data.iloc[i]['range_ratio'])  # Range expansion confirmation
            break_weighted = break_eff * 0.7  # Reduced break efficiency threshold
            final_factor = range_expansion + break_weighted
            
        else:  # normal regime
            # Balanced combination with equal weighting
            momentum_component = momentum_vol_conv * 0.5
            volume_component = data.iloc[i]['volume_pressure'] * 0.5
            base_factor = momentum_component + volume_component
            final_factor = base_factor + break_eff  # Standard break efficiency
            
        factor_values.append(final_factor)
    
    # Create final factor series
    factor_series = pd.Series(factor_values, index=data.index)
    
    # Remove any extreme outliers for stability
    factor_series = np.clip(factor_series, factor_series.quantile(0.01), factor_series.quantile(0.99))
    
    return factor_series
