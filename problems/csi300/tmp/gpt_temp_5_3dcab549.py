import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Entropy-Efficiency Flow System
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['high_low_range'] = df['high'] - df['low']
    df['prev_high_low_range'] = df['high_low_range'].shift(1)
    df['body_position'] = (df['close'] - df['low']) / df['high_low_range']
    df['center_deviation'] = np.abs(df['close'] - (df['high'] + df['low']) / 2) / df['high_low_range']
    df['efficiency_ratio'] = (df['close'] - df['open']) / df['high_low_range']
    df['avg_trade_size'] = df['amount'] / df['volume']
    
    # Volume calculations
    df['volume_3d_avg'] = df['volume'].rolling(window=3, min_periods=1).mean()
    df['volume_concentration'] = df['volume'] / df['volume_3d_avg']
    df['volume_velocity'] = df['volume'] / df['volume'].shift(1)
    
    # Directional probabilities
    df['p_up'] = (df['price_change'] > 0).rolling(window=5, min_periods=1).mean()
    df['p_down'] = (df['price_change'] < 0).rolling(window=5, min_periods=1).mean()
    
    # Volume regime probabilities
    df['p_high_vol'] = (df['volume'] > df['volume'].shift(1)).rolling(window=5, min_periods=1).mean()
    df['p_low_vol'] = (df['volume'] < df['volume'].shift(1)).rolling(window=5, min_periods=1).mean()
    
    # Avoid division by zero and log(0)
    epsilon = 1e-8
    
    # Entropy-Efficiency Integration
    df['directional_entropy_efficiency'] = (
        -df['p_up'] * np.log(df['p_up'] + epsilon) 
        - df['p_down'] * np.log(df['p_down'] + epsilon)
    ) * df['price_change'] / (df['high_low_range'] + epsilon)
    
    df['volatility_ratio'] = df['high_low_range'] / (df['prev_high_low_range'] + epsilon)
    df['volatility_change'] = np.abs(df['volatility_ratio'] - 1)
    df['volatility_entropy_efficiency'] = (
        df['volatility_change'] * np.log(df['volatility_change'] + epsilon)
    ) * df['efficiency_ratio']
    
    df['volume_entropy_efficiency'] = (
        -df['p_high_vol'] * np.log(df['p_high_vol'] + epsilon)
        - df['p_low_vol'] * np.log(df['p_low_vol'] + epsilon)
    ) * df['volume'] / (df['volume_3d_avg'] + epsilon)
    
    # Flow-Pressure Dynamics
    df['entropy_weighted_pressure'] = (
        df['body_position'] * df['volume'] * df['directional_entropy_efficiency']
    )
    
    df['efficiency_flow_decay'] = (
        df['efficiency_ratio'] * df['volume_velocity'] 
        * np.sign(df['close'] - df['close'].shift(5))
    )
    
    df['trade_size_momentum'] = df['avg_trade_size'] / (df['avg_trade_size'].shift(1) + epsilon)
    df['concentration_flow_pressure'] = (
        df['center_deviation'] * df['volume_concentration'] * df['trade_size_momentum']
    )
    
    # Microstructure Entropy-Efficiency
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
    df['rejection_measure'] = df['upper_shadow'] - df['lower_shadow']
    df['efficiency_ratio_change'] = np.abs(df['efficiency_ratio'] - df['efficiency_ratio'].shift(1))
    
    df['rejection_entropy_efficiency'] = (
        df['rejection_measure'] * df['efficiency_ratio_change']
        * np.log(df['efficiency_ratio_change'] + epsilon)
    )
    
    df['trade_size_ratio'] = df['avg_trade_size'] / (df['avg_trade_size'].shift(1) + epsilon)
    df['trade_size_change'] = np.abs(df['trade_size_ratio'] - 1)
    df['trade_size_entropy_flow'] = (
        df['trade_size_ratio'] * np.log(df['trade_size_change'] + epsilon)
        * df['volume_concentration']
    )
    
    df['order_imbalance'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high_low_range'] + epsilon)
    df['efficiency_momentum'] = df['efficiency_ratio'] / (df['efficiency_ratio'].shift(1) + epsilon)
    df['volume_efficiency_ratio'] = df['volume_velocity'] / (df['efficiency_momentum'] + epsilon)
    
    df['order_flow_entropy_efficiency'] = (
        df['order_imbalance'] * df['volume'] * np.abs(df['volume_efficiency_ratio'])
        * np.log(np.abs(df['volume_efficiency_ratio']) + epsilon)
    )
    
    # Entropy-Efficiency Momentum
    df['entropy_efficiency_acceleration'] = (
        df['directional_entropy_efficiency'] / (df['directional_entropy_efficiency'].shift(1) + epsilon)
        * (df['efficiency_ratio'] - df['efficiency_ratio'].shift(1))
    )
    
    df['current_pressure'] = df['body_position'] * df['volume'] / (df['volume'] + df['volume'].shift(1) + epsilon)
    df['prev_pressure'] = (
        (df['close'].shift(1) - df['low'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + epsilon)
        * df['volume'].shift(1) / (df['volume'].shift(1) + df['volume'].shift(2) + epsilon)
    )
    df['pressure_momentum'] = df['current_pressure'] - df['prev_pressure']
    
    df['current_concentration'] = df['center_deviation'] * df['volume_concentration']
    df['prev_concentration'] = (
        np.abs(df['close'].shift(1) - (df['high'].shift(1) + df['low'].shift(1)) / 2) 
        / (df['high'].shift(1) - df['low'].shift(1) + epsilon)
        * (df['volume'].shift(1) / (df['volume'].rolling(window=3, min_periods=1).mean().shift(1) + epsilon))
    )
    df['concentration_efficiency_acceleration'] = df['current_concentration'] / (df['prev_concentration'] + epsilon)
    
    # Multi-Regime Classification
    df['entropy_regime'] = df['directional_entropy_efficiency'] > 0
    df['efficiency_regime'] = df['efficiency_ratio'] > 0.5
    df['volume_persistence'] = (
        (df['volume'] > df['volume'].shift(1)).rolling(window=3, min_periods=1).sum() >= 2
    )
    df['volatility_momentum'] = df['high_low_range'] - df['high_low_range'].shift(1)
    df['volatility_regime'] = df['volatility_momentum'] > 0
    
    # Cross-Regime Alpha Synthesis
    for i in range(len(df)):
        if i < 5:  # Ensure enough data for calculations
            result.iloc[i] = 0
            continue
            
        row = df.iloc[i]
        
        # Regime-specific alphas
        if (row['entropy_regime'] and row['efficiency_regime'] and 
            row['volume_persistence'] and row['volatility_regime']):
            # High Entropy & High Efficiency & Persistent & Expanding
            alpha = (row['entropy_weighted_pressure'] * row['efficiency_flow_decay'] 
                    * (-row['volatility_entropy_efficiency']))
            
        elif (row['entropy_regime'] and not row['efficiency_regime'] and 
              not row['volume_persistence'] and not row['volatility_regime']):
            # High Entropy & Low Efficiency & Transient & Contracting
            alpha = (row['order_flow_entropy_efficiency'] * row['trade_size_entropy_flow'] 
                    * row['concentration_flow_pressure'])
            
        elif (not row['entropy_regime'] and row['efficiency_regime'] and 
              row['volume_persistence'] and not row['volatility_regime']):
            # Low Entropy & High Efficiency & Persistent & Contracting
            alpha = (row['rejection_entropy_efficiency'] * row['pressure_momentum'] 
                    * (-row['entropy_efficiency_acceleration']))
            
        elif (not row['entropy_regime'] and not row['efficiency_regime'] and 
              not row['volume_persistence'] and row['volatility_regime']):
            # Low Entropy & Low Efficiency & Transient & Expanding
            alpha = (row['concentration_efficiency_acceleration'] * row['volume_entropy_efficiency'] 
                    * row['efficiency_flow_decay'])
            
        else:
            # Adaptive Bridge Factors for mixed regimes
            bridge1 = (row['entropy_efficiency_acceleration'] * row['volume_velocity'] 
                      * row['trade_size_momentum'])
            bridge2 = row['pressure_momentum'] * row['volume_concentration'] * row['directional_entropy_efficiency']
            bridge3 = (row['volatility_entropy_efficiency'] * row['order_flow_entropy_efficiency'] 
                      * (-row['concentration_efficiency_acceleration']))
            alpha = (bridge1 + bridge2 + bridge3) / 3
        
        # Final Alpha Integration
        dynamic_weight = row['volume_concentration'] / (1 + np.abs(row['entropy_efficiency_acceleration']) + epsilon)
        final_signal = alpha * dynamic_weight * row['trade_size_entropy_flow']
        
        result.iloc[i] = final_signal
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
