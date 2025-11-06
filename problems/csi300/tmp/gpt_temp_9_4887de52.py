import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Volatility-Regime Microstructure Divergence factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Single-Asset Volatility Components
    df['high_low_range'] = df['high'] - df['low']
    df['short_term_vol'] = df['high_low_range'].rolling(window=5, min_periods=3).mean()
    df['returns'] = df['close'].pct_change()
    df['medium_term_vol'] = df['returns'].rolling(window=10, min_periods=5).std()
    
    # True Range calculation
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['long_term_vol'] = df['true_range'].rolling(window=20, min_periods=10).std()
    
    # Volatility regime classification
    df['vol_regime'] = 0
    vol_median = df['short_term_vol'].rolling(window=60, min_periods=30).median()
    df.loc[df['short_term_vol'] > vol_median * 1.2, 'vol_regime'] = 1  # High volatility
    df.loc[df['short_term_vol'] < vol_median * 0.8, 'vol_regime'] = -1  # Low volatility
    
    # Microstructure Momentum Analysis
    df['intraday_order_flow'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) * df['volume']
    
    # Order flow accumulation
    for window in [3, 5, 10]:
        df[f'order_flow_accum_{window}'] = df['intraday_order_flow'].rolling(window=window, min_periods=window//2).sum()
    
    # Order flow momentum
    df['order_flow_momentum_3'] = df['order_flow_accum_3'].pct_change(periods=2)
    df['order_flow_momentum_5'] = df['order_flow_accum_5'].pct_change(periods=3)
    
    # Price Efficiency Assessment
    df['range_utilization'] = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 1e-8)
    df['price_volume_efficiency'] = (df['close'] - df['open']) / (df['amount'] / (df['volume'] + 1e-8) + 1e-8)
    
    # Cross-Asset Microstructure Divergence Detection
    df['price_momentum_3'] = df['close'].pct_change(periods=3)
    df['price_momentum_5'] = df['close'].pct_change(periods=5)
    
    # Order flow vs price divergence
    df['divergence_3'] = (df['order_flow_momentum_3'] - df['price_momentum_3']) * df['volume']
    df['divergence_5'] = (df['order_flow_momentum_5'] - df['price_momentum_5']) * df['volume']
    
    # Multi-timeframe microstructure alignment
    df['micro_alignment_3'] = np.sign(df['order_flow_momentum_3']) * np.sign(df['price_momentum_3'])
    df['micro_alignment_5'] = np.sign(df['order_flow_momentum_5']) * np.sign(df['price_momentum_5'])
    
    # Volatility-Regime Adaptive Signal Generation
    for i, (idx, row) in enumerate(df.iterrows()):
        if i < 20:  # Skip initial period for reliable calculations
            continue
            
        current_data = df.iloc[:i+1]
        current_row = row
        
        # High volatility regime processing
        if current_row['vol_regime'] == 1:
            # Amplify cross-asset divergence significance
            vol_weight = current_row['short_term_vol'] / (current_data['short_term_vol'].mean() + 1e-8)
            divergence_signal = (
                current_row['divergence_3'] * 0.6 + 
                current_row['divergence_5'] * 0.4
            ) * vol_weight
            
        # Low volatility regime processing
        elif current_row['vol_regime'] == -1:
            # Emphasize convergence patterns and persistent alignments
            alignment_strength = (
                current_row['micro_alignment_3'] * 0.4 + 
                current_row['micro_alignment_5'] * 0.6
            )
            efficiency_strength = current_row['price_volume_efficiency']
            divergence_signal = alignment_strength * efficiency_strength * current_row['volume']
            
        else:  # Normal volatility regime
            # Balanced approach
            divergence_signal = (
                current_row['divergence_3'] * 0.5 + 
                current_row['divergence_5'] * 0.5
            ) * current_row['price_volume_efficiency']
        
        # Multi-horizon signal integration
        short_term = (
            current_row['order_flow_momentum_3'] * 0.4 +
            current_row['range_utilization'] * 0.3 +
            current_row['divergence_3'] * 0.3
        )
        
        medium_term = (
            current_row['order_flow_momentum_5'] * 0.5 +
            current_row['micro_alignment_5'] * 0.3 +
            current_row['price_volume_efficiency'] * 0.2
        )
        
        # Combine signals with volatility regime weighting
        final_signal = (
            short_term * 0.4 +
            medium_term * 0.4 +
            divergence_signal * 0.2
        )
        
        # Apply volatility regime scaling
        if current_row['vol_regime'] == 1:
            final_signal *= 1.2  # Amplify in high volatility
        elif current_row['vol_regime'] == -1:
            final_signal *= 0.8  # Reduce in low volatility
            
        result.loc[idx] = final_signal
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    # Remove intermediate columns to clean up
    cols_to_drop = ['high_low_range', 'short_term_vol', 'returns', 'medium_term_vol', 
                   'true_range', 'long_term_vol', 'vol_regime', 'intraday_order_flow',
                   'order_flow_accum_3', 'order_flow_accum_5', 'order_flow_accum_10',
                   'order_flow_momentum_3', 'order_flow_momentum_5', 'range_utilization',
                   'price_volume_efficiency', 'price_momentum_3', 'price_momentum_5',
                   'divergence_3', 'divergence_5', 'micro_alignment_3', 'micro_alignment_5']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return result
