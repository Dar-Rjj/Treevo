import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Liquidity-Adjusted Momentum
    
    # Compute Bid-Ask Spread Impact
    # Estimate spread using daily high-low range
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['spread_proxy'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    
    # Calculate raw price momentum
    data['momentum_5d'] = data['close'].pct_change(periods=5)
    data['momentum_10d'] = data['close'].pct_change(periods=10)
    
    # Adjust momentum for liquidity cost
    data['liquidity_adj_momentum_5d'] = data['momentum_5d'] / (1 + data['spread_proxy'])
    data['liquidity_adj_momentum_10d'] = data['momentum_10d'] / (1 + data['spread_proxy'])
    
    # Compute Volume-Price Efficiency
    data['daily_return'] = data['close'].pct_change()
    data['efficiency_ratio'] = abs(data['daily_return']) / (data['volume'] + 1e-8)
    data['avg_efficiency'] = data['efficiency_ratio'].rolling(window=5, min_periods=3).mean()
    
    # Identify efficiency anomalies
    data['efficiency_median_20d'] = data['efficiency_ratio'].rolling(window=20, min_periods=10).median()
    data['efficiency_anomaly'] = data['avg_efficiency'] / (data['efficiency_median_20d'] + 1e-8)
    
    # Assess Volatility Regime Characteristics
    
    # Calculate Multi-Timeframe Volatility
    data['returns'] = data['close'].pct_change()
    data['vol_5d'] = data['returns'].rolling(window=5, min_periods=3).std() * np.sqrt(252)
    data['vol_20d'] = data['returns'].rolling(window=20, min_periods=10).std() * np.sqrt(252)
    
    # Classify Volatility Regime
    data['vol_ratio'] = data['vol_5d'] / (data['vol_20d'] + 1e-8)
    
    # Detect regime transitions
    data['vol_regime'] = np.where(data['vol_ratio'] > 1.2, 'high', 
                                 np.where(data['vol_ratio'] < 0.8, 'low', 'normal'))
    
    # Flag recent regime shifts
    data['regime_shift'] = data['vol_regime'] != data['vol_regime'].shift(1)
    data['recent_shift'] = data['regime_shift'].rolling(window=3, min_periods=1).max()
    
    # Generate Adaptive Convergence Signal
    
    # Combine Liquidity and Momentum Components
    # Weight components by current efficiency
    efficiency_weight = np.where(data['efficiency_anomaly'] > 1, 
                                np.minimum(data['efficiency_anomaly'], 2), 
                                np.maximum(data['efficiency_anomaly'], 0.5))
    
    # Adjust for spread impact
    spread_adjustment = 1 / (1 + data['spread_proxy'])
    
    # Base momentum signal
    base_signal = (efficiency_weight * data['liquidity_adj_momentum_5d'] + 
                   (2 - efficiency_weight) * data['liquidity_adj_momentum_10d']) / 2
    
    # Apply spread adjustment
    adjusted_signal = base_signal * spread_adjustment
    
    # Apply Volatility-Regime Filtering
    def apply_regime_filter(row):
        if row['vol_regime'] == 'high':
            # High volatility regime: increase sensitivity to momentum breaks
            momentum_strength = abs(row['liquidity_adj_momentum_5d'])
            efficiency_weight_adj = 0.7  # Reduce weight on volume efficiency
            regime_multiplier = 1.3 * momentum_strength
            
        elif row['vol_regime'] == 'low':
            # Low volatility regime: emphasize volume confirmation
            efficiency_weight_adj = 1.5  # Increase weight on volume efficiency
            momentum_threshold = 0.02  # Require stronger momentum
            momentum_strength = max(abs(row['liquidity_adj_momentum_5d']), momentum_threshold)
            regime_multiplier = 0.8 * efficiency_weight_adj
            
        else:
            # Normal regime
            regime_multiplier = 1.0
            efficiency_weight_adj = 1.0
            
        return adjusted_signal[row.name] * regime_multiplier * efficiency_weight_adj
    
    # Calculate final factor values
    factor_values = pd.Series(index=data.index, dtype=float)
    for idx in data.index:
        factor_values[idx] = apply_regime_filter(data.loc[idx])
    
    # Handle any remaining NaN values
    factor_values = factor_values.fillna(0)
    
    return factor_values
