import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Liquidity Regime Identification
    # Calculate Effective Spread Proxy at different intervals
    data['spread_t4'] = (data['high'] - data['low']) / data['close']
    data['spread_t9'] = data['spread_t4'].shift(5).rolling(window=5, min_periods=3).mean()
    data['spread_t19'] = data['spread_t4'].shift(10).rolling(window=10, min_periods=5).mean()
    
    # Calculate liquidity persistence metrics
    data['liquidity_persistence_short'] = data['spread_t4'].rolling(window=5, min_periods=3).std()
    data['liquidity_persistence_medium'] = data['spread_t9'].rolling(window=5, min_periods=3).std()
    
    # Classify liquidity regimes
    data['liquidity_regime'] = 0  # 0: Normal, 1: High, -1: Low
    threshold_high = data['spread_t4'].rolling(window=20, min_periods=10).quantile(0.7)
    threshold_low = data['spread_t4'].rolling(window=20, min_periods=10).quantile(0.3)
    
    high_liquidity_condition = (data['spread_t4'] < threshold_low) & (data['liquidity_persistence_short'] < data['liquidity_persistence_short'].rolling(window=20, min_periods=10).quantile(0.4))
    low_liquidity_condition = (data['spread_t4'] > threshold_high) & (data['liquidity_persistence_short'] > data['liquidity_persistence_short'].rolling(window=20, min_periods=10).quantile(0.6))
    
    data.loc[high_liquidity_condition, 'liquidity_regime'] = 1
    data.loc[low_liquidity_condition, 'liquidity_regime'] = -1
    
    # Regime-Dependent Flow Asymmetry
    # Compute Flow Bias
    data['flow_bias'] = ((data['close'] - data['open']) * data['amount']) / (data['volume'] + 1e-8)
    
    # Calculate flow bias persistence
    data['flow_bias_persistence'] = data['flow_bias'].rolling(window=5, min_periods=3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0)
    
    # Generate Flow Asymmetry Signal
    data['flow_asymmetry_signal'] = 0.0
    
    # High liquidity + persistent flow bias → Momentum continuation
    momentum_condition = (data['liquidity_regime'] == 1) & (data['flow_bias_persistence'].abs() > 0.3)
    data.loc[momentum_condition, 'flow_asymmetry_signal'] = data['flow_bias_persistence'] * 2
    
    # Low liquidity + conflicting flow bias → Reversal probability
    reversal_condition = (data['liquidity_regime'] == -1) & (data['flow_bias_persistence'].abs() < 0.2)
    data.loc[reversal_condition, 'flow_asymmetry_signal'] = -data['flow_bias'].rolling(window=3, min_periods=2).mean() * 1.5
    
    # Momentum Compression Analysis
    # Analyze price clustering using Close price patterns
    data['close_clustering'] = data['close'].rolling(window=5, min_periods=3).std() / (data['close'].rolling(window=5, min_periods=3).mean() + 1e-8)
    
    # Calculate Flow Concentration
    data['flow_concentration'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + 1e-8)
    
    # Calculate compression metrics
    data['compression_ratio'] = data['close_clustering'] / (data['flow_concentration'].rolling(window=5, min_periods=3).mean() + 1e-8)
    
    # Generate Compression Signal
    data['compression_signal'] = 0.0
    
    # High compression + liquidity divergence → Breakout anticipation
    breakout_condition = (data['compression_ratio'] > data['compression_ratio'].rolling(window=20, min_periods=10).quantile(0.7)) & \
                        (data['liquidity_regime'].diff() != 0)
    data.loc[breakout_condition, 'compression_signal'] = data['flow_bias'].rolling(window=3, min_periods=2).mean() * 3
    
    # Low compression + aligned flow → Trend continuation
    trend_condition = (data['compression_ratio'] < data['compression_ratio'].rolling(window=20, min_periods=10).quantile(0.3)) & \
                     (data['flow_bias_persistence'].abs() > 0.4)
    data.loc[trend_condition, 'compression_signal'] = data['flow_bias_persistence'] * 1.5
    
    # Combine signals with regime weighting
    data['combined_factor'] = (
        data['flow_asymmetry_signal'] * (1 + 0.5 * data['liquidity_regime']) +
        data['compression_signal'] * (1 - 0.3 * data['liquidity_regime'])
    )
    
    # Normalize the final factor
    rolling_mean = data['combined_factor'].rolling(window=20, min_periods=10).mean()
    rolling_std = data['combined_factor'].rolling(window=20, min_periods=10).std()
    data['final_factor'] = (data['combined_factor'] - rolling_mean) / (rolling_std + 1e-8)
    
    return data['final_factor']
