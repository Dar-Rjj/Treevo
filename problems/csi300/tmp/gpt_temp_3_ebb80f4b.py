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
    data['spread_t9'] = data['spread_t4'].rolling(window=5, min_periods=1).mean()
    data['spread_t19'] = data['spread_t4'].rolling(window=15, min_periods=1).mean()
    
    # Calculate liquidity persistence using rolling correlation
    data['liquidity_persistence'] = data['spread_t4'].rolling(window=10, min_periods=1).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Classify regimes: 1 for high liquidity, -1 for low liquidity, 0 for neutral
    data['liquidity_regime'] = 0
    data.loc[data['spread_t4'] < data['spread_t4'].rolling(window=20, min_periods=1).quantile(0.3), 'liquidity_regime'] = 1
    data.loc[data['spread_t4'] > data['spread_t4'].rolling(window=20, min_periods=1).quantile(0.7), 'liquidity_regime'] = -1
    
    # Regime-Dependent Flow Asymmetry Analysis
    # Compute Flow Bias
    data['flow_bias'] = (data['close'] - data['open']) * data['amount'] / (data['volume'] + 1e-8)
    
    # Measure Volume Efficiency
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Calculate flow bias persistence
    data['flow_bias_persistence'] = data['flow_bias'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sign(x).mean() if len(x) > 0 else 0, raw=False
    )
    
    # Generate Flow Asymmetry Signal
    data['flow_asymmetry_signal'] = 0
    # High liquidity + persistent flow bias → Momentum continuation
    high_liquidity_mask = (data['liquidity_regime'] == 1) & (np.abs(data['flow_bias_persistence']) > 0.6)
    data.loc[high_liquidity_mask, 'flow_asymmetry_signal'] = data.loc[high_liquidity_mask, 'flow_bias_persistence']
    
    # Low liquidity + conflicting flow bias → Reversal probability
    low_liquidity_mask = (data['liquidity_regime'] == -1) & (np.abs(data['flow_bias_persistence']) < 0.3)
    data.loc[low_liquidity_mask, 'flow_asymmetry_signal'] = -data.loc[low_liquidity_mask, 'flow_bias']
    
    # Fractal Momentum-Liquidity Divergence
    # Analyze momentum compression using Close price clustering
    data['close_clustering'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['momentum_compression'] = data['close_clustering'].rolling(window=5, min_periods=1).std()
    
    # Calculate Flow Concentration
    data['flow_concentration'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + 1e-8)
    
    # Calculate momentum-liquidity divergence
    data['momentum_trend'] = data['close'].pct_change(periods=3).rolling(window=5, min_periods=1).mean()
    data['liquidity_trend'] = data['spread_t4'].pct_change(periods=3).rolling(window=5, min_periods=1).mean()
    data['momentum_liquidity_divergence'] = np.sign(data['momentum_trend']) * np.sign(data['liquidity_trend'])
    
    # Generate Momentum-Liquidity Signal
    data['momentum_liquidity_signal'] = 0
    # High compression + liquidity divergence → Breakout anticipation
    high_compression_mask = (data['momentum_compression'] < data['momentum_compression'].rolling(window=20, min_periods=1).quantile(0.3))
    divergence_mask = (data['momentum_liquidity_divergence'] < 0)
    data.loc[high_compression_mask & divergence_mask, 'momentum_liquidity_signal'] = data.loc[high_compression_mask & divergence_mask, 'flow_concentration']
    
    # Low compression + aligned flow → Trend continuation
    low_compression_mask = (data['momentum_compression'] > data['momentum_compression'].rolling(window=20, min_periods=1).quantile(0.7))
    aligned_mask = (data['momentum_liquidity_divergence'] > 0) & (data['flow_bias_persistence'] > 0)
    data.loc[low_compression_mask & aligned_mask, 'momentum_liquidity_signal'] = data.loc[low_compression_mask & aligned_mask, 'momentum_trend']
    
    # Combine signals with regime-dependent weights
    data['regime_weight'] = np.where(data['liquidity_regime'] == 1, 0.6, 
                                   np.where(data['liquidity_regime'] == -1, 0.4, 0.5))
    
    # Final alpha factor combining all components
    data['alpha_factor'] = (
        data['regime_weight'] * data['flow_asymmetry_signal'] +
        (1 - data['regime_weight']) * data['momentum_liquidity_signal'] +
        data['liquidity_persistence'] * 0.2
    )
    
    # Normalize the final factor
    data['alpha_factor'] = (data['alpha_factor'] - data['alpha_factor'].rolling(window=20, min_periods=1).mean()) / (data['alpha_factor'].rolling(window=20, min_periods=1).std() + 1e-8)
    
    return data['alpha_factor']
