import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Liquidity Momentum with Adaptive Flow Asymmetry
    """
    data = df.copy()
    
    # Volatility Assessment
    data['TR'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['ATR_10'] = data['TR'].rolling(window=10, min_periods=10).mean()
    data['ATR_median_20'] = data['ATR_10'].rolling(window=20, min_periods=20).median()
    
    # Liquidity Assessment - Effective Spread Proxy
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['spread_t4'] = data['spread_proxy'].shift(4)
    data['spread_t9'] = data['spread_proxy'].shift(9)
    data['spread_t19'] = data['spread_proxy'].shift(19)
    
    # Liquidity clustering patterns
    data['liquidity_cluster'] = (data['spread_t4'] + data['spread_t9'] + data['spread_t19']) / 3
    data['spread_median_10'] = data['spread_proxy'].rolling(window=10, min_periods=10).median()
    
    # Regime Classification
    high_vol_condition = data['ATR_10'] > (1.5 * data['ATR_median_20'])
    low_liquidity_condition = data['spread_proxy'] > data['spread_median_10']
    
    data['regime'] = 0  # Normal Volatility
    data.loc[high_vol_condition & low_liquidity_condition, 'regime'] = 1  # High Vol + Low Liq
    data.loc[high_vol_condition & ~low_liquidity_condition, 'regime'] = 2  # High Vol + High Liq
    
    # Multi-Scale Momentum Analysis
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum compression using Close price clustering
    data['close_range_3d'] = (data['close'].rolling(window=3).max() - 
                             data['close'].rolling(window=3).min()) / data['close'].rolling(window=3).mean()
    data['momentum_compression'] = data['momentum_3d'] / (data['close_range_3d'] + 1e-8)
    
    # Adaptive Flow Asymmetry Analysis
    data['flow_bias'] = (data['close'] - data['open']) * data['amount'] / (data['volume'] + 1e-8)
    data['flow_persistence'] = data['flow_bias'].rolling(window=5).apply(
        lambda x: np.sum(x > 0) / len(x) if len(x) == 5 else np.nan
    )
    
    # Volume Efficiency and Flow Concentration
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['flow_concentration'] = data['amount'] / (data['volume'] * (data['high'] - data['low']) + 1e-8)
    
    # Flow Pattern Identification
    data['flow_accumulation'] = ((data['close'] < data['open']) & 
                                (data['flow_concentration'] > data['flow_concentration'].rolling(window=10).quantile(0.7))).astype(int)
    data['flow_distribution'] = ((data['close'] > data['open']) & 
                                (data['flow_concentration'] > data['flow_concentration'].rolling(window=10).quantile(0.7))).astype(int)
    
    data['flow_imbalance'] = (data['flow_accumulation'].rolling(window=5).sum() - 
                             data['flow_distribution'].rolling(window=5).sum())
    
    # Regime-Adaptive Signal Generation
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for regime in [0, 1, 2]:
        regime_mask = data['regime'] == regime
        
        if regime == 0:  # Normal Volatility
            # Combine multi-scale momentum with flow concentration
            regime_signal = (0.4 * data['momentum_10d'] + 
                           0.3 * data['momentum_3d'] + 
                           0.3 * data['flow_concentration'])
        
        elif regime == 1:  # High Vol + Low Liq
            # Focus on short-term momentum compression and flow reversal
            regime_signal = (0.5 * data['momentum_compression'] + 
                           0.3 * -data['flow_imbalance'] + 
                           0.2 * -data['momentum_3d'])
        
        else:  # High Vol + High Liq (regime == 2)
            # Medium-term momentum with flow continuation
            regime_signal = (0.6 * data['momentum_10d'] + 
                           0.2 * data['flow_persistence'] + 
                           0.2 * data['volume_efficiency'])
        
        alpha_signal[regime_mask] = regime_signal[regime_mask]
    
    # Final composite factor with regime-based smoothing
    composite_alpha = alpha_signal.rolling(window=3).mean()
    
    return composite_alpha
