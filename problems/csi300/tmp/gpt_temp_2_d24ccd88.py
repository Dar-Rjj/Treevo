import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multiple market regimes and momentum patterns
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Price-Volume Regime Factor
    # Volatility Regime Classification
    data['vol_5d'] = (np.log(data['high'] / data['low'])).rolling(5).mean()  # Short-term Parkinson volatility
    data['vol_20d'] = (np.log(data['high'] / data['low'])).rolling(20).mean()  # Medium-term Parkinson volatility
    
    # Regime classification
    vol_5d_median = data['vol_5d'].rolling(50).median()
    vol_20d_median = data['vol_20d'].rolling(50).median()
    data['vol_regime'] = np.where(data['vol_5d'] > vol_5d_median * 1.2, 2, 
                                 np.where(data['vol_5d'] < vol_5d_median * 0.8, 0, 1))
    
    # Multi-timeframe Momentum Alignment
    data['mom_3d'] = data['close'].pct_change(3)
    data['mom_10d'] = data['close'].pct_change(10)
    data['mom_20d'] = data['close'].pct_change(20)
    
    # Momentum convergence/divergence
    data['mom_convergence'] = (np.sign(data['mom_3d']) == np.sign(data['mom_10d'])) & \
                             (np.sign(data['mom_10d']) == np.sign(data['mom_20d']))
    data['mom_alignment'] = data['mom_convergence'].astype(int) * \
                           (abs(data['mom_3d']) + abs(data['mom_10d']) + abs(data['mom_20d'])) / 3
    
    # Volume Regime Analysis
    data['volume_5d_trend'] = data['volume'].rolling(5).apply(lambda x: (x[-1] - x[0]) / x.mean() if x.mean() > 0 else 0)
    data['volume_20d_trend'] = data['volume'].rolling(20).apply(lambda x: (x[-1] - x[0]) / x.mean() if x.mean() > 0 else 0)
    
    volume_median = data['volume'].rolling(50).median()
    data['volume_spike'] = np.where(data['volume'] > volume_median * 1.5, 1, 0)
    data['volume_persistence'] = data['volume_spike'].rolling(5).sum()
    
    # Regime-Weighted Combined Signal
    vol_regime_weight = np.where(data['vol_regime'] == 2, 0.5, 
                                np.where(data['vol_regime'] == 0, 1.5, 1.0))
    volume_weight = 1 + data['volume_5d_trend'] * 2 + data['volume_persistence'] * 0.1
    regime_factor = data['mom_alignment'] * vol_regime_weight * volume_weight
    
    # Efficiency-Persistence Momentum Factor
    # Multi-day Range Efficiency
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_3d'] = data['daily_efficiency'].rolling(3).mean()
    data['efficiency_5d_trend'] = data['daily_efficiency'].rolling(5).apply(lambda x: (x[-1] - x[0]) / np.std(x) if np.std(x) > 0 else 0)
    
    # Efficiency Persistence Tracking
    data['high_efficiency'] = (data['daily_efficiency'] > data['daily_efficiency'].rolling(20).quantile(0.7)).astype(int)
    data['efficiency_streak'] = data['high_efficiency'].groupby((data['high_efficiency'] != data['high_efficiency'].shift()).cumsum()).cumcount() + 1
    data['efficiency_streak'] = data['efficiency_streak'] * data['high_efficiency']
    
    # Volume-Confirmed Efficiency
    volume_efficiency_corr = data['volume'].rolling(10).corr(data['daily_efficiency'])
    data['volume_confirmation'] = np.where(volume_efficiency_corr > 0, 1 + volume_efficiency_corr, 1)
    
    # Persistence-Enhanced Efficiency Factor
    efficiency_factor = data['efficiency_3d'] * (1 + data['efficiency_streak'] * 0.1) * data['volume_confirmation']
    
    # Amount Flow Regime Factor
    # Multi-timeframe Flow Analysis
    data['amount_flow'] = data['amount'] * np.sign(data['close'] - data['open'])
    data['flow_3d'] = data['amount_flow'].rolling(3).sum()
    data['flow_10d_trend'] = data['amount_flow'].rolling(10).apply(lambda x: (x[-1] - x[0]) / (abs(x).mean() + 1e-8))
    
    # Flow Persistence Regimes
    data['positive_flow'] = (data['amount_flow'] > 0).astype(int)
    data['flow_streak'] = data['positive_flow'].groupby((data['positive_flow'] != data['positive_flow'].shift()).cumsum()).cumcount() + 1
    data['flow_streak'] = data['flow_streak'] * data['positive_flow']
    
    # Price-Flow Alignment
    data['price_flow_alignment'] = (np.sign(data['close'].pct_change()) == np.sign(data['amount_flow'])).astype(int)
    data['alignment_streak'] = data['price_flow_alignment'].groupby((data['price_flow_alignment'] != data['price_flow_alignment'].shift()).cumsum()).cumcount() + 1
    
    # Regime-Aware Flow Momentum
    flow_factor = data['flow_3d'] * (1 + data['flow_streak'] * 0.05) * (1 + data['alignment_streak'] * 0.1)
    
    # Volatility-Adaptive Mean Reversion
    # Multi-timeframe Extreme Detection
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_10d'] = data['close'].pct_change(10)
    data['range_position'] = (data['close'] - data['low'].rolling(20).min()) / \
                            (data['high'].rolling(20).max() - data['low'].rolling(20).min()).replace(0, np.nan)
    
    # Extreme scoring
    data['extreme_score'] = (abs(data['ret_5d']) > data['ret_5d'].rolling(50).std() * 1.5).astype(int) + \
                           (abs(data['ret_10d']) > data['ret_10d'].rolling(50).std() * 1.5).astype(int) + \
                           ((data['range_position'] > 0.8) | (data['range_position'] < 0.2)).astype(int)
    
    # Volatility-Context Adjustment
    vol_adjustment = np.where(data['vol_regime'] == 2, 0.7, 
                             np.where(data['vol_regime'] == 0, 1.3, 1.0))
    data['adjusted_extreme'] = data['extreme_score'] * vol_adjustment
    
    # Volume Confirmation for Reversals
    data['volume_extreme'] = (data['volume'] > data['volume'].rolling(20).quantile(0.8)).astype(int)
    data['volume_confirmation_rev'] = data['volume_extreme'].rolling(3).sum()
    
    # Adaptive Mean Reversion Factor
    mean_reversion_factor = -data['adjusted_extreme'] * (1 + data['volume_confirmation_rev'] * 0.2)
    
    # Price-Range Momentum Convergence
    # Multi-timeframe Range Analysis
    data['daily_range'] = data['high'] - data['low']
    data['range_mom_3d'] = data['daily_range'].pct_change(3)
    data['range_mom_10d'] = data['daily_range'].pct_change(10)
    
    # Price-Range Alignment
    data['price_range_alignment'] = (np.sign(data['mom_3d']) == np.sign(data['range_mom_3d'])).astype(int)
    data['alignment_strength'] = abs(data['mom_3d'] * data['range_mom_3d'])
    
    # Volume-Range Relationship
    volume_range_corr = data['volume'].rolling(10).corr(data['daily_range'])
    data['volume_range_confirmation'] = np.where(volume_range_corr > 0, 1 + volume_range_corr, 1)
    
    # Convergent Momentum Factor
    convergence_factor = data['alignment_strength'] * data['price_range_alignment'] * data['volume_range_confirmation']
    
    # Final Alpha Factor Combination
    # Normalize individual factors
    factors = pd.DataFrame({
        'regime': regime_factor,
        'efficiency': efficiency_factor,
        'flow': flow_factor,
        'mean_reversion': mean_reversion_factor,
        'convergence': convergence_factor
    })
    
    # Z-score normalization
    factors_normalized = factors.apply(lambda x: (x - x.rolling(50).mean()) / x.rolling(50).std())
    
    # Weighted combination (equal weights for demonstration)
    final_alpha = (factors_normalized['regime'] * 0.25 + 
                   factors_normalized['efficiency'] * 0.25 + 
                   factors_normalized['flow'] * 0.2 + 
                   factors_normalized['mean_reversion'] * 0.15 + 
                   factors_normalized['convergence'] * 0.15)
    
    return final_alpha
