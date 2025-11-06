import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Cross-Asset Microstructure Divergence Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Primary Asset Volatility Measurement
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['parkinson_vol'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    data['tr_median_10d'] = data['true_range'].rolling(window=10, min_periods=5).median()
    
    # Volatility regime classification
    vol_threshold = data['tr_median_10d'].quantile(0.6)
    data['high_vol_regime'] = (data['true_range'] > vol_threshold).astype(int)
    
    # Microstructure Efficiency Metrics
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['range_utilization'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['effective_spread'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / ((data['high'] + data['low'])/2 + 1e-8)
    
    # Volume & Amount Efficiency
    data['volume_per_unit'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['amount_per_share'] = data['amount'] / (data['volume'] + 1e-8)
    
    # Momentum Analysis
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5).rolling(window=5, min_periods=3).mean() - 1
    
    # Bid-Ask Bounce Detection
    data['close_ret'] = data['close'].pct_change()
    data['reversal_pattern'] = ((data['close_ret'] * data['close_ret'].shift(1)) < 0).astype(int)
    data['bounce_probability'] = data['reversal_pattern'].rolling(window=5, min_periods=3).mean()
    
    # Cross-Asset Proxy (using sector/industry patterns from primary asset)
    # For single asset implementation, we use internal momentum divergence
    data['momentum_divergence'] = data['momentum_5d'] - data['momentum_10d']
    
    # Volatility transmission proxy (using internal volatility patterns)
    data['vol_transmission'] = data['parkinson_vol'].rolling(window=5, min_periods=3).std() / (data['parkinson_vol'] + 1e-8)
    transmission_threshold = data['vol_transmission'].quantile(0.7)
    data['strong_transmission'] = (data['vol_transmission'] > transmission_threshold).astype(int)
    
    # Regime classification
    data['regime'] = 0  # Default: Noise-dominated
    data.loc[(data['high_vol_regime'] == 1) & (data['strong_transmission'] == 1), 'regime'] = 1  # Momentum
    data.loc[(data['high_vol_regime'] == 1) & (data['strong_transmission'] == 0), 'regime'] = 2  # Isolated volatility
    data.loc[(data['high_vol_regime'] == 0) & (data['strong_transmission'] == 1), 'regime'] = 3  # Convergence
    
    # Core Divergence Signal
    data['core_divergence'] = data['momentum_divergence'] * data['intraday_strength']
    data['vol_weighted_divergence'] = data['core_divergence'] * (1 + data['vol_transmission'])
    
    # Microstructure noise adjustment
    data['noise_adjustment'] = 1 - (data['bounce_probability'] * data['effective_spread'])
    data['filtered_divergence'] = data['vol_weighted_divergence'] * data['noise_adjustment']
    
    # Efficiency & Liquidity Enhancement
    volume_efficiency = data['volume_per_unit'].rolling(window=10, min_periods=5).apply(
        lambda x: x.rank(pct=True).iloc[-1] if len(x) >= 5 else 0.5
    )
    data['volume_efficiency_weight'] = volume_efficiency
    
    liquidity_quality = data['amount_per_share'].rolling(window=10, min_periods=5).apply(
        lambda x: x.rank(pct=True).iloc[-1] if len(x) >= 5 else 0.5
    )
    data['liquidity_adjustment'] = liquidity_quality
    
    # Enhanced divergence with efficiency weighting
    data['efficiency_enhanced_div'] = (data['filtered_divergence'] * 
                                     data['volume_efficiency_weight'] * 
                                     data['liquidity_adjustment'])
    
    # Session-specific patterns (using time-based proxies)
    # Morning session proxy: first 2 hours efficiency
    data['open_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min() + 1e-8)
    
    # Afternoon session proxy: last 2 hours momentum
    data['close_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Session persistence
    data['session_persistence'] = data['open_efficiency'].rolling(window=3, min_periods=2).std()
    
    # Regime-Adaptive Scaling
    regime_scaling = np.ones(len(data))
    
    # High Vol + Strong Transmission: Scale by volatility magnitude
    mask1 = data['regime'] == 1
    regime_scaling[mask1] = 1 + data.loc[mask1, 'parkinson_vol'] / data.loc[mask1, 'parkinson_vol'].mean()
    
    # Low Vol + Strong Transmission: Scale by efficiency metrics
    mask2 = data['regime'] == 3
    regime_scaling[mask2] = 1 + data.loc[mask2, 'range_utilization'] / data.loc[mask2, 'range_utilization'].mean()
    
    # High Vol + Weak Transmission: Conservative scaling
    mask3 = data['regime'] == 2
    regime_scaling[mask3] = 0.7
    
    # Low Vol + Weak Transmission: Noise-dominated scaling
    mask4 = data['regime'] == 0
    regime_scaling[mask4] = 0.5 + 0.3 * data.loc[mask4, 'volume_efficiency_weight']
    
    # Final composite factor
    data['composite_factor'] = (data['efficiency_enhanced_div'] * 
                              regime_scaling * 
                              (1 - 0.3 * data['session_persistence']))
    
    # Cross-asset confirmation proxy (using internal consistency)
    data['cross_asset_confirmation'] = (data['momentum_5d'].rolling(window=5, min_periods=3).corr(data['volume_momentum_5d']) + 1) / 2
    
    # Final alpha output with cross-asset confirmation
    alpha = data['composite_factor'] * data['cross_asset_confirmation']
    
    # Normalize the final output
    alpha_normalized = (alpha - alpha.rolling(window=20, min_periods=10).mean()) / (alpha.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return alpha_normalized
