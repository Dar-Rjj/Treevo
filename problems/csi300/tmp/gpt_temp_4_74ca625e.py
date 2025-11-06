import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Multi-Asset Relative Alpha factor combining:
    - Cross-sectional relative strength with industry momentum
    - Temporal pattern asymmetry analysis
    - Order flow imbalance dynamics
    - Adaptive signal integration
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Cross-Sectional Relative Strength Framework
    # Industry-relative price momentum (using rolling percentiles)
    data['ret_5d'] = data['close'].pct_change(5)
    data['ret_10d'] = data['close'].pct_change(10)
    
    # Rolling industry percentile ranking (simulated with rolling window)
    window_industry = 20
    data['mom_rank_5d'] = data['ret_5d'].rolling(window_industry).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == window_industry else np.nan, 
        raw=False
    )
    data['mom_rank_10d'] = data['ret_10d'].rolling(window_industry).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == window_industry else np.nan, 
        raw=False
    )
    
    # Relative strength persistence scoring
    data['mom_persistence'] = (data['mom_rank_5d'] > 0.6).astype(int) + (data['mom_rank_10d'] > 0.6).astype(int)
    
    # Volume-adjusted relative performance
    data['volume_ma'] = data['volume'].rolling(10).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma']
    data['vol_adj_ret'] = data['ret_5d'] * np.log1p(data['volume_ratio'])
    
    # Risk-adjusted relative measures
    data['volatility_20d'] = data['close'].pct_change().rolling(20).std()
    data['risk_adj_mom'] = data['ret_10d'] / (data['volatility_20d'] + 1e-8)
    
    # 2. Temporal Pattern Asymmetry Analysis
    # Intraday vs overnight return divergences
    data['overnight_ret'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_ret'] = (data['close'] - data['open']) / data['open']
    data['divergence_signal'] = np.sign(data['overnight_ret']) * np.sign(data['intraday_ret'])
    
    # Multi-frequency price path complexity
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['close_to_close_vol'] = data['close'].pct_change().abs()
    data['range_efficiency'] = data['close_to_close_vol'] / (data['high_low_range'] + 1e-8)
    
    # Time-of-day effect interactions (using intraday patterns)
    data['morning_strength'] = (data['high'].rolling(5).max() - data['open']) / data['open']
    data['afternoon_reversal'] = (data['close'] - data['open'].rolling(5).mean()) / data['open'].rolling(5).mean()
    
    # 3. Order Flow Imbalance Dynamics
    # Volume clustering microstructure
    data['large_trade_threshold'] = data['volume'].rolling(20).quantile(0.8)
    data['large_trade_ratio'] = (data['volume'] > data['large_trade_threshold']).astype(int)
    data['volume_clustering'] = data['large_trade_ratio'].rolling(5).sum()
    
    # Price impact efficiency
    data['vwap'] = data['amount'] / (data['volume'] + 1e-8)
    data['price_impact'] = (data['close'] - data['vwap']) / data['vwap']
    data['execution_quality'] = -data['price_impact'].abs()  # Lower impact is better
    
    # 4. Adaptive Signal Integration
    # Volatility regime detection
    data['vol_regime'] = (data['volatility_20d'] > data['volatility_20d'].rolling(50).mean()).astype(int)
    
    # Dynamic signal combination with regime-dependent weights
    # High volatility regime: emphasize risk-adjusted measures
    # Low volatility regime: emphasize momentum and volume signals
    
    regime_weight_momentum = np.where(data['vol_regime'] == 1, 0.3, 0.5)
    regime_weight_volume = np.where(data['vol_regime'] == 1, 0.2, 0.3)
    regime_weight_risk_adj = np.where(data['vol_regime'] == 1, 0.4, 0.1)
    regime_weight_execution = np.where(data['vol_regime'] == 1, 0.1, 0.1)
    
    # Normalize components
    data['norm_mom'] = (data['mom_rank_5d'] + data['mom_rank_10d']) / 2
    data['norm_volume'] = data['vol_adj_ret'].rolling(10).mean()
    data['norm_risk_adj'] = data['risk_adj_mom'].rolling(10).mean()
    data['norm_execution'] = data['execution_quality'].rolling(10).mean()
    
    # Final alpha factor with dynamic weighting
    alpha = (
        regime_weight_momentum * data['norm_mom'] +
        regime_weight_volume * data['norm_volume'] +
        regime_weight_risk_adj * data['norm_risk_adj'] +
        regime_weight_execution * data['norm_execution']
    )
    
    # Add temporal pattern signals as moderators
    temporal_moderator = (
        0.3 * data['divergence_signal'].rolling(5).mean() +
        0.4 * (1 - data['range_efficiency'].rolling(10).mean()) +  # Inverse of inefficiency
        0.3 * data['afternoon_reversal'].rolling(5).mean()
    )
    
    final_alpha = alpha * (1 + 0.2 * temporal_moderator)
    
    return final_alpha
