import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Volume Convergence with Regime Adaptation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum Convergence Analysis
    # Stock Momentum Components
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_accel'] = data['momentum_5d'] - data['momentum_5d'].shift(5)
    
    # Volume Momentum Components
    data['volume_roc_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_accel'] = data['volume_roc_10d'] - data['volume_roc_10d'].shift(5)
    
    # Volume trend confirmation using rolling mean
    data['volume_trend'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    
    # 2. Correlation Asymmetry Detection
    # Multi-timeframe correlation analysis
    def rolling_corr_price_volume(window):
        return data['close'].rolling(window=window).corr(data['volume'])
    
    data['corr_3d'] = rolling_corr_price_volume(3)
    data['corr_10d'] = rolling_corr_price_volume(10)
    data['corr_diff'] = data['corr_3d'] - data['corr_10d']
    
    # Correlation regime identification
    data['corr_regime'] = data['corr_diff'].rolling(window=10, min_periods=5).apply(
        lambda x: 1 if (x > 0).sum() > (x < 0).sum() else -1, raw=False
    )
    
    # 3. Volatility and Liquidity Context
    # Volatility regime assessment
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    data['volatility_60d'] = data['close'].pct_change().rolling(window=60, min_periods=30).std()
    data['vol_ratio'] = data['volatility_20d'] / data['volatility_60d']
    
    # Volatility regime classification
    vol_threshold = data['vol_ratio'].rolling(window=60, min_periods=30).quantile(0.7)
    data['vol_regime'] = np.where(data['vol_ratio'] > vol_threshold, 1, 0)
    
    # Liquidity flow analysis
    data['price_impact'] = (data['high'] - data['low']) / data['amount'].replace(0, np.nan)
    data['flow_imbalance'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Abnormal liquidity detection
    data['liquidity_zscore'] = (
        data['amount'] - data['amount'].rolling(window=20, min_periods=10).mean()
    ) / data['amount'].rolling(window=20, min_periods=10).std()
    
    # 4. Signal Generation Framework
    # Convergence-divergence pattern recognition
    data['momentum_volume_alignment'] = (
        np.sign(data['momentum_5d']) * np.sign(data['volume_roc_10d'])
    )
    
    # Peer group convergence measurement using rolling percentiles
    data['momentum_rank'] = data['momentum_5d'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] > np.percentile(x[:-1], 70)) if len(x) > 1 else np.nan, raw=False
    )
    
    # Regime-adaptive signal conditioning
    # Volatility regime filtering
    vol_filter = np.where(data['vol_regime'] == 1, 0.7, 1.0)  # Reduce signal in high vol regimes
    
    # Correlation regime confirmation
    corr_weight = np.where(data['corr_regime'] == 1, 1.2, 0.8)  # Boost in positive correlation regimes
    
    # Liquidity context validation
    liquidity_weight = np.where(abs(data['liquidity_zscore']) > 2, 0.5, 1.0)
    
    # Final factor construction
    # Core momentum-volume convergence signal
    core_signal = (
        data['momentum_5d'] * 0.4 +
        data['momentum_accel'] * 0.2 +
        data['volume_roc_10d'] * 0.3 +
        data['volume_accel'] * 0.1
    )
    
    # Volume-weighted convergence signals
    volume_weight = data['volume_trend'].clip(0.5, 2.0)  # Cap extreme volume weights
    
    # Regime-scaled factor output
    factor = (
        core_signal * 
        volume_weight * 
        vol_filter * 
        corr_weight * 
        liquidity_weight * 
        data['momentum_volume_alignment']
    )
    
    # Market microstructure aligned timing - smooth with short MA
    final_factor = factor.rolling(window=3, min_periods=1).mean()
    
    return final_factor
