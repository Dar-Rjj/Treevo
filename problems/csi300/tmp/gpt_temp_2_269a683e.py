import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Volatility Regime Momentum factor
    Combines multi-timeframe volatility analysis with regime-specific momentum enhancement
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 1. Multi-Timeframe Volatility States
    # Short-term volatility (5-day ATR)
    data['short_vol'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    
    # Medium-term volatility (20-day close returns std)
    data['returns'] = data['close'].pct_change()
    data['medium_vol'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    # Volatility regime classification
    data['vol_ratio'] = data['short_vol'] / data['medium_vol']
    data['vol_regime'] = 0  # stable
    data.loc[data['vol_ratio'] > 1.2, 'vol_regime'] = 1  # expanding
    data.loc[data['vol_ratio'] < 0.8, 'vol_regime'] = -1  # contracting
    
    # 2. Base Price Momentum
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # 3. Regime-Specific Momentum Enhancement
    # Volatility regime multiplier
    regime_multiplier = np.where(
        data['vol_regime'] == 1, 1.5,  # expanding: higher weight
        np.where(data['vol_regime'] == -1, 0.7, 1.0)  # contracting: lower weight
    )
    
    # Momentum persistence adjustment
    data['momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_autocorr'] = data['momentum_1d'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Apply enhancement
    persistence_adjustment = 1 + np.clip(data['momentum_autocorr'], -0.5, 0.5)
    enhanced_momentum = data['momentum_10d'] * regime_multiplier * persistence_adjustment
    
    # 4. Volume-Volatility Coupling
    # Volume-volatility correlation (10-day rolling)
    data['volume_vol_corr'] = data['volume'].rolling(window=10, min_periods=5).corr(data['true_range'])
    
    # Volume shocks (z-score of volume)
    data['volume_ma'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_std'] = data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_zscore'] = (data['volume'] - data['volume_ma']) / data['volume_std']
    
    # Directional volume analysis
    data['up_day'] = (data['close'] > data['prev_close']).astype(int)
    data['down_day'] = (data['close'] < data['prev_close']).astype(int)
    
    # Up-volume volatility response
    up_mask = data['up_day'] == 1
    data['up_vol_response'] = np.where(
        up_mask, 
        data['true_range'] / data['short_vol'], 
        1.0
    )
    
    # Asymmetric coupling factor
    volume_shock = np.clip(data['volume_zscore'], -3, 3)
    directional_bias = np.where(
        data['up_day'] == 1,
        data['up_vol_response'],
        1.0 / data['up_vol_response']  # inverse for down days
    )
    asymmetric_coupling = volume_shock * directional_bias / (data['short_vol'] + 1e-8)
    
    # 5. Combine factors
    # Normalize enhanced momentum
    momentum_norm = (enhanced_momentum - enhanced_momentum.rolling(window=50, min_periods=25).mean()) / \
                   (enhanced_momentum.rolling(window=50, min_periods=25).std() + 1e-8)
    
    # Normalize asymmetric coupling
    coupling_norm = (asymmetric_coupling - asymmetric_coupling.rolling(window=50, min_periods=25).mean()) / \
                   (asymmetric_coupling.rolling(window=50, min_periods=25).std() + 1e-8)
    
    # Final factor combination
    final_factor = 0.6 * momentum_norm + 0.4 * coupling_norm
    
    # Clean up intermediate columns
    result = pd.Series(final_factor, index=data.index)
    
    return result
