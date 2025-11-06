import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Efficiency with Regime-Switching Dynamics
    """
    data = df.copy()
    
    # Price Fractal Dimension
    data['daily_efficiency'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['fractal_dimension'] = data['daily_efficiency'].rolling(window=5, min_periods=3).mean()
    
    # Volume Fractal Structure
    data['volume_median_10'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['volume_deviation'] = data['volume'] / data['volume_median_10'] - 1
    
    # Volume autocorrelation at lag 3
    data['volume_autocorr_3'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=3) if len(x) >= 6 else np.nan, raw=False
    )
    
    # Volatility Regime Classification
    data['volatility_3d'] = data['close'].pct_change().rolling(window=3, min_periods=2).std()
    data['volatility_10d'] = data['close'].pct_change().rolling(window=10, min_periods=5).std()
    data['volatility_ratio'] = data['volatility_3d'] / data['volatility_10d']
    data['high_vol_regime'] = (data['volatility_ratio'] > 1.2).astype(int)
    
    # Liquidity Regime Analysis
    data['volume_range_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['volume_efficiency_median'] = data['volume_range_efficiency'].rolling(window=10, min_periods=5).median()
    data['high_liquidity_regime'] = (data['volume_range_efficiency'] > data['volume_efficiency_median']).astype(int)
    
    # Fractal-Momentum Divergence
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['efficiency_trend'] = data['fractal_dimension'].diff(3)
    data['momentum_efficiency_divergence'] = data['price_momentum_5d'] - data['efficiency_trend']
    
    # Volume clustering during price consolidation
    data['price_range_5d'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close'].shift(5)
    data['volume_clustering'] = data['volume_deviation'] / (data['price_range_5d'] + 1e-8)
    
    # Multi-Timeframe Fractal Alignment
    data['fractal_3d'] = data['daily_efficiency'].rolling(window=3, min_periods=2).mean()
    data['fractal_8d'] = data['daily_efficiency'].rolling(window=8, min_periods=4).mean()
    data['fractal_convergence'] = data['fractal_3d'] - data['fractal_8d']
    
    # Regime-Adaptive Signal Generation
    # Volatility-regime specific signals
    data['vol_regime_signal'] = np.where(
        data['high_vol_regime'] == 1,
        data['fractal_dimension'] * data['volume_autocorr_3'],
        data['fractal_dimension'] * (1 - data['volume_autocorr_3'])
    )
    
    # Liquidity-regime specific signals
    data['liq_regime_signal'] = np.where(
        data['high_liquidity_regime'] == 1,
        data['volume_deviation'] * data['fractal_dimension'],
        data['volume_deviation'] * (1 - data['fractal_dimension'])
    )
    
    # Combined factor with regime weighting
    data['fractal_efficiency_factor'] = (
        data['vol_regime_signal'] * 0.4 + 
        data['liq_regime_signal'] * 0.3 + 
        data['momentum_efficiency_divergence'] * 0.2 + 
        data['fractal_convergence'] * 0.1
    )
    
    # Final factor normalization
    factor = data['fractal_efficiency_factor'].copy()
    factor = (factor - factor.rolling(window=20, min_periods=10).mean()) / factor.rolling(window=20, min_periods=10).std()
    
    return factor
