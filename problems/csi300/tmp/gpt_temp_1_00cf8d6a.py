import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Fractal Dimension
    # Daily price efficiency
    data['price_efficiency'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['price_efficiency'] = data['price_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # 5-day rolling efficiency ratio
    data['eff_ratio_5d'] = data['price_efficiency'].rolling(window=5, min_periods=3).mean()
    
    # Fractal dimension
    data['fractal_dim'] = 1 + np.log(data['eff_ratio_5d']) / np.log(2)
    
    # Volume Fractal Structure
    # Volume clustering intensity
    data['vol_median_10d'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['vol_deviation'] = (data['volume'] - data['vol_median_10d']) / data['vol_median_10d']
    
    # Consecutive same-direction volume days
    data['vol_direction'] = np.where(data['volume'] > data['volume'].shift(1), 1, -1)
    data['vol_direction_change'] = data['vol_direction'] != data['vol_direction'].shift(1)
    data['vol_consecutive_days'] = data.groupby(data['vol_direction_change'].cumsum()).cumcount() + 1
    
    # Volume autocorrelation at lag 3
    def vol_autocorr(x):
        if len(x) < 4:
            return np.nan
        return pd.Series(x).autocorr(lag=3)
    
    data['vol_autocorr_3'] = data['volume'].rolling(window=10, min_periods=4).apply(vol_autocorr, raw=False)
    
    # Hurst exponent approximation for volume
    def hurst_approx(x):
        if len(x) < 8:
            return np.nan
        lags = range(2, min(8, len(x)))
        tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['hurst_volume'] = data['volume'].rolling(window=20, min_periods=8).apply(hurst_approx, raw=True)
    
    # Regime Detection System
    # Volatility regime classification
    data['volatility_3d'] = data['close'].pct_change().rolling(window=3, min_periods=2).std()
    data['volatility_10d'] = data['close'].pct_change().rolling(window=10, min_periods=5).std()
    data['vol_ratio'] = data['volatility_3d'] / data['volatility_10d']
    
    # Volatility regime (1: high, 0: normal, -1: low)
    data['vol_regime'] = np.where(data['vol_ratio'] > 1.5, 1, 
                                 np.where(data['vol_ratio'] < 0.7, -1, 0))
    
    # Volatility momentum
    data['vol_momentum'] = data['vol_ratio'] - data['vol_ratio'].shift(3)
    
    # Liquidity regime analysis
    data['volume_to_range'] = data['volume'] / (data['high'] - data['low'])
    data['volume_to_range'] = data['volume_to_range'].replace([np.inf, -np.inf], np.nan)
    data['vt_ratio_median'] = data['volume_to_range'].rolling(window=10, min_periods=5).median()
    data['liquidity_deviation'] = data['volume_to_range'] / data['vt_ratio_median']
    
    # Market depth proxy
    data['amount_volatility'] = data['amount'].rolling(window=5, min_periods=3).std()
    data['liquidity_shock'] = np.where(data['amount_volatility'] > data['amount_volatility'].rolling(window=10).quantile(0.8), 1, 0)
    
    # Fractal-Momentum Divergence
    # Price efficiency vs momentum divergence
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['eff_trend'] = data['eff_ratio_5d'] - data['eff_ratio_5d'].shift(3)
    
    # Momentum-efficiency divergence
    data['mom_eff_divergence'] = np.sign(data['momentum_5d']) * np.sign(data['eff_trend'])
    
    # Volume fractal vs price pattern
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['consolidation'] = np.where(data['price_range'] < data['price_range'].rolling(window=10).quantile(0.3), 1, 0)
    
    # Volume clustering during consolidation
    data['vol_cluster_consolidation'] = data['vol_consecutive_days'] * data['consolidation']
    
    # Multi-Timeframe Fractal Alignment
    # Short-term (3-day) fractal analysis
    data['eff_ratio_3d'] = data['price_efficiency'].rolling(window=3, min_periods=2).mean()
    data['fractal_dim_3d'] = 1 + np.log(data['eff_ratio_3d']) / np.log(2)
    
    # Medium-term (8-day) fractal consistency
    data['eff_ratio_8d'] = data['price_efficiency'].rolling(window=8, min_periods=4).mean()
    data['fractal_dim_8d'] = 1 + np.log(data['eff_ratio_8d']) / np.log(2)
    
    # Fractal convergence
    data['fractal_convergence'] = np.abs(data['fractal_dim_3d'] - data['fractal_dim_8d'])
    
    # Regime-Adaptive Signal Generation
    # Base fractal efficiency signal
    data['base_fractal_signal'] = data['fractal_dim'] - data['fractal_dim'].rolling(window=20).mean()
    
    # Volatility-regime specific adjustments
    data['vol_regime_signal'] = np.where(data['vol_regime'] == 1, data['base_fractal_signal'] * 1.5,
                                       np.where(data['vol_regime'] == -1, data['vol_cluster_consolidation'], 
                                               data['base_fractal_signal']))
    
    # Liquidity-regime adjustments
    data['liquidity_signal'] = np.where(data['liquidity_deviation'] > 1.2, data['vol_autocorr_3'],
                                      np.where(data['liquidity_deviation'] < 0.8, data['base_fractal_signal'],
                                              data['vol_regime_signal']))
    
    # Dynamic threshold optimization
    data['fractal_std'] = data['fractal_dim'].rolling(window=20, min_periods=10).std()
    data['adaptive_threshold'] = data['fractal_std'] * 1.5
    
    # Final signal with regime validation
    data['final_signal'] = np.where(
        np.abs(data['liquidity_signal']) > data['adaptive_threshold'],
        data['liquidity_signal'] * (1 - data['fractal_convergence']),
        0
    )
    
    # Apply liquidity shock filter
    data['final_signal'] = np.where(data['liquidity_shock'] == 1, 
                                  data['final_signal'] * 0.5, 
                                  data['final_signal'])
    
    # Ensure no future data leakage
    result = data['final_signal'].copy()
    
    return result
