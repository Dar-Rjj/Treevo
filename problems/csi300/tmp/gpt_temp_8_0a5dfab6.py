import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining fractal efficiency, momentum regime transitions, 
    cross-timeframe pressure imbalance, volatility clustering, and price-volume fractal dimensions
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Fractal Efficiency Divergence
    # Multi-scale efficiency
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # 5-day and 20-day fractal efficiency windows
    data['eff_5d'] = data['intraday_efficiency'].rolling(window=5, min_periods=3).mean()
    data['eff_20d'] = data['intraday_efficiency'].rolling(window=20, min_periods=10).mean()
    
    # Efficiency divergence
    data['eff_divergence'] = data['eff_5d'] - data['eff_20d']
    
    # Volume asymmetry during divergence formation (5-day window)
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_asymmetry'] = (data['volume_ma_5'] - data['volume_ma_20']) / (data['volume_ma_20'] + 1e-8)
    
    # Weighted fractal efficiency divergence
    data['fractal_eff_factor'] = data['eff_divergence'] * (1 + np.tanh(data['volume_asymmetry']))
    
    # 2. Momentum Regime Transition
    # Price range expansion/contraction
    data['daily_range'] = data['high'] - data['low']
    data['range_ma_5'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['range_ma_20'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    data['range_expansion'] = data['range_ma_5'] / (data['range_ma_20'] + 1e-8) - 1
    
    # Volume clustering patterns (z-score of volume)
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / (data['volume'].rolling(window=20, min_periods=10).std() + 1e-8)
    data['volume_clustering'] = data['volume_zscore'].rolling(window=5, min_periods=3).std()
    
    # Gap persistence (overnight gaps)
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['gap_persistence'] = data['overnight_gap'].rolling(window=3, min_periods=2).std()
    
    # Momentum regime transition factor
    data['momentum_transition'] = (np.tanh(data['range_expansion']) * 
                                 (1 + data['volume_clustering']) * 
                                 (1 + data['gap_persistence']))
    
    # 3. Cross-Timeframe Pressure Imbalance
    # Intraday pressure (current day)
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Daily pressure (1-day momentum)
    data['daily_pressure'] = data['close'] / data['close'].shift(1) - 1
    
    # Weekly pressure (5-day momentum)
    data['weekly_pressure'] = data['close'] / data['close'].shift(5) - 1
    
    # Pressure differentials
    data['pressure_diff_daily_intra'] = data['daily_pressure'] - data['intraday_pressure']
    data['pressure_diff_weekly_daily'] = data['weekly_pressure'] - data['daily_pressure']
    
    # Volume confirmation across timeframes
    data['volume_confirm_intra'] = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_confirm_weekly'] = data['volume'].rolling(window=5, min_periods=3).mean() / data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Cross-timeframe pressure factor
    data['cross_timeframe_pressure'] = (data['pressure_diff_daily_intra'] * data['volume_confirm_intra'] + 
                                      data['pressure_diff_weekly_daily'] * data['volume_confirm_weekly'])
    
    # 4. Volatility Clustering Efficiency
    # Volatility clusters using range volatility
    data['range_volatility'] = data['daily_range'].rolling(window=10, min_periods=5).std()
    data['volatility_regime'] = data['range_volatility'] > data['range_volatility'].rolling(window=20, min_periods=10).mean()
    
    # Efficiency persistence during volatility clusters
    high_vol_efficiency = data['intraday_efficiency'][data['volatility_regime']].rolling(window=5, min_periods=3).mean()
    low_vol_efficiency = data['intraday_efficiency'][~data['volatility_regime']].rolling(window=5, min_periods=3).mean()
    
    # Map back to original index
    data['high_vol_efficiency'] = high_vol_efficiency.reindex(data.index).fillna(method='ffill')
    data['low_vol_efficiency'] = low_vol_efficiency.reindex(data.index).fillna(method='ffill')
    
    # Efficiency decay rate (difference between regimes)
    data['efficiency_decay'] = data['high_vol_efficiency'] - data['low_vol_efficiency']
    
    # Volume concentration during regime boundaries
    regime_changes = data['volatility_regime'].astype(int).diff().abs()
    volume_during_changes = data['volume'] * regime_changes
    data['volume_concentration'] = volume_during_changes.rolling(window=10, min_periods=5).mean()
    
    # Volatility clustering factor
    data['vol_clustering_factor'] = data['efficiency_decay'] * (1 + np.tanh(data['volume_concentration'] / (data['volume_concentration'].mean() + 1e-8)))
    
    # 5. Price-Volume Fractal Dimension
    # Price fractal dimension using range data (Hurst-like exponent)
    def hurst_exponent(series, max_lag=20):
        lags = range(2, min(max_lag, len(series)))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Calculate rolling Hurst exponent for price ranges
    data['price_fractal'] = data['daily_range'].rolling(window=30, min_periods=15).apply(
        lambda x: hurst_exponent(x.values) if len(x) >= 15 else np.nan, raw=False
    )
    
    # Volume distribution fractal characteristics
    data['volume_fractal'] = data['volume'].rolling(window=30, min_periods=15).apply(
        lambda x: hurst_exponent(x.values) if len(x) >= 15 else np.nan, raw=False
    )
    
    # Fractal dimension divergence
    data['fractal_divergence'] = data['price_fractal'] - data['volume_fractal']
    
    # Regime shifts via fractal dimension convergence (rolling correlation)
    data['fractal_correlation'] = data['price_fractal'].rolling(window=10, min_periods=5).corr(data['volume_fractal'])
    
    # Price-volume fractal factor
    data['pv_fractal_factor'] = data['fractal_divergence'] * (1 - data['fractal_correlation'])
    
    # Combine all factors with equal weighting
    factors = ['fractal_eff_factor', 'momentum_transition', 'cross_timeframe_pressure', 
               'vol_clustering_factor', 'pv_fractal_factor']
    
    # Normalize each factor by its rolling z-score
    for factor in factors:
        if factor in data.columns:
            mean = data[factor].rolling(window=20, min_periods=10).mean()
            std = data[factor].rolling(window=20, min_periods=10).std()
            data[f'{factor}_norm'] = (data[factor] - mean) / (std + 1e-8)
    
    # Final combined alpha factor
    normalized_factors = [f'{factor}_norm' for factor in factors if f'{factor}_norm' in data.columns]
    data['alpha_factor'] = data[normalized_factors].mean(axis=1)
    
    return data['alpha_factor']
