import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume-Range Fractal Alpha Factor
    """
    data = df.copy()
    
    # Helper function for fractal dimension estimation using Hurst exponent
    def hurst_exponent(series, window):
        """Calculate Hurst exponent as proxy for fractal dimension"""
        if len(series) < window:
            return np.nan
        
        lags = range(2, min(20, window//2))
        tau = []
        for lag in lags:
            if lag >= len(series):
                break
            # RS method for Hurst exponent
            series_lag = series.diff(lag).dropna()
            if len(series_lag) == 0:
                continue
            R = series_lag.max() - series_lag.min()
            S = series_lag.std()
            if S > 0:
                tau.append(np.log(R/S))
        
        if len(tau) < 2:
            return 0.5
        
        # Linear regression to get Hurst exponent
        lags_used = list(range(2, 2+len(tau)))
        hurst = np.polyfit(np.log(lags_used), tau, 1)[0]
        return hurst
    
    # Fractal Price Structure Analysis
    data['fractal_close_5d'] = data['close'].rolling(window=5).apply(
        lambda x: hurst_exponent(x, 5), raw=False
    )
    
    data['range'] = data['high'] - data['low']
    data['fractal_range_10d'] = data['range'].rolling(window=10).apply(
        lambda x: hurst_exponent(x, 10), raw=False
    )
    
    # Volume-price relationship fractal
    data['volume_price'] = data['volume'] * data['close']
    data['fractal_vol_price_20d'] = data['volume_price'].rolling(window=20).apply(
        lambda x: hurst_exponent(x, 20), raw=False
    )
    
    # Asymmetric Range Expansion
    data['upside_expansion'] = (data['high'] - data['open']) / (data['range'] + 1e-8)
    data['downside_expansion'] = (data['open'] - data['low']) / (data['range'] + 1e-8)
    data['range_asymmetry'] = data['upside_expansion'] - data['downside_expansion']
    
    # Volume-Price Fractal Divergence
    def rolling_fractal_correlation(series1, series2, window):
        """Calculate rolling correlation between fractal dimensions"""
        corrs = []
        for i in range(len(series1)):
            if i < window:
                corrs.append(np.nan)
                continue
            window_data1 = series1.iloc[i-window:i]
            window_data2 = series2.iloc[i-window:i]
            valid_mask = (~window_data1.isna()) & (~window_data2.isna())
            if valid_mask.sum() >= 3:
                corr = window_data1[valid_mask].corr(window_data2[valid_mask])
                corrs.append(corr)
            else:
                corrs.append(np.nan)
        return pd.Series(corrs, index=series1.index)
    
    # 3-day rolling fractal correlations
    data['vol_price_fractal_corr_3d'] = rolling_fractal_correlation(
        data['fractal_close_5d'], data['fractal_vol_price_20d'], 3
    )
    
    # 8-day rolling fractal correlations
    data['vol_range_fractal_corr_8d'] = rolling_fractal_correlation(
        data['fractal_range_10d'], data['fractal_vol_price_20d'], 8
    )
    
    data['cross_fractal_momentum'] = data['vol_price_fractal_corr_3d'] - data['vol_range_fractal_corr_8d']
    
    # Multi-Timeframe Fractal Regime Detection
    data['fractal_regime'] = np.where(
        data['fractal_close_5d'] > data['fractal_close_5d'].rolling(10).mean(), 
        'high', 
        np.where(data['fractal_close_5d'] < data['fractal_close_5d'].rolling(10).quantile(0.3),
                'low', 'transition')
    )
    
    # Price-Range Efficiency Fractals
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['range'] + 1e-8)
    data['fractal_efficiency_5d'] = data['intraday_efficiency'].rolling(window=5).apply(
        lambda x: hurst_exponent(x, 5), raw=False
    )
    
    # Multi-day range expansion fractal
    data['range_expansion_3d'] = data['range'] / data['range'].rolling(3).mean()
    data['fractal_range_exp_5d'] = data['range_expansion_3d'].rolling(window=5).apply(
        lambda x: hurst_exponent(x, 5), raw=False
    )
    
    # Efficiency-asymmetry correlation
    data['eff_asym_corr_5d'] = data['intraday_efficiency'].rolling(5).corr(data['range_asymmetry'])
    
    # Fractal-Adaptive Signal Generation
    def generate_fractal_signal(row):
        if pd.isna(row['fractal_regime']):
            return 0
        
        if row['fractal_regime'] == 'high':
            # Complex patterns: Range expansion persistence + Volume-price alignment
            range_persistence = row['range'] / row['range'].rolling(3).mean().iloc[-1] if not pd.isna(row['range']) else 1
            vol_price_alignment = row['vol_price_fractal_corr_3d'] if not pd.isna(row['vol_price_fractal_corr_3d']) else 0
            signal = range_persistence * vol_price_alignment * row['range_asymmetry']
            
        elif row['fractal_regime'] == 'low':
            # Simple trends: Pure range asymmetry + Efficiency momentum
            eff_momentum = row['fractal_efficiency_5d'] if not pd.isna(row['fractal_efficiency_5d']) else 0.5
            signal = row['range_asymmetry'] * eff_momentum
            
        else:  # transition
            # Cross-fractal divergence + Multi-timeframe convergence
            cross_divergence = row['cross_fractal_momentum'] if not pd.isna(row['cross_fractal_momentum']) else 0
            multi_timeframe_convergence = (
                (row['fractal_close_5d'] if not pd.isna(row['fractal_close_5d']) else 0.5) +
                (row['fractal_range_10d'] if not pd.isna(row['fractal_range_10d']) else 0.5) +
                (row['fractal_vol_price_20d'] if not pd.isna(row['fractal_vol_price_20d']) else 0.5)
            ) / 3
            signal = cross_divergence * multi_timeframe_convergence
        
        return signal
    
    # Generate final factor values
    factor_values = []
    for idx, row in data.iterrows():
        signal = generate_fractal_signal(row)
        factor_values.append(signal)
    
    factor_series = pd.Series(factor_values, index=data.index)
    
    return factor_series
