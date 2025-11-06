import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Volume Fractal Dimensions
    df['short_term_fractal'] = np.log(df['volume'] / df['volume'].shift(1)) * np.log(df['volume'].shift(1) / df['volume'].shift(2))
    df['medium_term_fractal'] = np.log(df['volume'] / df['volume'].shift(3)) * np.log(df['volume'].shift(3) / df['volume'].shift(6))
    df['long_term_fractal'] = np.log(df['volume'] / df['volume'].shift(5)) * np.log(df['volume'].shift(5) / df['volume'].shift(10))
    
    # Volume Scaling Patterns
    df['scaling_exponent'] = (np.log(df['volume']) - np.log(df['volume'].shift(1))) / (
        np.log(df['high'] - df['low']) - np.log(df['high'].shift(1) - df['low'].shift(1)) + 1e-6)
    
    # Multi-Scale Correlation
    df['multi_scale_corr'] = df['volume'].rolling(window=5).apply(
        lambda x: pd.Series(np.log(x)).corr(pd.Series(np.log(df.loc[x.index, 'high'] - df.loc[x.index, 'low']))) if len(x) == 5 else np.nan
    )
    
    # Volume Clustering Behavior
    df['intraday_clustering'] = df['volume'] / (df['volume'].shift(1).rolling(window=3).max() + 1e-6)
    df['interday_clustering'] = df['volume'] / (df['volume'].shift(1).rolling(window=3).min() + 1e-6)
    
    # Multi-Scale Range Patterns
    df['micro_range_ratio'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-6)
    df['meso_range_ratio'] = (df['high'] - df['low']) / (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min() + 1e-6)
    df['macro_range_ratio'] = (df['high'] - df['low']) / (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min() + 1e-6)
    
    # Fractal Efficiency Measures
    df['path_efficiency'] = np.abs(df['close'] - df['open']) / (
        np.abs(df['high'] - df['low']).rolling(window=5).mean() + 1e-6)
    df['directional_efficiency'] = (df['close'] - df['open']) / (
        df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min() + 1e-6)
    df['volatility_efficiency'] = (df['high'] - df['low']) / (df['close'].rolling(window=5).std() * np.sqrt(5) + 1e-6)
    
    # Price Scaling Properties
    df['hurst_exponent'] = np.log(df['high'] - df['low'] + 1e-6) / np.log(df['high'].shift(5) - df['low'].shift(5) + 1e-6)
    df['multi_fractal_spectrum'] = (np.log(df['high'] - df['low'] + 1e-6) - np.log(df['high'].shift(1) - df['low'].shift(1) + 1e-6)) / (
        np.log(df['volume'] + 1e-6) - np.log(df['volume'].shift(1) + 1e-6) + 1e-6)
    
    # Fractal Dimension
    df['fractal_dimension'] = np.log(np.abs(df['close'] - df['close'].shift(1)).rolling(window=5).sum() + 1e-6) / np.log(
        df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min() + 1e-6)
    
    # Fractal Regime Identification
    df['volume_regime'] = 0
    df.loc[(df['short_term_fractal'] > 0) & (df['medium_term_fractal'] > 0), 'volume_regime'] = 0.4
    df.loc[(df['short_term_fractal'] < 0) & (df['medium_term_fractal'] < 0), 'volume_regime'] = -0.3
    df.loc[np.abs(df['short_term_fractal'] - df['medium_term_fractal']) > 0.5, 'volume_regime'] = 0.25
    
    df['price_regime'] = 0
    df.loc[(df['path_efficiency'] > 0.7) & (df['directional_efficiency'] > 0.5), 'price_regime'] = 0.35
    df.loc[(df['path_efficiency'] < 0.3) & (df['directional_efficiency'] < 0.2), 'price_regime'] = -0.4
    df.loc[(np.abs(df['path_efficiency'] - 0.5) < 0.1) & (df['volatility_efficiency'] > 1.2), 'price_regime'] = 0.2
    
    # Cross-Fractal Alignment
    df['volume_price_alignment'] = df['scaling_exponent'] * df['multi_scale_corr']
    df['fractal_convergence'] = df['short_term_fractal'] * df['micro_range_ratio'] * df['path_efficiency']
    df['fractal_divergence'] = df['long_term_fractal'] * df['macro_range_ratio'] * df['volatility_efficiency']
    
    # Fractal Momentum Construction
    df['short_vol_momentum'] = df['short_term_fractal'] * df['intraday_clustering']
    df['medium_vol_momentum'] = df['medium_term_fractal'] * df['interday_clustering']
    df['multi_scale_vol_momentum'] = df['long_term_fractal'] * (df['intraday_clustering'] + df['interday_clustering']) / 2
    
    df['micro_price_momentum'] = df['micro_range_ratio'] * df['path_efficiency']
    df['meso_price_momentum'] = df['meso_range_ratio'] * df['directional_efficiency']
    df['macro_price_momentum'] = df['macro_range_ratio'] * df['volatility_efficiency']
    
    df['aligned_momentum'] = df['volume_price_alignment'] * df['fractal_convergence']
    df['divergent_momentum'] = df['scaling_exponent'] * df['fractal_divergence']
    df['hybrid_momentum'] = df['multi_fractal_spectrum'] * df['hurst_exponent']
    
    # Alpha Synthesis Engine
    df['primary_fractal'] = (df['short_vol_momentum'] + df['medium_vol_momentum'] + df['multi_scale_vol_momentum']) * (
        df['micro_price_momentum'] + df['meso_price_momentum'] + df['macro_price_momentum'])
    
    df['secondary_fractal'] = (df['aligned_momentum'] + df['divergent_momentum'] + df['hybrid_momentum']) * (
        df['volume_regime'] + df['price_regime'])
    
    df['tertiary_fractal'] = (df['scaling_exponent'] + df['multi_scale_corr']) * (
        df['hurst_exponent'] + df['multi_fractal_spectrum'] + df['fractal_dimension'])
    
    # Regime-Based Weighting
    df['volume_regime_weight'] = df['volume_regime']
    df['price_regime_weight'] = df['price_regime']
    df['cross_regime_alignment'] = df['volume_price_alignment'] + df['fractal_convergence']
    
    # Momentum Convergence
    df['bullish_fractal'] = df['aligned_momentum'] * df['primary_fractal'] * (1 + df['volume_regime_weight'])
    df['bearish_fractal'] = df['divergent_momentum'] * df['secondary_fractal'] * (1 + df['price_regime_weight'])
    df['neutral_fractal'] = df['hybrid_momentum'] * df['tertiary_fractal'] * (1 + df['cross_regime_alignment'])
    
    # Final Alpha Construction
    df['base_alpha'] = df['primary_fractal'] * df['secondary_fractal'] * df['tertiary_fractal']
    df['regime_adjustment'] = (df['bullish_fractal'] - df['bearish_fractal']) * df['neutral_fractal']
    df['final_alpha'] = df['base_alpha'] * (1 + df['regime_adjustment']) / (1 + np.abs(df['regime_adjustment']) + 1e-6)
    
    return df['final_alpha']
