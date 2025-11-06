import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Adaptive Momentum
    # Calculate true range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling volatility (10-day)
    df['volatility'] = df['true_range'].rolling(window=10, min_periods=5).std()
    
    # Classify volatility regime
    volatility_median = df['volatility'].rolling(window=30, min_periods=15).median()
    df['high_vol_regime'] = (df['volatility'] > volatility_median).astype(int)
    
    # Regime-specific momentum
    df['momentum_short'] = df['close'] / df['close'].shift(3) - 1  # 3-day momentum
    df['momentum_medium'] = df['close'] / df['close'].shift(8) - 1  # 8-day momentum
    
    # Adaptive momentum based on regime
    df['adaptive_momentum'] = np.where(
        df['high_vol_regime'] == 1,
        df['momentum_short'],
        df['momentum_medium']
    )
    
    # Volume-Weighted Price Fractality
    # Calculate price range fractal dimension approximation
    df['hl_range'] = df['high'] - df['low']
    df['oc_range'] = abs(df['open'] - df['close'])
    df['range_ratio'] = df['hl_range'] / (df['oc_range'] + 1e-8)
    
    # Fractal complexity measure (simplified)
    df['fractal_measure'] = np.log1p(df['range_ratio']) * np.log1p(df['hl_range'])
    
    # Volume-weighted fractal
    df['volume_weighted_fractal'] = df['fractal_measure'] * np.log1p(df['volume'])
    
    # Opening Auction Imbalance Persistence
    # Opening range efficiency (simulated with daily data)
    df['opening_range'] = df['high'].rolling(window=5, min_periods=3).apply(
        lambda x: (x[-1] - x[0]) / (x.max() - x.min() + 1e-8)
    )
    
    # Persistence measure
    df['opening_move'] = (df['close'] - df['open']) / df['open']
    df['persistence_strength'] = df['opening_move'] * df['opening_range']
    
    # Pressure Differential Accumulation
    # Pressure proxy
    df['pressure_diff'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-8)
    
    # Accumulate pressure imbalances
    df['pressure_sign'] = np.where(df['pressure_diff'] > 0, 1, -1)
    df['pressure_accumulation'] = df['pressure_sign'].rolling(window=5, min_periods=3).sum()
    
    # Volume-Volatility Confluence Divergence
    df['daily_range'] = df['high'] - df['low']
    df['volume_vol_ratio'] = df['volume'] / (df['daily_range'] + 1e-8)
    
    # Normal ratio (20-day median)
    normal_ratio = df['volume_vol_ratio'].rolling(window=20, min_periods=10).median()
    df['volume_vol_divergence'] = df['volume_vol_ratio'] / (normal_ratio + 1e-8)
    
    # Multi-Scale Trend Congruence
    # Micro-trend (intraday)
    df['micro_trend'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) * 100
    
    # Macro-trend (5-day)
    df['macro_trend'] = (df['close'] / df['close'].shift(5) - 1) * 100
    
    # Trend congruence
    df['trend_congruence'] = np.sign(df['micro_trend']) * np.sign(df['macro_trend']) * np.sqrt(abs(df['micro_trend'] * df['macro_trend']))
    
    # Efficiency-Weighted Momentum
    # Price movement efficiency
    df['efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Efficiency-weighted momentum (5-day)
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['efficiency_weighted_momentum'] = df['momentum_5d'] * df['efficiency']
    
    # Volume-Anchor Price Discovery
    # Volume anchors (price levels with high volume concentration)
    df['volume_price_ratio'] = df['volume'] / (abs(df['close'] - df['close'].shift(1)) + 1e-8)
    
    # Price discovery from volume anchors (5-day rolling)
    df['price_discovery'] = (df['close'] - df['close'].shift(5)) * np.log1p(df['volume_price_ratio'].rolling(window=5, min_periods=3).mean())
    
    # Combine all factors with weights
    factors = [
        'adaptive_momentum',
        'volume_weighted_fractal', 
        'persistence_strength',
        'pressure_accumulation',
        'volume_vol_divergence',
        'trend_congruence',
        'efficiency_weighted_momentum',
        'price_discovery'
    ]
    
    # Normalize each factor by its rolling z-score
    final_factor = pd.Series(0, index=df.index)
    for factor in factors:
        if factor in df.columns:
            # Remove outliers and normalize
            factor_series = df[factor].copy()
            rolling_mean = factor_series.rolling(window=20, min_periods=10).mean()
            rolling_std = factor_series.rolling(window=20, min_periods=10).std()
            normalized_factor = (factor_series - rolling_mean) / (rolling_std + 1e-8)
            normalized_factor = np.clip(normalized_factor, -3, 3)  # Winsorize
            final_factor += normalized_factor
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'tr1', 'tr2', 'tr3', 'true_range', 'volatility', 
                   'high_vol_regime', 'momentum_short', 'momentum_medium', 'hl_range',
                   'oc_range', 'range_ratio', 'fractal_measure', 'opening_range',
                   'opening_move', 'pressure_diff', 'pressure_sign', 'daily_range',
                   'volume_vol_ratio', 'micro_trend', 'macro_trend', 'efficiency',
                   'momentum_5d', 'volume_price_ratio']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return final_factor
