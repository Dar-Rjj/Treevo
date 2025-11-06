import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining regime-adaptive momentum, 
    asymmetric volume dynamics, efficiency-filtered breakouts, and liquidity-adaptive synthesis.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Volatility regime classification
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Short-term ATR (5-day)
    data['atr_5'] = data['true_range'].rolling(window=5).mean()
    
    # Medium-term return std (20-day)
    data['returns'] = data['close'].pct_change()
    data['vol_20'] = data['returns'].rolling(window=20).std()
    
    # Volatility regime (0=low, 1=medium, 2=high)
    atr_quantiles = data['atr_5'].rolling(window=60).apply(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop').iloc[-1] if len(x) == 60 else np.nan, raw=False)
    vol_quantiles = data['vol_20'].rolling(window=60).apply(lambda x: pd.qcut(x, 3, labels=False, duplicates='drop').iloc[-1] if len(x) == 60 else np.nan, raw=False)
    data['vol_regime'] = (atr_quantiles + vol_quantiles) / 2
    
    # 2. Base momentum (10-day ROC)
    data['momentum_10'] = data['close'].pct_change(10)
    
    # Regime-based momentum weighting
    # Higher weight in low volatility regimes
    data['regime_momentum'] = data['momentum_10'] * (1 - data['vol_regime'] / 2)
    
    # 3. Volume-volatility coupling
    # Volume/True Range correlation (10-day)
    data['volume_tr_corr'] = data['volume'].rolling(window=10).corr(data['true_range'])
    
    # 4. Directional asymmetry
    # Up/down volume analysis
    data['price_change'] = data['close'] - data['open']
    data['up_volume'] = np.where(data['price_change'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['price_change'] < 0, data['volume'], 0)
    
    data['up_volume_5'] = data['up_volume'].rolling(window=5).sum()
    data['down_volume_5'] = data['down_volume'].rolling(window=5).sum()
    data['volume_ratio'] = data['up_volume_5'] / (data['down_volume_5'] + 1e-8)
    
    # Price-volume divergence
    data['price_trend'] = data['close'].rolling(window=5).mean() / data['close'].rolling(window=20).mean() - 1
    data['volume_trend'] = data['volume'].rolling(window=5).mean() / data['volume'].rolling(window=20).mean() - 1
    data['pv_divergence'] = data['price_trend'] - data['volume_trend']
    
    # 5. Multi-scale efficiency (fractal dimension approximation)
    def hurst_exponent(series, window):
        """Approximate Hurst exponent as fractal dimension proxy"""
        def calc_hurst(x):
            if len(x) < window:
                return np.nan
            lags = range(2, min(10, len(x)))
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        
        return series.rolling(window=window).apply(calc_hurst, raw=True)
    
    data['fractal_short'] = hurst_exponent(data['close'], 20)
    data['fractal_medium'] = hurst_exponent(data['close'], 50)
    
    # 6. Breakout quality
    # Recent high breakout with volume confirmation
    data['high_20'] = data['high'].rolling(window=20).max()
    data['is_breakout'] = (data['close'] > data['high_20'].shift(1)).astype(int)
    data['breakout_volume_ratio'] = np.where(
        data['is_breakout'] == 1,
        data['volume'] / data['volume'].rolling(window=20).mean(),
        0
    )
    
    # Volatility expansion on breakout
    data['volatility_expansion'] = np.where(
        data['is_breakout'] == 1,
        data['true_range'] / data['true_range'].rolling(window=20).mean(),
        1
    )
    
    # 7. Liquidity regime (Volume-price elasticity)
    data['price_elasticity'] = data['volume'].rolling(window=10).corr(data['close']) * data['amount'].rolling(window=10).mean()
    
    # 8. Signal integration
    # Normalize components
    components = [
        'regime_momentum', 'volume_tr_corr', 'volume_ratio', 
        'pv_divergence', 'fractal_short', 'fractal_medium',
        'breakout_volume_ratio', 'volatility_expansion'
    ]
    
    for col in components:
        if col in data.columns:
            data[f'{col}_norm'] = (data[col] - data[col].rolling(window=60).mean()) / (data[col].rolling(window=60).std() + 1e-8)
    
    # Liquidity-adaptive weights
    liquidity_weight = 1 / (1 + abs(data['price_elasticity']))
    
    # Final factor synthesis
    data['alpha_factor'] = (
        liquidity_weight * data.get('regime_momentum_norm', 0) * 0.3 +
        data.get('volume_tr_corr_norm', 0) * 0.2 +
        data.get('volume_ratio_norm', 0) * 0.15 +
        data.get('pv_divergence_norm', 0) * 0.15 +
        (data.get('fractal_short_norm', 0) - data.get('fractal_medium_norm', 0)) * 0.1 +
        data.get('breakout_volume_ratio_norm', 0) * data.get('volatility_expansion_norm', 0) * 0.1
    )
    
    # Clean up intermediate columns
    result = data['alpha_factor'].copy()
    
    return result
