import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Fractal Momentum Divergence factor
    Combines fractal analysis of volume and price patterns with momentum divergence detection
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Fractal Structure Analysis
    # Multi-timeframe volume clustering patterns
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_clustering'] = data['volume_ma_5'] / data['volume_ma_20']
    
    # Price movement self-similarity across scales
    data['price_range_5'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close'].rolling(window=5).mean()
    data['price_range_20'] = (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()) / data['close'].rolling(window=20).mean()
    data['price_fractal_similarity'] = data['price_range_5'] / data['price_range_20']
    
    # Volume distribution fractal dimension approximation
    data['volume_std_5'] = data['volume'].rolling(window=5, min_periods=3).std()
    data['volume_std_20'] = data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_fractal_dim'] = np.log(data['volume_std_5'] / data['volume_std_20']) / np.log(5/20)
    
    # Momentum Divergence
    # Volume momentum vs price momentum divergence
    data['volume_momentum'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    data['price_momentum'] = data['close'] / data['close'].rolling(window=10, min_periods=5).mean()
    data['momentum_divergence'] = data['volume_momentum'] - data['price_momentum']
    
    # Acceleration/deceleration phase identification
    data['volume_accel'] = data['volume_momentum'].diff(3)
    data['price_accel'] = data['price_momentum'].diff(3)
    data['accel_divergence'] = data['volume_accel'] - data['price_accel']
    
    # Market Memory Effects
    # Volume echo patterns from previous levels
    data['volume_echo'] = data['volume'] / data['volume'].shift(5).rolling(window=10, min_periods=5).mean()
    
    # Price memory through volume confirmation
    data['price_level'] = (data['close'] - data['close'].rolling(window=20, min_periods=10).min()) / \
                         (data['close'].rolling(window=20, min_periods=10).max() - data['close'].rolling(window=20, min_periods=10).min())
    data['volume_at_level'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    data['memory_confirmation'] = data['price_level'] * data['volume_at_level']
    
    # Liquidity Fracture Detection
    # Volume gap identification at key price levels
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / \
                           data['volume'].rolling(window=20, min_periods=10).std()
    data['liquidity_fracture'] = np.where(data['volume_zscore'].abs() > 2, data['volume_zscore'], 0)
    
    # Volume concentration fracture points
    data['volume_quantile'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['volume_concentration'] = np.where(data['volume_quantile'] > 0.8, 1, 
                                          np.where(data['volume_quantile'] < 0.2, -1, 0))
    
    # Regime-Sensitive Processing
    # High-volatility fractal compression
    data['volatility'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    data['volatility_regime'] = data['volatility'] / data['volatility'].rolling(window=50, min_periods=25).mean()
    
    # Adaptive scale selection based on volatility
    data['adaptive_scale'] = np.where(data['volatility_regime'] > 1.2, 0.7,
                                    np.where(data['volatility_regime'] < 0.8, 1.3, 1.0))
    
    # Signal Integration
    # Multi-fractal convergence signals
    fractal_signals = (data['volume_clustering'] * 0.2 + 
                      data['price_fractal_similarity'] * 0.3 + 
                      data['volume_fractal_dim'] * 0.2)
    
    # Divergence confirmation across timeframes
    divergence_signals = (data['momentum_divergence'] * 0.25 + 
                         data['accel_divergence'] * 0.2 + 
                         data['volume_echo'] * 0.15)
    
    # Fracture-implied momentum shifts
    fracture_signals = (data['liquidity_fracture'] * 0.15 + 
                       data['volume_concentration'] * 0.1 + 
                       data['memory_confirmation'] * 0.2)
    
    # Combine all components with adaptive scaling
    factor = (fractal_signals + divergence_signals + fracture_signals) * data['adaptive_scale']
    
    # Final normalization
    factor_series = (factor - factor.rolling(window=50, min_periods=25).mean()) / \
                   factor.rolling(window=50, min_periods=25).std()
    
    return factor_series
