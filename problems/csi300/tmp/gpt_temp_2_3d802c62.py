import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volatility-Normalized Momentum
    # Short-term momentum calculation
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_3d'] = df['close'].pct_change(3)
    
    # Volatility estimation
    df['daily_range'] = df['high'] - df['low']
    df['avg_range_5d'] = df['daily_range'].rolling(window=5).mean()
    
    # Normalization
    df['norm_momentum_5d'] = df['momentum_5d'] / df['avg_range_5d']
    df['norm_momentum_3d'] = df['momentum_3d'] / df['daily_range']
    
    # Volume Divergence Analysis
    # Volume trend calculation
    def volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(5)
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    df['volume_slope'] = df['volume'].rolling(window=5).apply(volume_slope, raw=True)
    df['volume_vs_avg'] = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Direction comparison
    df['momentum_dir'] = np.sign(df['momentum_5d'])
    df['volume_dir'] = np.sign(df['volume_slope'])
    
    # Divergence strength
    momentum_threshold = df['momentum_5d'].abs().rolling(window=20).quantile(0.7)
    df['strong_divergence'] = ((df['momentum_5d'].abs() > momentum_threshold) & 
                              (df['momentum_dir'] != df['volume_dir']))
    df['weak_divergence'] = ((df['momentum_5d'].abs() <= momentum_threshold) | 
                            (df['momentum_dir'] == df['volume_dir']))
    
    # Regime Detection
    # Volatility regime
    df['atr'] = (df['high'] - df['low']).rolling(window=20).mean()
    median_atr_60d = df['atr'].rolling(window=60).median()
    df['high_vol_regime'] = df['atr'] > median_atr_60d
    df['low_vol_regime'] = df['atr'] <= median_atr_60d
    
    # Market regime
    df['market_return_20d'] = df['close'].pct_change(20)
    df['bullish_regime'] = df['market_return_20d'] > 0
    df['bearish_regime'] = df['market_return_20d'] <= 0
    
    # Signal Integration
    # Base signal construction
    df['base_signal'] = 0.6 * df['norm_momentum_5d'] + 0.4 * df['norm_momentum_3d']
    
    # Volume divergence multiplier
    df['divergence_multiplier'] = 1.0
    df.loc[df['strong_divergence'], 'divergence_multiplier'] = 1.5
    df.loc[df['weak_divergence'], 'divergence_multiplier'] = 0.8
    
    # Regime adjustment
    df['regime_weight'] = 1.0
    df.loc[df['high_vol_regime'], 'regime_weight'] = -0.7  # Mean reversion emphasis
    df.loc[df['low_vol_regime'], 'regime_weight'] = 1.2    # Momentum continuation
    df.loc[df['bullish_regime'], 'regime_weight'] *= 1.3   # Amplification in bullish
    df.loc[df['bearish_regime'], 'regime_weight'] *= 0.8   # Reduction in bearish
    
    # Final alpha factor
    alpha_factor = (df['base_signal'] * 
                   df['divergence_multiplier'] * 
                   df['regime_weight'])
    
    return alpha_factor
