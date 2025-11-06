import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Divergence Momentum factor
    Combines multi-timeframe price fractal analysis with volume fractal characteristics
    to detect divergence patterns and regime transitions
    """
    df = df.copy()
    
    # Multi-Timeframe Price Fractal Analysis
    # Short-term (3-day) fractal pattern detection
    df['short_fractal_high'] = (df['high'] > df['high'].shift(1)) & \
                               (df['high'] > df['high'].shift(2)) & \
                               (df['high'] > df['high'].shift(-1)) & \
                               (df['high'] > df['high'].shift(-2))
    df['short_fractal_low'] = (df['low'] < df['low'].shift(1)) & \
                              (df['low'] < df['low'].shift(2)) & \
                              (df['low'] < df['low'].shift(-1)) & \
                              (df['low'] < df['low'].shift(-2))
    
    # Medium-term (5-day) fractal structure mapping
    df['medium_fractal_high'] = (df['high'] > df['high'].shift(1)) & \
                                (df['high'] > df['high'].shift(2)) & \
                                (df['high'] > df['high'].shift(3)) & \
                                (df['high'] > df['high'].shift(-1)) & \
                                (df['high'] > df['high'].shift(-2))
    df['medium_fractal_low'] = (df['low'] < df['low'].shift(1)) & \
                               (df['low'] < df['low'].shift(2)) & \
                               (df['low'] < df['low'].shift(3)) & \
                               (df['low'] < df['low'].shift(-1)) & \
                               (df['low'] < df['low'].shift(-2))
    
    # Fractal momentum and directional bias calculation
    df['fractal_momentum'] = 0
    df.loc[df['short_fractal_high'], 'fractal_momentum'] += 1
    df.loc[df['medium_fractal_high'], 'fractal_momentum'] += 1
    df.loc[df['short_fractal_low'], 'fractal_momentum'] -= 1
    df.loc[df['medium_fractal_low'], 'fractal_momentum'] -= 1
    
    # Volume Fractal Characteristics
    # Volume pattern complexity using rolling entropy
    volume_rolling = df['volume'].rolling(window=5, min_periods=3)
    volume_pct_rank = volume_rolling.apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    df['volume_complexity'] = -volume_pct_rank * np.log(volume_pct_rank + 1e-8)
    
    # Volume clustering analysis using Z-score
    volume_mean = df['volume'].rolling(window=10, min_periods=5).mean()
    volume_std = df['volume'].rolling(window=10, min_periods=5).std()
    df['volume_clustering'] = (df['volume'] - volume_mean) / (volume_std + 1e-8)
    
    # Price-volume fractal divergence detection
    price_change = df['close'].pct_change(3)
    volume_change = df['volume'].pct_change(3)
    df['pv_divergence'] = np.sign(price_change) * np.sign(volume_change) * \
                         np.abs(price_change - volume_change)
    
    # Fractal persistence and stability assessment
    fractal_stability_short = df['short_fractal_high'].rolling(window=5).sum() + \
                             df['short_fractal_low'].rolling(window=5).sum()
    fractal_stability_medium = df['medium_fractal_high'].rolling(window=8).sum() + \
                              df['medium_fractal_low'].rolling(window=8).sum()
    df['fractal_stability'] = (fractal_stability_short + fractal_stability_medium) / 13.0
    
    # Fractal Divergence Signal Generation
    # Multi-scale fractal information integration
    fractal_momentum_smooth = df['fractal_momentum'].rolling(window=5).mean()
    volume_complexity_smooth = df['volume_complexity'].rolling(window=5).mean()
    
    # Fractal regime transition identification
    regime_change = (df['fractal_momentum'].diff(3).abs() > 1.5).astype(int)
    stability_change = (df['fractal_stability'].diff(3).abs() > 0.3).astype(int)
    df['regime_transition'] = regime_change + stability_change
    
    # Final factor calculation integrating all components
    factor = (fractal_momentum_smooth * 0.4 + 
              df['pv_divergence'] * 0.3 + 
              volume_complexity_smooth * 0.2 + 
              df['fractal_stability'] * 0.1 + 
              df['regime_transition'] * 0.2)
    
    # Clean up intermediate columns
    cols_to_drop = ['short_fractal_high', 'short_fractal_low', 'medium_fractal_high', 
                   'medium_fractal_low', 'fractal_momentum', 'volume_complexity', 
                   'volume_clustering', 'pv_divergence', 'fractal_stability', 'regime_transition']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return factor
