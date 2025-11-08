import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Divergence Analysis
    data['short_medium_div'] = data['momentum_5d'] - data['momentum_10d']
    data['medium_long_div'] = data['momentum_10d'] - data['momentum_20d']
    data['combined_momentum_div'] = (0.6 * data['short_medium_div'] + 
                                    0.4 * data['medium_long_div'])
    
    # Volume Divergence Component
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['volume_dev_5d'] = (data['volume'] - data['volume_5d_avg']) / data['volume_5d_avg']
    data['volume_dev_10d'] = (data['volume'] - data['volume_10d_avg']) / data['volume_10d_avg']
    data['volume_div'] = data['volume_dev_5d'] - data['volume_dev_10d']
    
    # Volatility Regime Assessment
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_20d'] = data['daily_range'].rolling(window=20).std()
    
    # Calculate volatility percentiles for regime classification
    volatility_percentiles = data['volatility_20d'].rolling(window=60, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 20 else np.nan, raw=False
    )
    
    # Regime classification
    high_vol_regime = volatility_percentiles > 0.6
    low_vol_regime = volatility_percentiles < 0.4
    
    # Signal Integration
    data['raw_divergence'] = data['combined_momentum_div'] * data['volume_div']
    
    # Regime-Adaptive Weighting
    data['regime_adjusted_signal'] = np.nan
    
    # High volatility regime adjustments
    high_vol_mask = high_vol_regime & ~low_vol_regime
    data.loc[high_vol_mask, 'regime_adjusted_signal'] = (
        data.loc[high_vol_mask, 'raw_divergence'] * 1.5
    )
    
    # Low volatility regime adjustments
    low_vol_mask = low_vol_regime & ~high_vol_regime
    data.loc[low_vol_mask, 'regime_adjusted_signal'] = (
        data.loc[low_vol_mask, 'raw_divergence'] * 0.8
    )
    
    # Normal regime (between 40th and 60th percentiles)
    normal_mask = ~high_vol_regime & ~low_vol_regime
    data.loc[normal_mask, 'regime_adjusted_signal'] = data.loc[normal_mask, 'raw_divergence']
    
    # Final Factor Generation with volatility scaling
    momentum_volatility = data['momentum_20d'].rolling(window=20).std()
    data['final_factor'] = data['regime_adjusted_signal'] / (momentum_volatility + 1e-8)
    
    # Return the final factor series
    return data['final_factor']
