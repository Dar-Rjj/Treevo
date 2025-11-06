import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Dynamic Volatility-Normalized Momentum with Volume Confirmation alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility-Normalized Momentum
    # Short-term momentum calculation (5-day close-to-close returns)
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Volatility estimation using daily range
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_5d'] = data['daily_range'].rolling(window=5).mean()
    
    # Volatility-normalized momentum
    data['vol_norm_momentum'] = data['momentum_5d'] / data['avg_range_5d']
    
    # 2. Volume Confirmation Signal
    # Volume trend calculation using linear regression slope
    def calc_volume_slope(volume_series):
        if len(volume_series) < 5:
            return np.nan
        x = np.arange(5)
        y = volume_series.values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    data['volume_slope'] = data['volume'].rolling(window=5).apply(calc_volume_slope, raw=False)
    
    # Price-volume alignment check and confirmation strength
    data['volume_confirmation'] = 0
    data['confirmation_strength'] = 0.0
    
    # Positive momentum + positive volume slope = confirmation
    mask_pos_conf = (data['momentum_5d'] > 0) & (data['volume_slope'] > 0)
    # Negative momentum + negative volume slope = confirmation
    mask_neg_conf = (data['momentum_5d'] < 0) & (data['volume_slope'] < 0)
    
    data.loc[mask_pos_conf | mask_neg_conf, 'volume_confirmation'] = 1
    data['confirmation_strength'] = np.abs(data['volume_slope'])
    
    # 3. Adaptive Regime Detection
    # True Range calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = np.abs(data['high'] - data['close'].shift(1))
    data['tr3'] = np.abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # 20-day ATR
    data['atr_20d'] = data['true_range'].rolling(window=20).mean()
    
    # Volatility regime classification
    data['atr_median_60d'] = data['atr_20d'].rolling(window=60).median()
    data['high_vol_regime'] = data['atr_20d'] > data['atr_median_60d']
    
    # Regime-specific weights
    data['mean_reversion_weight'] = np.where(data['high_vol_regime'], 0.7, 0.3)
    data['momentum_weight'] = np.where(data['high_vol_regime'], 0.3, 0.7)
    
    # 4. Final Alpha Factor Construction
    # Base signal combination with volume confirmation
    data['base_signal'] = np.where(
        data['volume_confirmation'] == 1,
        data['vol_norm_momentum'] * 1.5,  # Confirmed signal amplified
        data['vol_norm_momentum'] * 0.5   # Unconfirmed signal dampened
    )
    
    # Apply regime adjustment
    # In high volatility: emphasize mean reversion (inverse of momentum)
    # In low volatility: emphasize momentum continuation
    data['mean_reversion_component'] = -data['base_signal'] * data['mean_reversion_weight']
    data['momentum_component'] = data['base_signal'] * data['momentum_weight']
    
    # Final alpha factor
    data['alpha_factor'] = data['mean_reversion_component'] + data['momentum_component']
    
    # Apply confirmation strength as final weighting
    data['final_alpha'] = data['alpha_factor'] * data['confirmation_strength']
    
    return data['final_alpha']
