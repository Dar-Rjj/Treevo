import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(data):
    """
    Dynamic Volatility-Adjusted Momentum with Volume Confirmation factor
    """
    df = data.copy()
    
    # Momentum Calculation
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    
    # Daily volatility proxy
    df['daily_range'] = df['high'] - df['low']
    
    # Volatility Adjustment
    df['vol_adjusted_momentum'] = df['momentum_5d'] / (1 + df['daily_range'])
    
    # Volume Confirmation
    def calculate_volume_trend(volume_series):
        if len(volume_series) < 5:
            return 0
        x = np.arange(5)
        y = volume_series.values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    df['volume_trend_slope'] = df['volume'].rolling(window=5).apply(
        calculate_volume_trend, raw=False
    )
    
    df['volume_confirmation'] = np.where(
        (df['momentum_5d'] > 0) & (df['volume_trend_slope'] > 0) |
        (df['momentum_5d'] < 0) & (df['volume_trend_slope'] < 0),
        2, 1
    )
    
    # Regime Classification
    df['atr_20'] = df['daily_range'].rolling(window=20).mean()
    df['atr_60_median'] = df['atr_20'].rolling(window=60).median()
    
    df['high_vol_regime'] = df['atr_20'] > df['atr_60_median']
    
    # Signal Generation
    df['base_signal'] = df['vol_adjusted_momentum'] * df['volume_confirmation']
    
    df['final_signal'] = np.where(
        df['high_vol_regime'],
        df['base_signal'] * -1,  # Mean reversion in high volatility
        df['base_signal'] * 1    # Momentum continuation in low volatility
    )
    
    return df['final_signal']
