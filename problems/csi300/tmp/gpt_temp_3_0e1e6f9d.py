import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Component
    # Short-term price momentum
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Intraday pressure
    data['intraday_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_pressure'] = data['intraday_pressure'].replace([np.inf, -np.inf], np.nan)
    
    # Combined momentum score
    data['momentum_score'] = 0.6 * data['momentum_5d'] + 0.4 * data['intraday_pressure']
    
    # Volume Confirmation
    # Volume ratio
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_5d']
    
    # Volume trend (3-day slope)
    def volume_slope(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0] / series.mean()
    
    data['volume_trend'] = data['volume'].rolling(window=3, min_periods=3).apply(
        volume_slope, raw=False
    )
    
    # Volume confirmation score
    data['volume_score'] = 0.7 * data['volume_ratio'] + 0.3 * data['volume_trend']
    
    # Regime Adaptation
    # Volatility regime (5-day ATR vs 20-day median)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_5d'] = data['tr'].rolling(window=5, min_periods=3).mean()
    data['atr_median_20d'] = data['tr'].rolling(window=20, min_periods=10).median()
    data['volatility_regime'] = data['atr_5d'] / data['atr_median_20d']
    
    # Oscillation pattern (5-day price direction changes)
    def direction_changes(series):
        if len(series) < 5:
            return np.nan
        returns = series.pct_change().dropna()
        direction_changes = sum((returns.iloc[i] * returns.iloc[i-1] < 0) 
                              for i in range(1, len(returns)))
        return direction_changes / (len(returns) - 1)
    
    data['oscillation'] = data['close'].rolling(window=5, min_periods=5).apply(
        direction_changes, raw=False
    )
    
    # Regime score
    data['regime_score'] = np.where(
        data['volatility_regime'] > 1.2,
        0.8,  # High volatility regime - reduce weight
        np.where(
            data['oscillation'] > 0.6,
            0.6,  # High oscillation - moderate weight
            1.0   # Normal regime - full weight
        )
    )
    
    # Liquidity Filter
    # Price impact
    data['price_impact'] = (data['high'] - data['low']) / data['close']
    
    # Volume efficiency
    data['volume_efficiency'] = data['volume'] / data['amount']
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Liquidity score (inverse of illiquidity)
    data['liquidity_score'] = 1 / (0.6 * data['price_impact'] + 0.4 * data['volume_efficiency'])
    data['liquidity_score'] = np.clip(data['liquidity_score'], 0.1, 10)  # Bound extreme values
    
    # Composite Construction
    # Momentum × Volume weighting with regime scaling and liquidity filtering
    data['composite_factor'] = (
        data['momentum_score'] * 
        data['volume_score'] * 
        data['regime_score'] * 
        data['liquidity_score']
    )
    
    # Final factor with normalization
    factor = data['composite_factor']
    factor = (factor - factor.rolling(window=20, min_periods=10).mean()) / factor.rolling(window=20, min_periods=10).std()
    
    return factor
