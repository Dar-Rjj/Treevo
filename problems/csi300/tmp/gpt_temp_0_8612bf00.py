import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Momentum Divergence factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Realized Volatility (Parkinson estimator using High-Low)
    data['hl_range'] = (data['high'] - data['low']) / data['close']
    data['realized_vol_20d'] = data['hl_range'].rolling(window=20).std()
    
    # Volatility regime classification
    vol_median = data['realized_vol_20d'].rolling(window=60).median()
    data['vol_regime'] = np.where(data['realized_vol_20d'] > vol_median, 1, 0)
    
    # Volume-Weighted Price Trend (VWAP)
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap'] = (data['typical_price'] * data['volume']).rolling(window=10).sum() / data['volume'].rolling(window=10).sum()
    
    # VWAP trend slope (linear regression over 10 days)
    def vwap_slope(series):
        if len(series) < 10:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    data['vwap_trend'] = data['vwap'].rolling(window=10).apply(vwap_slope, raw=False)
    
    # Volume trend
    data['volume_trend'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 10 else np.nan, 
        raw=False
    )
    
    # Momentum Divergence Calculation
    # Divergence between short-term momentum and VWAP trend
    data['momentum_divergence'] = data['momentum_3d'] - data['vwap_trend']
    
    # Volume confirmation (alignment between momentum and volume trends)
    data['volume_confirmation'] = np.sign(data['momentum_3d']) * np.sign(data['volume_trend'])
    
    # Volatility adjustment
    vol_normalized = data['realized_vol_20d'] / data['realized_vol_20d'].rolling(window=60).mean()
    data['volatility_adjustment'] = 1 / (1 + vol_normalized)
    
    # Final factor: Volatility-Adjusted Momentum Divergence with Volume Confirmation
    data['factor'] = (
        data['momentum_divergence'] * 
        data['volatility_adjustment'] * 
        (1 + 0.5 * data['volume_confirmation'])
    )
    
    # Handle any remaining NaN values
    data['factor'] = data['factor'].fillna(0)
    
    return data['factor']
