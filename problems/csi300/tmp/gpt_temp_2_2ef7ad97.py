import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Regime-Adaptive Price-Volume Momentum factor
    """
    data = df.copy()
    
    # Calculate Volatility Regime
    # Short-Term Volatility (5-day ATR)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['short_vol'] = data['tr'].rolling(window=5).mean()
    
    # Long-Term Volatility (20-day ATR)
    data['long_vol'] = data['tr'].rolling(window=20).mean()
    
    # Volatility Regime Classification
    data['vol_ratio'] = data['short_vol'] / data['long_vol']
    data['high_vol_regime'] = (data['vol_ratio'] > 1.2).astype(int)
    
    # Calculate Price-Volume Trend Alignment
    def calc_trend_strength(series, window=10):
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['price_trend'] = calc_trend_strength(data['close'], 10)
    data['volume_trend'] = calc_trend_strength(data['volume'], 10)
    
    # Trend Convergence
    data['trend_alignment'] = np.sign(data['price_trend']) * np.sign(data['volume_trend'])
    data['trend_strength'] = (abs(data['price_trend']) + abs(data['volume_trend'])) / 2
    
    # Generate Regime-Specific Signals
    # High Volatility Regime Processing
    data['price_momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    
    # Intraday pressure weighting
    data['buying_pressure'] = ((data['close'] > data['open']) * 
                              (data['close'] - data['open']) / data['open'])
    data['selling_pressure'] = ((data['close'] < data['open']) * 
                               (data['open'] - data['close']) / data['open'])
    data['net_pressure_3d'] = (data['buying_pressure'].rolling(window=3).sum() - 
                              data['selling_pressure'].rolling(window=3).sum())
    
    high_vol_signal = (data['price_momentum_3d'] * 0.6 + 
                      data['net_pressure_3d'] * 0.4)
    
    # Low Volatility Regime Processing
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['ma_20'] = data['close'].rolling(window=20).mean()
    data['price_deviation'] = (data['close'] - data['ma_20']) / data['ma_20']
    
    # Amihud illiquidity ratio (10-day average)
    data['daily_illiquidity'] = abs(data['close'].pct_change()) / data['amount']
    data['amihud_ratio'] = data['daily_illiquidity'].rolling(window=10).mean()
    
    low_vol_signal = (data['price_momentum_10d'] * 0.7 - 
                     data['price_deviation'] * 0.2 - 
                     data['amihud_ratio'] * 0.1)
    
    # Combine regime signals
    data['regime_signal'] = np.where(
        data['high_vol_regime'] == 1,
        high_vol_signal,
        low_vol_signal
    )
    
    # Volume Validation
    data['vol_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_significance'] = data['volume'] / data['vol_20d_avg']
    data['volume_volatility_ratio'] = data['volume'] / data['short_vol']
    
    # Generate Final Alpha Factor
    data['volume_weight'] = np.minimum(data['volume_significance'], 2.0)
    data['final_signal'] = (data['regime_signal'] * 
                           data['volume_weight'] * 
                           (1 + data['trend_alignment'] * data['trend_strength']))
    
    # Range normalization using High-Low differences
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5).mean()
    data['alpha_factor'] = data['final_signal'] / data['range_5d_avg']
    
    # Clean and return
    alpha_series = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
