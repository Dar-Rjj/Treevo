import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Volume Divergence with Fractal Market Structure alpha factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for different timeframes
    data['returns_5'] = data['close'].pct_change(5)
    data['returns_10'] = data['close'].pct_change(10)
    data['returns_20'] = data['close'].pct_change(20)
    
    # Hurst exponent calculation function
    def hurst_exponent(ts, window=20):
        hurst_values = []
        for i in range(len(ts)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = ts.iloc[i-window:i].dropna()
            if len(window_data) < 10:
                hurst_values.append(np.nan)
                continue
                
            # Rescaled range analysis
            lags = range(2, min(10, len(window_data)//2))
            tau = []
            for lag in lags:
                # Calculate R/S for each lag
                rs_values = []
                for j in range(0, len(window_data)-lag, lag):
                    segment = window_data.iloc[j:j+lag]
                    mean_segment = segment.mean()
                    deviations = segment - mean_segment
                    Z = deviations.cumsum()
                    R = Z.max() - Z.min()
                    S = segment.std()
                    if S > 0:
                        rs_values.append(R/S)
                
                if rs_values:
                    tau.append(np.log(np.mean(rs_values)))
                else:
                    tau.append(np.nan)
            
            # Remove NaN values
            valid_lags = [(np.log(lag), val) for lag, val in zip(lags, tau) if not np.isnan(val)]
            if len(valid_lags) < 2:
                hurst_values.append(np.nan)
                continue
                
            lags_log, tau_vals = zip(*valid_lags)
            slope, _, _, _, _ = linregress(lags_log, tau_vals)
            hurst_values.append(slope)
        
        return pd.Series(hurst_values, index=ts.index)
    
    # Calculate multi-timeframe Hurst exponents
    data['hurst_short'] = hurst_exponent(data['close'], 20)
    data['hurst_medium'] = hurst_exponent(data['close'], 40)
    data['hurst_long'] = hurst_exponent(data['close'], 60)
    
    # Fractal regime classification
    data['regime_short'] = np.where(data['hurst_short'] > 0.55, 1, 
                                   np.where(data['hurst_short'] < 0.45, -1, 0))
    data['regime_medium'] = np.where(data['hurst_medium'] > 0.55, 1, 
                                    np.where(data['hurst_medium'] < 0.45, -1, 0))
    data['regime_long'] = np.where(data['hurst_long'] > 0.55, 1, 
                                  np.where(data['hurst_long'] < 0.45, -1, 0))
    
    # Volume distribution analysis
    data['price_range'] = data['high'] - data['low']
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['volume_to_range'] = data['volume'] / (data['price_range'] + 1e-8)
    
    # Volume concentration at support/resistance levels
    data['volume_ma_5'] = data['volume'].rolling(5).mean()
    data['volume_ma_20'] = data['volume'].rolling(20).mean()
    data['volume_concentration'] = data['volume'] / (data['volume_ma_20'] + 1e-8)
    
    # Price-volume pressure indicators
    data['price_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['volume_pressure'] = data['price_pressure'] * data['volume_concentration']
    
    # Buy/sell pressure asymmetry
    data['buy_pressure'] = np.where(data['close'] > data['open'], 
                                   data['volume'] * (data['close'] - data['open']), 0)
    data['sell_pressure'] = np.where(data['close'] < data['open'], 
                                    data['volume'] * (data['open'] - data['close']), 0)
    
    data['pressure_imbalance'] = (data['buy_pressure'] - data['sell_pressure']) / \
                                (data['buy_pressure'] + data['sell_pressure'] + 1e-8)
    
    # Multi-scale divergence factors
    # Short-term momentum divergence
    data['price_momentum_5'] = data['close'].pct_change(5)
    data['volume_momentum_5'] = data['volume'].pct_change(5)
    data['short_divergence'] = data['price_momentum_5'] - data['volume_momentum_5']
    
    # Medium-term regime-break divergence
    data['price_trend_10'] = data['close'].rolling(10).apply(
        lambda x: linregress(range(len(x)), x)[0] if len(x) > 5 else np.nan, raw=False
    )
    data['volume_trend_10'] = data['volume'].rolling(10).apply(
        lambda x: linregress(range(len(x)), x)[0] if len(x) > 5 else np.nan, raw=False
    )
    data['medium_divergence'] = data['price_trend_10'] - data['volume_trend_10']
    
    # Long-term structural divergence
    data['price_volatility_20'] = data['returns_20'].rolling(20).std()
    data['volume_volatility_20'] = data['volume'].pct_change().rolling(20).std()
    data['long_divergence'] = data['price_volatility_20'] - data['volume_volatility_20']
    
    # Fractal-adaptive signal combination
    # Regime-based weights
    data['trend_strength'] = (data['regime_short'] + data['regime_medium'] + data['regime_long']) / 3
    
    # Scale signals based on fractal persistence
    data['short_weight'] = np.where(data['trend_strength'] > 0.3, 1.2, 
                                   np.where(data['trend_strength'] < -0.3, 0.8, 1.0))
    data['medium_weight'] = np.where(np.abs(data['trend_strength']) < 0.2, 1.3, 1.0)
    data['long_weight'] = np.where(np.abs(data['trend_strength']) > 0.4, 1.2, 1.0)
    
    # Apply weights and combine signals
    data['weighted_short'] = data['short_divergence'] * data['short_weight']
    data['weighted_medium'] = data['medium_divergence'] * data['medium_weight']
    data['weighted_long'] = data['long_divergence'] * data['long_weight']
    
    # Composite alpha factor with volume pressure adjustment
    data['composite_alpha'] = (data['weighted_short'] * 0.4 + 
                              data['weighted_medium'] * 0.35 + 
                              data['weighted_long'] * 0.25) * \
                             (1 + data['pressure_imbalance'])
    
    # Final normalization and smoothing
    alpha = data['composite_alpha'].rolling(5).mean()
    
    return alpha
