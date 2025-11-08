import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(data):
    """
    Generate alpha factor combining RSI adjusted by volume trend slope and volatility.
    """
    df = data.copy()
    
    # Calculate RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(df['close'])
    
    # Calculate Volume Trend Slope standardized to [-1, 1]
    def volume_trend_slope(volume, window=20):
        slopes = pd.Series(index=volume.index, dtype=float)
        
        for i in range(window-1, len(volume)):
            if i >= window-1:
                vol_window = volume.iloc[i-window+1:i+1].values
                X = np.arange(len(vol_window)).reshape(-1, 1)
                y = vol_window
                
                model = LinearRegression()
                model.fit(X, y)
                slope = model.coef_[0]
                
                # Standardize slope to [-1, 1] range using min-max scaling
                max_slope = volume.rolling(window).std().iloc[i] * 2  # Empirical scaling
                standardized_slope = np.clip(slope / max_slope if max_slope != 0 else 0, -1, 1)
                slopes.iloc[i] = standardized_slope
        
        return slopes
    
    volume_slope = volume_trend_slope(df['volume'])
    
    # Calculate rolling volatility (standard deviation of returns)
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=20).std()
    
    # Combine RSI and Volume Trend Slope, adjusted by volatility
    combined_factor = rsi * volume_slope
    
    # Volatility adjustment - divide by volatility (avoid division by zero)
    volatility_adj = volatility.replace(0, np.nan)
    final_factor = combined_factor / volatility_adj
    
    return final_factor
