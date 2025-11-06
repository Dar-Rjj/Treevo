import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Entropy calculation function
    def calculate_entropy(series):
        if len(series) < 2:
            return 1.0
        # Normalize the series to probabilities
        normalized = (series - series.min()) / (series.max() - series.min() + 1e-8)
        normalized = normalized + 1e-8  # Avoid zeros
        probabilities = normalized / normalized.sum()
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log(probabilities))
        return entropy + 1e-8  # Avoid division by zero
    
    # Entropy-Based Price Momentum: (Close[t] - SMA(Close[t-9:t])) / Entropy(Close[t-9:t])
    close_sma_10 = data['close'].rolling(window=10, min_periods=10).mean()
    price_momentum = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 9:
            window = data['close'].iloc[i-9:i+1]
            entropy_val = calculate_entropy(window)
            momentum = (data['close'].iloc[i] - close_sma_10.iloc[i]) / entropy_val
            price_momentum.iloc[i] = momentum
    
    # Volume Fractal Entropy: Log(Volume[t] / Volume[t-5]) / Entropy(Volume[t-4:t])
    volume_fractal_entropy = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 5:
            vol_ratio = np.log(data['volume'].iloc[i] / (data['volume'].iloc[i-5] + 1e-8))
            vol_window = data['volume'].iloc[i-4:i+1]
            entropy_val = calculate_entropy(vol_window)
            volume_fractal_entropy.iloc[i] = vol_ratio / (entropy_val + 1e-8)
    
    # Fractal Regime Detection
    high_fractal = pd.Series(index=data.index, dtype=float)
    low_fractal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 5:
            # High-Fractal Regime
            high_diff = data['high'].iloc[i] - data['high'].iloc[i-5]
            high_volatility = sum(abs(data['high'].iloc[j] - data['high'].iloc[j-1]) 
                                for j in range(i-4, i+1))
            high_fractal.iloc[i] = high_diff / (high_volatility + 1e-8)
            
            # Low-Fractal Regime
            low_diff = data['low'].iloc[i] - data['low'].iloc[i-5]
            low_volatility = sum(abs(data['low'].iloc[j] - data['low'].iloc[j-1]) 
                               for j in range(i-4, i+1))
            low_fractal.iloc[i] = low_diff / (low_volatility + 1e-8)
    
    # Price-Volume Fractal Divergence
    price_fractal = pd.Series(index=data.index, dtype=float)
    volume_fractal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 10:
            # Price Fractal
            price_diff = data['close'].iloc[i] - data['close'].iloc[i-10]
            price_volatility = sum(abs(data['close'].iloc[j] - data['close'].iloc[j-1]) 
                                 for j in range(i-9, i+1))
            price_fractal.iloc[i] = price_diff / (price_volatility + 1e-8)
            
            # Volume Fractal
            vol_ratio = np.log(data['volume'].iloc[i] / (data['volume'].iloc[i-10] + 1e-8))
            vol_volatility = sum(abs(np.log(data['volume'].iloc[j] / (data['volume'].iloc[j-1] + 1e-8)))
                               for j in range(i-9, i+1))
            volume_fractal.iloc[i] = vol_ratio / (vol_volatility + 1e-8)
    
    # Price-Volume Fractal Divergence (combined)
    price_volume_divergence = price_fractal - volume_fractal
    
    # Adaptive Entropy Alpha: Entropy-Based Price Momentum × (High-Fractal Regime - Low-Fractal Regime) × Price-Volume Fractal Divergence
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 10:
            regime_diff = high_fractal.iloc[i] - low_fractal.iloc[i]
            alpha_factor.iloc[i] = (price_momentum.iloc[i] * 
                                  regime_diff * 
                                  price_volume_divergence.iloc[i])
    
    # Fill NaN values with 0
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
