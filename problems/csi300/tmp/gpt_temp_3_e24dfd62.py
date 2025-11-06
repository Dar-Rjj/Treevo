import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Regime-Adaptive Volatility-Weighted Momentum factor
    Combines price and volume entropy for regime identification with volatility-adjusted momentum
    """
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Market Regime Identification
    # Price Entropy Calculation
    returns_5d = data['close'].pct_change(periods=5)
    returns_10d = data['close'].pct_change(periods=10)
    returns_20d = data['close'].pct_change(periods=20)
    
    # Calculate rolling entropy for price returns (20-day window)
    def calculate_price_entropy(series, window=20):
        entropy_values = []
        for i in range(len(series)):
            if i < window:
                entropy_values.append(np.nan)
            else:
                window_data = series.iloc[i-window:i].dropna()
                if len(window_data) > 0:
                    # Create histogram bins for entropy calculation
                    hist, _ = np.histogram(window_data, bins=10, density=True)
                    hist = hist[hist > 0]  # Remove zero bins for log calculation
                    if len(hist) > 1:
                        entropy_val = entropy(hist)
                        entropy_values.append(entropy_val)
                    else:
                        entropy_values.append(np.nan)
                else:
                    entropy_values.append(np.nan)
        return pd.Series(entropy_values, index=series.index)
    
    price_entropy = calculate_price_entropy(data['close'].pct_change(), window=20)
    
    # Volume Entropy Calculation
    def calculate_volume_entropy(volume_series, window=10):
        entropy_values = []
        for i in range(len(volume_series)):
            if i < window:
                entropy_values.append(np.nan)
            else:
                window_data = volume_series.iloc[i-window:i]
                if len(window_data) > 0:
                    # Normalize volume data
                    norm_volume = (window_data - window_data.min()) / (window_data.max() - window_data.min() + 1e-8)
                    hist, _ = np.histogram(norm_volume, bins=8, density=True)
                    hist = hist[hist > 0]
                    if len(hist) > 1:
                        entropy_val = entropy(hist)
                        entropy_values.append(entropy_val)
                    else:
                        entropy_values.append(np.nan)
                else:
                    entropy_values.append(np.nan)
        return pd.Series(entropy_values, index=volume_series.index)
    
    volume_entropy = calculate_volume_entropy(data['volume'], window=10)
    
    # Combined regime score (normalized entropy measures)
    price_entropy_norm = (price_entropy - price_entropy.rolling(50).mean()) / price_entropy.rolling(50).std()
    volume_entropy_norm = (volume_entropy - volume_entropy.rolling(50).mean()) / volume_entropy.rolling(50).std()
    regime_score = 0.6 * price_entropy_norm + 0.4 * volume_entropy_norm
    
    # 2. Raw Momentum Calculation
    momentum_10d = data['close'].pct_change(periods=10)
    
    # 3. Volatility Adjustment
    daily_range = (data['high'] - data['low']) / data['close'].shift(1)
    volatility_20d = daily_range.rolling(window=20).mean()
    
    # 4. Regime-Adaptive Weighting
    factor_values = []
    
    for i in range(len(data)):
        if (pd.isna(regime_score.iloc[i]) or pd.isna(momentum_10d.iloc[i]) or 
            pd.isna(volatility_20d.iloc[i])):
            factor_values.append(np.nan)
            continue
        
        regime = regime_score.iloc[i]
        momentum = momentum_10d.iloc[i]
        volatility = volatility_20d.iloc[i]
        
        # High entropy regime (chaotic markets)
        if regime > 0.5:
            # Mean-reversion emphasis with volatility filtering
            if volatility < volatility_20d.rolling(50).mean().iloc[i]:
                # Higher weight during low volatility in chaotic markets
                weight = 1.2 - (volatility / (volatility_20d.rolling(50).mean().iloc[i] + 1e-8))
            else:
                # Reduced weight during high volatility
                weight = 0.8 - (volatility / (volatility_20d.rolling(50).mean().iloc[i] + 1e-8))
            
            # Apply mean-reversion logic (negative momentum gets positive weight)
            factor_value = -momentum * weight
        
        # Low entropy regime (trending markets)
        else:
            # Momentum emphasis with asymmetric volatility weighting
            if volatility < volatility_20d.rolling(50).mean().iloc[i]:
                # Stronger weight during low volatility
                weight = 1.5 - (0.5 * volatility / (volatility_20d.rolling(50).mean().iloc[i] + 1e-8))
            else:
                # Reduced weight during high volatility
                weight = 0.7 - (0.3 * volatility / (volatility_20d.rolling(50).mean().iloc[i] + 1e-8))
            
            # Pure momentum in trending markets
            factor_value = momentum * weight
        
        factor_values.append(factor_value)
    
    factor_series = pd.Series(factor_values, index=data.index)
    
    return factor_series
