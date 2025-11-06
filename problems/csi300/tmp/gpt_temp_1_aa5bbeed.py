import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Decay with Volume Confirmation
    # Compute Price Momentum
    short_term_return = data['close'] / data['close'].shift(5) - 1
    medium_term_return = data['close'] / data['close'].shift(20) - 1
    
    # Apply Exponential Decay
    momentum_score = 0.7 * short_term_return + 0.3 * medium_term_return
    
    # Volume Confirmation
    volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    momentum_factor = momentum_score * volume_ratio
    
    # Intraday Range Efficiency
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compute Price Movement
    price_movement = abs(data['close'] - data['open'])
    
    # Calculate Efficiency Ratio
    efficiency_ratio = price_movement / true_range
    efficiency_factor = efficiency_ratio.rolling(window=5).mean()
    
    # Volatility Regime Adjusted Momentum
    # Calculate Rolling Volatility
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=20).std()
    
    # Compute Momentum Signal
    momentum_10d = data['close'] / data['close'].shift(10) - 1
    momentum_signal = np.sign(momentum_10d)
    
    # Adjust by Volatility Regime
    volatility_adjusted = momentum_signal / (volatility + 1e-8)
    
    # Volume-Price Divergence Factor
    # Price Trend Component
    def linear_regression_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    price_slope = linear_regression_slope(data['close'], 5)
    volume_slope = linear_regression_slope(data['volume'], 5)
    
    # Calculate Divergence
    divergence = price_slope * volume_slope
    divergence_factor = divergence.diff()
    
    # Liquidity-Adjusted Reversal
    # Compute Short-Term Reversal
    reversal_1d = -data['close'].pct_change()
    
    # Assess Liquidity Condition
    def volume_percentile(volume, window):
        percentiles = []
        for i in range(len(volume)):
            if i < window - 1:
                percentiles.append(np.nan)
            else:
                window_data = volume.iloc[i-window+1:i+1]
                current_vol = volume.iloc[i]
                percentile = (window_data <= current_vol).sum() / window
                percentiles.append(percentile)
        return pd.Series(percentiles, index=volume.index)
    
    volume_pct = volume_percentile(data['volume'], 10)
    
    # Adjust Reversal Strength
    reversal_factor = reversal_1d * volume_pct
    
    # Combine all factors with equal weights
    final_factor = (
        momentum_factor.fillna(0) +
        efficiency_factor.fillna(0) +
        volatility_adjusted.fillna(0) +
        divergence_factor.fillna(0) +
        reversal_factor.fillna(0)
    ) / 5
    
    return final_factor
