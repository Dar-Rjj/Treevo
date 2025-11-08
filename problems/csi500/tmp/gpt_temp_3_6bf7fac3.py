import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Fractal Coherence Factor
    Combines multi-timeframe volume analysis with price fractal dimension
    to detect coherent momentum signals vs. divergence patterns.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Multi-timeframe Volume Distribution
        vol_5d = current_data['volume'].rolling(window=5, min_periods=5).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        ).iloc[-1]
        
        vol_10d = current_data['volume'].rolling(window=10, min_periods=10).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        ).iloc[-1]
        
        vol_20d = current_data['volume'].rolling(window=20, min_periods=20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
        ).iloc[-1]
        
        # Volume concentration at price levels
        recent_high = current_data['high'].iloc[-5:].max()
        recent_low = current_data['low'].iloc[-5:].min()
        current_close = current_data['close'].iloc[-1]
        
        price_level = (current_close - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        volume_concentration = abs(price_level - 0.5) * 2  # Higher when near extremes
        
        # Price Fractal Dimension Analysis (simplified Hurst exponent)
        def compute_fractal_dimension(prices, window=10):
            if len(prices) < window:
                return 0.5
            log_returns = np.log(prices / prices.shift(1)).dropna()
            if len(log_returns) < window:
                return 0.5
            rs_values = []
            for j in range(len(log_returns) - window + 1):
                window_returns = log_returns.iloc[j:j+window]
                mean_return = window_returns.mean()
                cumulative_deviation = (window_returns - mean_return).cumsum()
                range_val = cumulative_deviation.max() - cumulative_deviation.min()
                std_val = window_returns.std()
                if std_val > 0:
                    rs_values.append(range_val / std_val)
            
            if len(rs_values) > 1:
                hurst = np.log(np.mean(rs_values)) / np.log(window)
                return max(0.1, min(0.9, hurst))
            return 0.5
        
        # Compute fractal dimensions for different periods
        fractal_10d = compute_fractal_dimension(current_data['close'].iloc[-15:], 10)
        fractal_15d = compute_fractal_dimension(current_data['close'].iloc[-20:], 15)
        
        # Price path complexity vs efficiency
        price_efficiency = 1.0 - abs(fractal_10d - 0.5) * 2  # Higher when near 0.5 (random walk)
        
        # Coherence Signal Generation
        volume_momentum = (vol_5d + vol_10d + vol_20d) / 3
        fractal_momentum = (fractal_10d + fractal_15d) / 2
        
        # High coherence: strong volume support for price trend
        trend_strength = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-5]) / current_data['close'].iloc[-5]
        volume_trend_alignment = volume_momentum * trend_strength * volume_concentration
        
        # Low coherence: volume-price divergence
        volume_spike = vol_5d > 0.8 and vol_10d < 0.5  # Recent spike without sustained volume
        fractal_breakdown = fractal_10d > 0.7 and fractal_15d < 0.6  # Short-term complexity increasing
        
        divergence_signal = -1.0 if (volume_spike or fractal_breakdown) else 0.0
        
        # Final factor calculation
        coherence_factor = (
            volume_trend_alignment * 0.4 +
            price_efficiency * 0.3 +
            fractal_momentum * 0.2 +
            divergence_signal * 0.1
        )
        
        result.iloc[i] = coherence_factor
    
    return result
