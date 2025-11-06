import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Acceleration Divergence with Regime-Adaptive Memory Dynamics
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    df['price_change'] = df['close'] / df['close'].shift(1) - 1
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['range_percentage'] = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, np.nan)
    df['order_flow'] = df['volume'] * (df['close'] - df['close'].shift(1))
    df['cumulative_order_flow'] = df['order_flow'].rolling(window=5, min_periods=1).sum()
    df['price_efficiency'] = abs(df['close'] - df['close'].shift(1)) / df['true_range'].replace(0, np.nan)
    df['volume_efficiency'] = df['volume'] / df['amount'].replace(0, np.nan)
    
    # Fractal dimension approximation using Hurst exponent-like calculation
    def calculate_fractal_dimension(series, window=10):
        lags = range(2, min(window, len(series))//2)
        tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0] if len(tau) > 1 else 1.0
    
    # Calculate fractal dimensions
    df['price_fractal'] = df['close'].rolling(window=10, min_periods=5).apply(
        lambda x: calculate_fractal_dimension(x), raw=False
    )
    df['volume_fractal'] = df['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: calculate_fractal_dimension(x), raw=False
    )
    
    # Multi-Scale Fractal Acceleration Analysis
    # Short-term fractal acceleration (3-day)
    df['short_fractal_accel'] = (
        (df['price_fractal'] - df['price_fractal'].shift(3)) - 
        (df['price_fractal'].shift(3) - df['price_fractal'].shift(6))
    )
    
    # Medium-term fractal acceleration (10-day)
    df['medium_fractal_accel'] = (
        (df['price_fractal'] - df['price_fractal'].shift(10)) - 
        (df['price_fractal'].shift(10) - df['price_fractal'].shift(20))
    )
    
    # Volume fractal acceleration
    df['volume_fractal_accel'] = (
        (df['volume_fractal'] - df['volume_fractal'].shift(5)) - 
        (df['volume_fractal'].shift(5) - df['volume_fractal'].shift(10))
    )
    
    # Fractal acceleration divergence
    df['fractal_accel_divergence'] = df['short_fractal_accel'] - df['medium_fractal_accel']
    
    # Regime-Efficient Microstructure Memory Integration
    # Efficiency acceleration
    df['efficiency_accel'] = df['price_efficiency'] - df['price_efficiency'].shift(3)
    df['efficiency_momentum'] = (
        (df['price_efficiency'] - df['price_efficiency'].shift(3)) - 
        (df['price_efficiency'].shift(3) - df['price_efficiency'].shift(6))
    )
    
    # Volatility regime acceleration
    df['volatility_accel'] = df['true_range'] - df['true_range'].shift(5)
    
    # Order flow memory acceleration
    df['order_flow_accel'] = (
        (df['cumulative_order_flow'] - df['cumulative_order_flow'].shift(3)) - 
        (df['cumulative_order_flow'].shift(3) - df['cumulative_order_flow'].shift(6))
    )
    
    # Microstructure impact acceleration
    df['range_ratio'] = (df['high'] - df['low']) / (df['close'] - df['open']).replace(0, np.nan)
    df['intraday_momentum'] = df['close'] / df['open'] - 1
    df['intraday_momentum_accel'] = df['intraday_momentum'] - df['intraday_momentum'].shift(3)
    
    # Range Asymmetry Acceleration
    df['upside_range_accel'] = (
        (df['range_percentage'] - df['range_percentage'].shift(3)) - 
        (df['range_percentage'].shift(3) - df['range_percentage'].shift(6))
    )
    
    # Volume efficiency acceleration
    df['volume_efficiency_accel'] = (
        (df['volume_efficiency'] - df['volume_efficiency'].shift(3)) - 
        (df['volume_efficiency'].shift(3) - df['volume_efficiency'].shift(6))
    )
    
    # Composite factor calculation
    for i in range(len(df)):
        if i < 20:  # Ensure enough data for calculations
            result.iloc[i] = 0
            continue
            
        # Fractal Acceleration Core (40% weight)
        fractal_component = (
            0.6 * df['short_fractal_accel'].iloc[i] +
            0.4 * df['medium_fractal_accel'].iloc[i] +
            0.3 * df['fractal_accel_divergence'].iloc[i] +
            0.2 * df['volume_fractal_accel'].iloc[i]
        )
        
        # Regime-Adaptive Memory Acceleration (30% weight)
        regime_component = (
            0.4 * df['efficiency_accel'].iloc[i] +
            0.3 * df['order_flow_accel'].iloc[i] +
            0.2 * df['intraday_momentum_accel'].iloc[i] +
            0.1 * df['volatility_accel'].iloc[i]
        )
        
        # Range Asymmetry Acceleration (20% weight)
        range_component = (
            0.6 * df['upside_range_accel'].iloc[i] +
            0.4 * df['volume_efficiency_accel'].iloc[i]
        )
        
        # Liquidity Efficiency Acceleration (10% weight)
        liquidity_component = df['volume_efficiency_accel'].iloc[i]
        
        # Dynamic regime adjustment
        current_efficiency = df['price_efficiency'].iloc[i]
        if current_efficiency > df['price_efficiency'].rolling(window=10).mean().iloc[i]:
            # High efficiency regime - amplify signals
            regime_multiplier = 1.2
        else:
            # Low efficiency regime - maintain original strength
            regime_multiplier = 1.0
        
        # Final composite factor
        composite_factor = (
            0.4 * fractal_component +
            0.3 * regime_component +
            0.2 * range_component +
            0.1 * liquidity_component
        ) * regime_multiplier
        
        # Cross-component validation
        acceleration_alignment = (
            np.sign(fractal_component) == np.sign(regime_component) and
            np.sign(fractal_component) == np.sign(range_component)
        )
        
        if acceleration_alignment:
            confidence_multiplier = 1.5
        else:
            confidence_multiplier = 0.7
            
        result.iloc[i] = composite_factor * confidence_multiplier
    
    # Normalize the final result
    result = (result - result.rolling(window=20, min_periods=10).mean()) / result.rolling(window=20, min_periods=10).std()
    
    return result.fillna(0)
