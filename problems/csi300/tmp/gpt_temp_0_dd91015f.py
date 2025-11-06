import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Efficiency Component
    # Directional Price Movement
    price_direction = df['close'].diff().abs()
    
    # Total Price Oscillation
    price_oscillation = df['high'] - df['low']
    
    # Price Efficiency Ratio
    price_efficiency = price_direction / price_oscillation.replace(0, np.nan)
    price_efficiency = price_efficiency.fillna(0)
    
    # Calculate Volume Confirmation Component
    # Abnormal Volume Pattern
    vwap = df['amount'] / df['volume'].replace(0, np.nan)
    price_impact = (df['close'] - vwap).abs() / vwap
    abnormal_volume = df['volume'] * price_impact
    
    # Volume Breakout Events
    volume_median = df['volume'].rolling(window=10, min_periods=5).median()
    volume_breakout = (df['volume'] > volume_median * 1.5).astype(float)
    
    # Volume Confirmation Signal
    volume_confirmation = abnormal_volume * volume_breakout
    
    # Identify Market Regime Context
    # Volatility Regime
    daily_ranges = (df['high'] - df['low']) / df['close'].shift(1)
    volatility_cluster = daily_ranges.rolling(window=10, min_periods=5).std()
    
    # Trend vs Range-Bound Detection
    close_returns = df['close'].pct_change()
    directional_persistence = close_returns.rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x > 0) / len(x) if len(x) > 0 else 0.5
    )
    trend_strength = (directional_persistence - 0.5).abs() * 2
    
    # Regime Adjustment Factor
    volatility_weight = 1 / (1 + volatility_cluster * 10)
    trend_weight = 1 + trend_strength
    regime_factor = volatility_weight * trend_weight
    
    # Generate Contextual Alpha Signal
    # Combine Efficiency and Volume Components
    combined_signal = price_efficiency * volume_confirmation
    
    # Apply Regime-Based Adjustment
    final_factor = combined_signal * regime_factor
    
    # Clean and return
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return final_factor
