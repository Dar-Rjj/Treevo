import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Efficiency with Regime Persistence factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate Fractal Efficiency Metrics
    # Price Path Efficiency Ratio
    daily_ranges = df['high'] - df['low']
    price_path_length = daily_ranges.rolling(window=20, min_periods=10).sum()
    straight_line_distance = (df['close'] - df['close'].shift(20)).abs()
    price_efficiency = np.where(price_path_length > 0, 
                               straight_line_distance / price_path_length, 0)
    
    # Volume Fractal Dimension (simplified as volume fluctuation complexity)
    volume_returns = df['volume'].pct_change()
    volume_volatility = volume_returns.rolling(window=20, min_periods=10).std()
    volume_mean = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_fractal = np.where(volume_mean > 0, 
                             volume_volatility / volume_mean, 0)
    
    # Efficiency-Volume Divergence
    efficiency_volume_divergence = price_efficiency / (1 + volume_fractal)
    
    # Analyze Regime Persistence Patterns
    # Trend Persistence Score
    returns = df['close'].pct_change()
    signed_returns = np.sign(returns)
    
    # Autocorrelation of signed returns (5-day lag)
    trend_persistence = signed_returns.rolling(window=20, min_periods=10).apply(
        lambda x: x.autocorr(lag=5) if len(x) >= 15 else 0, raw=False
    ).fillna(0)
    
    # Volatility Clustering Strength
    daily_volatility = (df['high'] - df['low']) / df['close']
    vol_persistence = daily_volatility.rolling(window=20, min_periods=10).apply(
        lambda x: x.autocorr(lag=5) if len(x) >= 15 else 0, raw=False
    ).fillna(0)
    
    # Regime Transition Resistance (simplified as volatility regime stability)
    vol_ma_short = daily_volatility.rolling(window=5, min_periods=3).mean()
    vol_ma_long = daily_volatility.rolling(window=20, min_periods=10).mean()
    regime_resistance = 1 / (1 + (vol_ma_short - vol_ma_long).abs())
    
    # Combine regime persistence components
    regime_persistence = (trend_persistence + vol_persistence + regime_resistance) / 3
    
    # Construct Multi-Timeframe Fractal Signals
    # Short-Term (5-day) Fractal Efficiency
    price_path_5d = daily_ranges.rolling(window=5, min_periods=3).sum()
    straight_line_5d = (df['close'] - df['close'].shift(5)).abs()
    efficiency_5d = np.where(price_path_5d > 0, 
                            straight_line_5d / price_path_5d, 0)
    
    volume_vol_5d = volume_returns.rolling(window=5, min_periods=3).std()
    volume_mean_5d = df['volume'].rolling(window=5, min_periods=3).mean()
    volume_fractal_5d = np.where(volume_mean_5d > 0, 
                                volume_vol_5d / volume_mean_5d, 0)
    fractal_5d = efficiency_5d / (1 + volume_fractal_5d)
    
    # Medium-Term (20-day) Fractal Efficiency
    fractal_20d = efficiency_volume_divergence  # Already calculated above
    
    # Fractal Efficiency Momentum
    fractal_momentum = fractal_5d - fractal_20d
    
    # Final Integration
    # Multiply Fractal Efficiency Metrics by Regime Persistence Patterns
    base_signal = efficiency_volume_divergence * regime_persistence
    
    # Scale by Multi-Timeframe Fractal Signals
    scaled_signal = base_signal * (1 + fractal_momentum)
    
    # Apply regime-dependent weighting (higher weight when regimes are persistent)
    regime_weight = 1 + regime_persistence.abs()
    final_signal = scaled_signal * regime_weight
    
    # Generate directional signals based on efficiency-persistence alignment
    # Positive when high efficiency aligns with persistent regimes
    directional_signal = np.sign(efficiency_volume_divergence) * final_signal
    
    result = directional_signal.fillna(0)
    
    return result
