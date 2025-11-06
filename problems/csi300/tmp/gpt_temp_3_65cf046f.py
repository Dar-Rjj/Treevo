import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Multi-Period Price Acceleration
    # Short-Term Price Acceleration (3-day momentum of 5-day momentum)
    price_5d_return_t = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    price_5d_return_t3 = (data['close'].shift(3) - data['close'].shift(8)) / data['close'].shift(8)
    price_accel_5d = price_5d_return_t - price_5d_return_t3
    
    # Medium-Term Price Acceleration (5-day momentum of 10-day momentum)
    price_10d_return_t = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    price_10d_return_t5 = (data['close'].shift(5) - data['close'].shift(15)) / data['close'].shift(15)
    price_accel_10d = price_10d_return_t - price_10d_return_t5
    
    # Calculate Multi-Period Volume Acceleration
    # Short-Term Volume Acceleration (3-day momentum of 5-day momentum)
    volume_5d_return_t = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    volume_5d_return_t3 = (data['volume'].shift(3) - data['volume'].shift(8)) / (data['volume'].shift(8) + 1e-8)
    volume_accel_5d = volume_5d_return_t - volume_5d_return_t3
    
    # Medium-Term Volume Acceleration (5-day momentum of 10-day momentum)
    volume_10d_return_t = (data['volume'] - data['volume'].shift(10)) / (data['volume'].shift(10) + 1e-8)
    volume_10d_return_t5 = (data['volume'].shift(5) - data['volume'].shift(15)) / (data['volume'].shift(15) + 1e-8)
    volume_accel_10d = volume_10d_return_t - volume_10d_return_t5
    
    # Analyze Acceleration Divergence Patterns
    # Short-Term Divergence Score
    short_term_divergence = np.where(
        price_accel_5d > volume_accel_5d, 1, 
        np.where(price_accel_5d < volume_accel_5d, -1, 0)
    )
    
    # Medium-Term Divergence Score
    medium_term_divergence = np.where(
        price_accel_10d > volume_accel_10d, 1,
        np.where(price_accel_10d < volume_accel_10d, -1, 0)
    )
    
    # Identify Market Regime
    # Calculate daily returns
    daily_returns = data['close'].pct_change()
    
    # Calculate Bidirectional Volatility (30-day window)
    upside_volatility = daily_returns.rolling(window=30, min_periods=15).apply(
        lambda x: x[x > 0].std() if len(x[x > 0]) > 0 else 0
    )
    downside_volatility = daily_returns.rolling(window=30, min_periods=15).apply(
        lambda x: abs(x[x < 0]).std() if len(x[x < 0]) > 0 else 0
    )
    
    # Calculate Volatility Asymmetry
    volatility_asymmetry = (upside_volatility / (downside_volatility + 1e-8)) - 1
    
    # Assign Regime Flags
    bull_regime = volatility_asymmetry > 0.2
    bear_regime = volatility_asymmetry < -0.2
    neutral_regime = ~bull_regime & ~bear_regime
    
    # Regime-Adaptive Signal Adjustment
    regime_multiplier = np.ones_like(volatility_asymmetry)
    regime_multiplier = np.where(bull_regime, 1.5, regime_multiplier)
    regime_multiplier = np.where(bear_regime, 0.7, regime_multiplier)
    
    # Add momentum bias for bull regime
    momentum_bias = np.where(bull_regime, price_accel_5d.abs() * 0.1, 0)
    
    # Apply mean reversion for bear regime
    mean_reversion = np.where(bear_regime, -price_accel_5d * 0.05, 0)
    
    # Small noise filter for neutral regime
    volume_accel_vol = volume_accel_5d.rolling(window=10, min_periods=5).std()
    noise_filter = np.where(neutral_regime, 1 / (1 + volume_accel_vol.abs()), 1)
    
    # Incorporate Liquidity Dynamics
    # Calculate Volume-Weighted Price Impact
    daily_price_range = (data['high'] - data['low']) / data['close']
    volume_weighted_impact = daily_price_range / (data['volume'] + 1e-8)
    vwap_impact_10d = volume_weighted_impact.rolling(window=10, min_periods=5).mean()
    
    # Normalize by Historical Context
    vwap_median_60d = vwap_impact_10d.rolling(window=60, min_periods=30).median()
    liquidity_pressure = (vwap_impact_10d / (vwap_median_60d + 1e-8)) - 1
    
    # Apply Liquidity Adjustment (inverse relationship)
    liquidity_multiplier = 1 / (1 + liquidity_pressure.abs())
    
    # Generate Composite Alpha Factor
    # Calculate Acceleration Strength
    avg_accel_magnitude = (price_accel_5d.abs() + price_accel_10d.abs()) / 2
    
    # Combine Divergence Signals
    total_divergence_score = short_term_divergence + medium_term_divergence
    
    # Final Factor Calculation
    regime_adjusted_signal = (total_divergence_score + momentum_bias + mean_reversion) * regime_multiplier * noise_filter
    final_factor = regime_adjusted_signal * avg_accel_magnitude * liquidity_multiplier
    
    return pd.Series(final_factor, index=data.index)
