import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Reversal with Volume Confirmation
    # Calculate Short-Term Momentum (5-day price change)
    short_term_momentum = data['close'].pct_change(periods=5)
    
    # Calculate Medium-Term Reversal (15-day price change)
    medium_term_reversal = -data['close'].pct_change(periods=15)
    
    # Combine Momentum and Reversal
    momentum_reversal = short_term_momentum * medium_term_reversal
    
    # Volume Confirmation (5-day volume trend)
    volume_trend = data['volume'].pct_change(periods=5)
    factor1 = momentum_reversal * volume_trend
    
    # Volatility Regime Adjusted Price Impact
    # Calculate High-Low Range
    high_low_range = (data['high'] - data['low']) / data['close']
    
    # Compute Volume-Price Relationship
    volume_price_relationship = data['amount'] / data['close']
    
    # Assess Volatility Regime (rolling standard deviation of returns)
    returns = data['close'].pct_change()
    volatility_regime = returns.rolling(window=20, min_periods=10).std()
    
    # Adjust Price Impact
    factor2 = (volume_price_relationship / high_low_range) * volatility_regime
    
    # Liquidity-Adjusted Momentum Divergence
    # Compute Price Momentum (difference between 10-day and 20-day returns)
    momentum_10d = data['close'].pct_change(periods=10)
    momentum_20d = data['close'].pct_change(periods=20)
    momentum_divergence = momentum_10d - momentum_20d
    
    # Assess Liquidity Conditions
    volume_volatility = data['volume'].rolling(window=10, min_periods=5).std()
    turnover_rate = data['amount'] / data['close']
    liquidity_indicator = volume_volatility * turnover_rate
    
    # Adjust for Liquidity
    factor3 = momentum_divergence * liquidity_indicator
    
    # Opening Gap Mean Reversion Factor
    # Calculate Opening Gap
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Compute Historical Gap Behavior (rolling mean reversion probability)
    gap_reversion = (opening_gap.rolling(window=10, min_periods=5).apply(
        lambda x: np.mean(np.sign(x) != np.sign(x.shift(1))) if len(x.dropna()) > 0 else np.nan
    ))
    
    # Volume Confirmation
    volume_ratio = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    factor4 = opening_gap * gap_reversion * volume_ratio
    
    # Trend Acceleration with Volume Breakout
    # Calculate Price Trend Acceleration (second derivative of moving average)
    ma_5 = data['close'].rolling(window=5, min_periods=3).mean()
    ma_10 = data['close'].rolling(window=10, min_periods=5).mean()
    trend_acceleration = (ma_5 - ma_10).diff()
    
    # Identify Volume Breakouts
    volume_ma = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_breakout = (data['volume'] - volume_ma) / volume_ma
    
    # Combine Signals
    factor5 = trend_acceleration * volume_breakout
    
    # Relative Strength with Sector Momentum
    # Calculate Stock-Specific Momentum (recent return)
    stock_momentum = data['close'].pct_change(periods=5)
    
    # Assess Sector Momentum (using rolling mean as sector proxy)
    sector_momentum = data['close'].pct_change(periods=5).rolling(window=20, min_periods=10).mean()
    
    # Compute Relative Strength
    relative_strength = (stock_momentum - sector_momentum) * (data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean())
    factor6 = relative_strength
    
    # Intraday Pressure Accumulation Factor
    # Calculate Buying/Selling Pressure
    intraday_pressure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Accumulate Pressure Over Time (5 days)
    accumulated_pressure = intraday_pressure.rolling(window=5, min_periods=3).sum()
    
    # Volume Intensity Assessment
    volume_intensity = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    factor7 = accumulated_pressure * volume_intensity
    
    # Price-Efficiency Ratio with Volume Clustering
    # Calculate Price Efficiency
    true_range = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    price_efficiency = (data['close'] - data['open']) / true_range.replace(0, np.nan)
    
    # Detect Volume Clustering (volume concentration ratio)
    volume_clustering = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Combine Efficiency and Clustering
    factor8 = price_efficiency * volume_clustering
    
    # Combine all factors with equal weighting
    combined_factor = (
        factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
        factor4.fillna(0) + factor5.fillna(0) + factor6.fillna(0) + 
        factor7.fillna(0) + factor8.fillna(0)
    ) / 8
    
    return combined_factor
