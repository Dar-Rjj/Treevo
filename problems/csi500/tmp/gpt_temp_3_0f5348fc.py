import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adjusted Momentum Efficiency with Volume Persistence
    """
    # Compute Multi-Timeframe Price Efficiency
    # Intraday Efficiency Ratio
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Short-Term Trend Efficiency
    close_momentum = df['close'].rolling(window=3).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    high_low_range = (df['high'] - df['low']).rolling(window=3).mean()
    trend_efficiency = close_momentum / high_low_range
    trend_efficiency = trend_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Combine Efficiency Signals
    price_efficiency = (intraday_efficiency + trend_efficiency) / 2
    
    # Assess Volume Persistence Patterns
    # Volume Momentum
    volume_pct_change = df['volume'].pct_change(periods=5)
    
    # Detect consecutive volume increases/decreases
    volume_direction = np.sign(df['volume'].diff())
    consecutive_volume = volume_direction.rolling(window=5).apply(
        lambda x: np.sum(x == x.iloc[-1]) if x.iloc[-1] != 0 else 0, raw=False
    )
    volume_momentum = volume_pct_change * consecutive_volume
    
    # Volume Breakout Regimes
    median_volume_20d = df['volume'].rolling(window=20).median()
    volume_regime = np.where(df['volume'] > median_volume_20d, 1.0, 0.5)
    
    # Combine Volume Signals
    volume_persistence = volume_momentum * volume_regime
    
    # Blend Efficiency with Volume Confirmation
    # Efficiency-Volume Interaction
    efficiency_volume_interaction = price_efficiency * volume_persistence
    
    # Apply Volatility Scaling
    close_volatility = df['close'].pct_change().rolling(window=10).std()
    volatility_scaled_signal = efficiency_volume_interaction / (close_volatility + 1e-8)
    
    # Generate Final Alpha Factor with Regime-Aware Adjustments
    # Use volatility level to weight signal strength
    volatility_regime = close_volatility.rolling(window=20).apply(
        lambda x: 2.0 if x.iloc[-1] > x.quantile(0.7) else 1.0 if x.iloc[-1] > x.quantile(0.3) else 0.5, raw=False
    )
    
    # Final factor with regime adjustments
    final_factor = volatility_scaled_signal * volatility_regime
    
    return final_factor
