import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Multi-Timeframe Price Efficiency
    # Intraday Efficiency
    intraday_return = df['close'] - df['open']
    intraday_range = df['high'] - df['low']
    intraday_efficiency = intraday_return / intraday_range
    
    # Overnight Efficiency
    overnight_return = df['open'] - df['close'].shift(1)
    overnight_range = df['high'].shift(1) - df['low'].shift(1)
    overnight_efficiency = overnight_return / overnight_range
    
    # Assess Directional Persistence
    # Determine Efficiency Regime
    regime_conditions = [
        (intraday_efficiency > 0) & (overnight_efficiency > 0),  # Positive
        (intraday_efficiency < 0) & (overnight_efficiency < 0),  # Negative
        True  # Mixed
    ]
    regime_values = [1, -1, 0]
    regime = pd.Series(np.select(regime_conditions, regime_values), index=df.index)
    
    # Calculate Regime Persistence
    regime_change = regime != regime.shift(1)
    regime_group = regime_change.cumsum()
    regime_persistence = regime.groupby(regime_group).cumcount() + 1
    
    # Incorporate Volume Dynamics
    # Volume-to-Amount Ratio
    volume_efficiency = df['volume'] / df['amount']
    
    # Volume Persistence
    volume_trend = volume_efficiency.rolling(window=3).apply(
        lambda x: 1 if (x.diff().dropna() > 0).all() else (-1 if (x.diff().dropna() < 0).all() else 0),
        raw=False
    )
    
    # Generate Composite Alpha Factor
    # Combine price regime persistence with volume confirmation
    composite_factor = regime_persistence * np.sign(regime) * volume_trend
    
    # Volatility-Regime Adjustment
    volatility = df['close'].rolling(window=5).std()
    adjusted_factor = composite_factor / volatility
    
    return adjusted_factor
