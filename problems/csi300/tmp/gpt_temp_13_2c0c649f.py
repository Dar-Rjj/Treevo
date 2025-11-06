import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor: Micro-Structure Momentum with Dynamic Volume-Pressure Efficiency
    Captures immediate price momentum enhanced by dynamic volume-pressure alignment and 
    micro-structure efficiency signals, with adaptive directional confirmation
    """
    # Micro-term price momentum (1-day)
    price_momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Dynamic volume pressure: volume surge relative to adaptive recent window (2-5 days)
    volume_window = np.where(price_momentum.abs() > price_momentum.rolling(5).std(), 2, 5)
    volume_pressure = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        if i >= volume_window[i]:
            volume_pressure.iloc[i] = df['volume'].iloc[i] / df['volume'].iloc[i-volume_window[i]:i].mean() - 1
    
    # Micro-structure directional efficiency: closing efficiency with momentum confirmation
    range_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    directional_efficiency = range_efficiency * np.sign(price_momentum)
    
    # Opening gap momentum with micro-structure context
    gap_strength = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_efficiency = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    gap_momentum = gap_strength * gap_efficiency
    
    # Immediate volume-pressure alignment with directional confirmation
    volume_alignment = volume_pressure * directional_efficiency
    
    # Micro-volatility efficiency: intraday range relative to price movement
    price_range = df['high'] - df['low']
    micro_volatility = price_range / df['open']
    volatility_efficiency = np.abs(price_momentum) / (micro_volatility + 1e-7)
    
    # Amount flow intensity with micro-structure timing
    avg_trade_size = df['amount'] / (df['volume'] + 1e-7)
    trade_size_intensity = avg_trade_size / avg_trade_size.rolling(3).mean() - 1
    
    # Combined factor: micro-momentum enhanced by dynamic volume-pressure alignment,
    # gap momentum efficiency, and trade size intensity, scaled by volatility efficiency
    factor = (price_momentum + gap_momentum + trade_size_intensity) * (1 + volume_alignment) * volatility_efficiency
    
    return factor
