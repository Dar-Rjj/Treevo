import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate short-term momentum
    close = df['close']
    momentum_5d = close.pct_change(periods=5)
    momentum_10d = close.pct_change(periods=10)
    
    # Calculate price volatility using high-low range
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    volatility_20d = daily_range.rolling(window=20).mean()
    
    # Calculate volume trend
    volume = df['volume']
    volume_trend_5d = volume.rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Create divergence signal
    momentum_divergence = momentum_5d - momentum_10d
    
    # Adjust by inverse volatility
    volatility_adjusted = momentum_divergence / (volatility_20d + 1e-8)
    
    # Add volume confirmation
    volume_confirmation = np.sign(volume_trend_5d) * np.sign(momentum_divergence)
    volume_confirmation = np.where(volume_confirmation > 0, 1.0, 0.5)
    
    # Final factor
    factor = volatility_adjusted * volume_confirmation
    
    return pd.Series(factor, index=df.index)
