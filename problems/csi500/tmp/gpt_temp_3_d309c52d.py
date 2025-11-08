import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Component
    # Short-Term Momentum
    mom_5d = df['close'] / df['close'].shift(5) - 1
    mom_3d = df['close'] / df['close'].shift(3) - 1
    
    # Medium-Term Momentum
    mom_10d = df['close'] / df['close'].shift(10) - 1
    mom_20d = df['close'] / df['close'].shift(20) - 1
    
    # Combined momentum score (weighted average)
    momentum_score = 0.4 * mom_5d + 0.3 * mom_3d + 0.2 * mom_10d + 0.1 * mom_20d
    
    # Volume Confirmation
    # Volume Trend
    vol_5d_avg = df['volume'].rolling(window=5).mean()
    vol_10d_avg = df['volume'].rolling(window=10).mean()
    
    # Volume-to-Price Ratio
    dollar_volume = df['close'] * df['volume']
    dollar_volume_5d_avg = dollar_volume.rolling(window=5).mean()
    volume_ratio = dollar_volume / dollar_volume_5d_avg
    
    # Volume confirmation score
    volume_score = 0.6 * (df['volume'] / vol_5d_avg) + 0.4 * (df['volume'] / vol_10d_avg) + 0.3 * volume_ratio
    
    # Overnight Gap Analysis
    # Overnight return
    overnight_return = df['open'] / df['close'].shift(1) - 1
    
    # Gap persistence
    gap_sign = np.sign(overnight_return)
    gap_consistency = gap_sign.rolling(window=5).apply(lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 5 else np.nan, raw=False)
    
    # Gap magnitude vs intraday range
    daily_range = df['high'] - df['low']
    gap_magnitude_ratio = np.abs(overnight_return) / (daily_range / df['close'].shift(1))
    
    # Gap analysis score
    gap_score = 0.5 * gap_consistency + 0.5 * gap_magnitude_ratio
    
    # Intraday Efficiency
    # Price Efficiency Ratio
    close_to_close_return = np.abs(df['close'] / df['close'].shift(1) - 1)
    price_efficiency = close_to_close_return / (daily_range / df['close'])
    
    # Volume Efficiency
    # Volume concentration (proxy using volume per unit price movement)
    volume_concentration = df['volume'] / (daily_range + 1e-8)
    
    # Volume persistence
    volume_persistence = df['volume'] / vol_5d_avg
    
    # Volume efficiency score
    volume_efficiency = 0.6 * volume_concentration + 0.4 * volume_persistence
    
    # Combined Efficiency Score
    efficiency_score = 0.7 * price_efficiency + 0.3 * volume_efficiency
    
    # Final alpha factor: Combine all components
    alpha_factor = (
        0.35 * momentum_score +
        0.25 * volume_score +
        0.20 * gap_score +
        0.20 * efficiency_score
    )
    
    return alpha_factor
