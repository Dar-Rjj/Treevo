import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate daily returns for reference
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        current = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Skip if we don't have previous day data
        if pd.isna(prev['close']) or pd.isna(prev['volume']):
            continue
            
        # 1. Intraday Range-Pressure Divergence
        range_eff = abs(current['close'] - current['open']) / (current['high'] - current['low'])
        pressure_div = (current['high'] - current['open']) - (current['close'] - current['low'])
        divergence_magnitude = range_eff * pressure_div
        volume_confirmation = (current['volume'] / prev['volume']) * divergence_magnitude
        
        # 2. Amount-Volume Breakout Efficiency
        avg_price = (current['high'] + current['low']) / 2
        trading_eff = abs(current['amount'] / current['volume'] - avg_price) / (current['high'] - current['low'])
        breakout_strength = (current['close'] - avg_price) / (current['high'] - current['low'])
        combined_signal = trading_eff * breakout_strength
        volume_acceleration = (current['volume'] / prev['volume']) * combined_signal
        
        # 3. Overnight Gap Anchoring Momentum
        overnight_gap = (current['open'] - prev['close']) / prev['close']
        intraday_momentum = (current['close'] - current['open']) / current['open']
        
        # Handle division by zero for anchoring ratio
        if abs(overnight_gap) > 1e-8:
            anchoring_ratio = abs(intraday_momentum) / abs(overnight_gap)
        else:
            anchoring_ratio = 0
            
        volume_weighted_factor = (current['volume'] / prev['volume']) * anchoring_ratio * intraday_momentum
        
        # Combine all three components with equal weighting
        combined_factor = (volume_confirmation + volume_acceleration + volume_weighted_factor) / 3
        
        factor.iloc[i] = combined_factor
    
    # Fill NaN values with 0
    factor = factor.fillna(0)
    
    return factor
