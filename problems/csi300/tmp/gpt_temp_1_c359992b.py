import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate multiple alpha factors using OHLCV data with volume adjustments and contrarian signals.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Asymmetric Price Movement Factor
    intraday_range = data['high'] - data['low']
    abs_price_change = abs(data['close'] - data['open'])
    range_vs_change = abs_price_change / intraday_range.replace(0, np.nan)
    factor1 = range_vs_change * data['volume']
    
    # Factor 2: Volume-Adjusted Price Reversal
    prev_day_return = data['close'].shift(1) / data['close'].shift(2) - 1
    volume_change = data['volume'] / data['volume'].shift(1) - 1
    factor2 = -prev_day_return * volume_change  # Negative sign for contrarian effect
    
    # Factor 3: Efficiency Ratio Momentum (5-day period)
    n_period = 5
    net_change = data['close'] - data['close'].shift(n_period)
    
    # Calculate total price movement (sum of absolute daily changes)
    total_movement = 0
    for i in range(n_period):
        total_movement += abs(data['close'].shift(i) - data['close'].shift(i+1))
    
    efficiency_ratio = net_change / total_movement.replace(0, np.nan)
    avg_volume = data['volume'].rolling(window=n_period).mean()
    factor3 = efficiency_ratio * avg_volume
    
    # Factor 4: Pressure Accumulation Factor
    buying_pressure = data['close'] - data['low']
    selling_pressure = data['high'] - data['close']
    pressure_ratio = buying_pressure / selling_pressure.replace(0, np.nan)
    factor4 = np.log(pressure_ratio.replace([np.inf, -np.inf], np.nan)) * data['volume']
    
    # Factor 5: Volatility-Regulated Return
    short_term_return = data['close'] / data['close'].shift(1) - 1
    
    # Calculate 20-day volatility
    returns_20d = data['close'].pct_change().rolling(window=20)
    volatility = returns_20d.std()
    
    vol_adjusted_return = short_term_return / volatility.replace(0, np.nan)
    
    # Volume confirmation
    volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    factor5 = vol_adjusted_return * volume_ratio
    
    # Factor 6: Gap Exploitation Factor
    overnight_gap = data['open'] / data['close'].shift(1) - 1
    
    # Calculate gap fulfillment
    gap_filled = np.zeros(len(data))
    for i in range(len(data)):
        if overnight_gap.iloc[i] > 0:  # Positive gap
            # Check if low price went below previous close (gap filled)
            if data['low'].iloc[i] <= data['close'].shift(1).iloc[i]:
                gap_filled[i] = min(1, abs((data['close'].shift(1).iloc[i] - data['low'].iloc[i]) / 
                                         (data['open'].iloc[i] - data['close'].shift(1).iloc[i])))
        else:  # Negative gap
            # Check if high price went above previous close (gap filled)
            if data['high'].iloc[i] >= data['close'].shift(1).iloc[i]:
                gap_filled[i] = min(1, abs((data['high'].iloc[i] - data['close'].shift(1).iloc[i]) / 
                                         (data['close'].shift(1).iloc[i] - data['open'].iloc[i])))
    
    gap_signal = overnight_gap * (1 - gap_filled)
    
    # Volume intensity adjustment
    volume_anomaly = data['volume'] / data['volume'].rolling(window=10).mean()
    factor6 = gap_signal * volume_anomaly
    
    # Combine factors (equal weighting for simplicity)
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + 
                      factor4.fillna(0) + factor5.fillna(0) + factor6.fillna(0)) / 6
    
    return combined_factor
