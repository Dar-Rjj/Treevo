import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price range
    price_range = df['high'] - df['low']
    price_range = price_range.replace(0, np.nan)  # Avoid division by zero
    
    # Calculate asymmetric pressure
    buying_pressure = (df['close'] - df['low']) / price_range
    selling_pressure = (df['high'] - df['close']) / price_range
    
    # Calculate volume ratio (current volume vs 20-day average)
    volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
    volume_ratio = df['volume'] / volume_ratio
    
    # Weight pressures by volume ratio
    weighted_buying = buying_pressure * volume_ratio
    weighted_selling = selling_pressure * volume_ratio
    
    # Calculate net pressure difference
    pressure_diff = weighted_buying - weighted_selling
    
    # Apply EMA with 8-day span for accumulation with decay
    alpha = 2.0 / (8 + 1)
    pressure_cumulation = pressure_diff.ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate rolling percentiles for mean reversion signal
    rolling_median = pressure_cumulation.rolling(window=20, min_periods=1).median()
    rolling_q25 = pressure_cumulation.rolling(window=20, min_periods=1).quantile(0.25)
    rolling_q75 = pressure_cumulation.rolling(window=20, min_periods=1).quantile(0.75)
    
    # Generate contrarian signal based on position relative to percentiles
    factor = pd.Series(index=df.index, dtype=float)
    
    # When above 75th percentile, expect mean reversion down
    above_upper = pressure_cumulation > rolling_q75
    factor[above_upper] = -1.0
    
    # When below 25th percentile, expect mean reversion up  
    below_lower = pressure_cumulation < rolling_q25
    factor[below_lower] = 1.0
    
    # When in middle range, use normalized distance from median
    middle_range = ~above_upper & ~below_lower
    iqr = rolling_q75 - rolling_q25
    normalized_deviation = (pressure_cumulation - rolling_median) / iqr
    factor[middle_range] = -normalized_deviation[middle_range]
    
    return factor
