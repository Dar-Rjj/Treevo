import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Weighted Price Reversal Factor
    Combines price reversal patterns with volatility weighting and volume confirmation
    """
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Compute Price Volatility
    # Daily range volatility: (High_t - Low_t) / Close_t
    daily_range_vol = (data['high'] - data['low']) / data['close']
    
    # Rolling volatility: StdDev(Close_{t-9} to Close_t)
    rolling_vol = data['close'].rolling(window=10).std()
    
    # 2. Compute Price Reversal Patterns
    # Gap reversal: (Open_t - Close_{t-1}) / Close_{t-1}
    gap_reversal = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Intraday reversal: (Close_t - Open_t) / Open_t
    intraday_reversal = (data['close'] - data['open']) / data['open']
    
    # 3. Compute Volume Confirmation
    # Volume spike detection: Volume_t / Average(Volume_{t-4} to Volume_{t-1})
    volume_avg_4d = data['volume'].shift(1).rolling(window=4).mean()
    volume_spike = data['volume'] / volume_avg_4d
    
    # Volume persistence: Count(Volume > Average(Volume_{t-9} to Volume_{t-1}) for past 3 days)
    volume_avg_9d = data['volume'].shift(1).rolling(window=9).mean()
    volume_persistence = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 3:
            # Check last 3 days (excluding current day)
            recent_volumes = data['volume'].iloc[i-3:i]
            recent_avg_9d = volume_avg_9d.iloc[i-3:i]
            volume_persistence.iloc[i] = (recent_volumes > recent_avg_9d).sum()
        else:
            volume_persistence.iloc[i] = np.nan
    
    # 4. Combine Components
    # Volatility-adjusted gap: Gap reversal / Daily range volatility
    volatility_adjusted_gap = gap_reversal / daily_range_vol
    
    # Volume-confirmed intraday: Intraday reversal × Volume spike detection
    volume_confirmed_intraday = intraday_reversal * volume_spike
    
    # Stability filter: Rolling volatility × Volume persistence
    stability_filter = rolling_vol * volume_persistence
    
    # Final factor: Volatility-adjusted gap + Volume-confirmed intraday - Stability filter
    factor = volatility_adjusted_gap + volume_confirmed_intraday - stability_filter
    
    return factor
