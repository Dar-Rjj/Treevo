import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Multi-Timeframe Reversal Pattern
    # Short-term Reversal: (High_{t-2} - Close_t) / High_{t-2}
    short_term_reversal = (data['high'].shift(2) - data['close']) / data['high'].shift(2)
    
    # Medium-term Reversal: (Close_t - Close_{t-5}) / Close_{t-5}
    medium_term_reversal = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Reversal Divergence: Short-term Reversal - Medium-term Reversal
    reversal_divergence = short_term_reversal - medium_term_reversal
    
    # Volatility Regime Filter
    # Short-term Volatility: (High_t - Low_t) / Close_{t-1}
    short_term_vol = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Medium-term Volatility: std(Close_{t-4:t}) / mean(Close_{t-4:t})
    medium_term_vol = data['close'].rolling(window=5).std() / data['close'].rolling(window=5).mean()
    
    # Volatility Regime Shift: Short-term Volatility / Medium-term Volatility
    volatility_regime_shift = short_term_vol / medium_term_vol
    
    # Volume Regime Filter
    # Volume Concentration: Volume_t / sum(Volume_{t-4:t})
    volume_concentration = data['volume'] / data['volume'].rolling(window=5).sum()
    
    # Volume Persistence: count of days with Volume > median(Volume_{t-19:t}) over past 5 days
    volume_median = data['volume'].rolling(window=20).median()
    volume_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            window_data = data['volume'].iloc[i-4:i+1]
            window_median = volume_median.iloc[i]
            volume_persistence.iloc[i] = (window_data > window_median).sum()
        else:
            volume_persistence.iloc[i] = np.nan
    
    # Volume Regime Strength: Volume Concentration × Volume Persistence
    volume_regime_strength = volume_concentration * volume_persistence
    
    # Final Alpha Factor: Regime-Filtered Reversal
    alpha_factor = reversal_divergence * volatility_regime_shift * volume_regime_strength
    
    return alpha_factor
