import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function for consecutive count
    def consecutive_count(series):
        count = pd.Series(0, index=series.index, dtype=float)
        current_streak = 0
        for i in range(len(series)):
            if i == 0:
                count.iloc[i] = 0
            else:
                if series.iloc[i]:
                    current_streak += 1
                else:
                    current_streak = 0
                count.iloc[i] = current_streak
        return count
    
    # Cross-Association Momentum components
    # Price-Volume Association
    price_volume_corr = data['close'].rolling(window=5).corr(data['volume'])
    volume_price_corr = data['volume'].rolling(window=5).corr(data['close'])
    
    # High-Low Association
    price_range = data['high'] - data['low']
    price_change = abs(data['close'] - data['close'].shift(1))
    range_efficiency = price_range / np.where(price_change == 0, 1, price_change)
    
    gap_condition = data['high'] > data['close'].shift(1)
    gap_persistence = consecutive_count(gap_condition)
    
    # Multi-timeframe Association
    short_term_vwap = (data['close'].rolling(window=3).apply(lambda x: (x * data.loc[x.index, 'volume']).sum() / 
                                                           data.loc[x.index, 'volume'].sum(), raw=False))
    medium_term_vwap = (data['close'].rolling(window=8).apply(lambda x: (x * data.loc[x.index, 'volume']).sum() / 
                                                            data.loc[x.index, 'volume'].sum(), raw=False))
    
    # Volatility-Dampened Volume components
    # Volume Stability
    volume_std = data['volume'].rolling(window=5).std()
    volume_mean = data['volume'].rolling(window=5).mean()
    volume_smoothness = 1 / (volume_std / np.where(volume_mean == 0, 1, volume_mean))
    
    volume_momentum = (data['volume'] - data['volume'].shift(5)) / np.where(data['volume'].shift(5) == 0, 1, data['volume'].shift(5))
    
    # Price Volatility Dampening
    dampened_volume = data['volume'] / np.where(price_range == 0, 1, price_range)
    volatility_adjusted_volume = data['volume'] / np.where(price_change == 0, 1, price_change)
    
    # Regime Consistency
    volume_regime_condition = data['volume'] > data['volume'].shift(1)
    volume_regime = consecutive_count(volume_regime_condition)
    
    volatility_regime_condition = price_range > (data['high'].shift(1) - data['low'].shift(1))
    volatility_regime = consecutive_count(volatility_regime_condition)
    
    # Factor Integration
    # Cross-momentum score
    cross_momentum_score = (price_volume_corr + volume_price_corr) * range_efficiency * (gap_persistence + 1)
    
    # Volume stability score
    volume_stability_score = volume_smoothness * volume_momentum * dampened_volume
    
    # Final alpha
    final_alpha = cross_momentum_score * volume_stability_score * volatility_adjusted_volume
    
    return final_alpha
