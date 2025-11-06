import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily range percentage
    data['range_pct'] = (data['high'] - data['low']) / data['close']
    
    # Calculate Volume-Weighted Price Acceleration for different timeframes
    
    # Short-Term VWPA (5-day)
    data['short_vwp'] = (data['close'].rolling(window=5).apply(lambda x: (x * data.loc[x.index, 'volume']).sum() / data.loc[x.index, 'volume'].sum(), raw=False))
    data['short_price_accel'] = ((data['close'] - data['close'].shift(5)) / data['close'].shift(5)) - ((data['close'].shift(5) - data['close'].shift(10)) / data['close'].shift(10))
    data['short_vwpa'] = data['short_price_accel'] * data['volume'].rolling(window=5).sum()
    
    # Medium-Term VWPA (10-day)
    data['medium_vwp'] = (data['close'].rolling(window=10).apply(lambda x: (x * data.loc[x.index, 'volume']).sum() / data.loc[x.index, 'volume'].sum(), raw=False))
    data['medium_price_accel'] = ((data['close'] - data['close'].shift(10)) / data['close'].shift(10)) - ((data['close'].shift(10) - data['close'].shift(20)) / data['close'].shift(20))
    data['medium_vwpa'] = data['medium_price_accel'] * data['volume'].rolling(window=10).sum()
    
    # Long-Term VWPA (20-day)
    data['long_vwp'] = (data['close'].rolling(window=20).apply(lambda x: (x * data.loc[x.index, 'volume']).sum() / data.loc[x.index, 'volume'].sum(), raw=False))
    data['long_price_accel'] = ((data['close'] - data['close'].shift(20)) / data['close'].shift(20)) - ((data['close'].shift(20) - data['close'].shift(40)) / data['close'].shift(40))
    data['long_vwpa'] = data['long_price_accel'] * data['volume'].rolling(window=20).sum()
    
    # Detect Market Regime
    data['high_range_days'] = data['range_pct'].rolling(window=20).apply(lambda x: (x > 0.02).sum(), raw=False)
    data['low_range_days'] = data['range_pct'].rolling(window=20).apply(lambda x: (x < 0.01).sum(), raw=False)
    data['regime_indicator'] = data['high_range_days'] / (data['high_range_days'] + data['low_range_days'])
    
    # Combine VWPA with regime-adaptive weights
    def calculate_final_factor(row):
        if pd.isna(row['regime_indicator']) or pd.isna(row['short_vwpa']) or pd.isna(row['medium_vwpa']) or pd.isna(row['long_vwpa']):
            return np.nan
        
        regime = row['regime_indicator']
        
        if regime > 0.6:
            # High-Volatility Regime
            short_weight, medium_weight, long_weight = 0.3, 0.4, 0.3
        elif regime < 0.4:
            # Low-Volatility Regime
            short_weight, medium_weight, long_weight = 0.6, 0.3, 0.1
        else:
            # Normal Regime
            short_weight, medium_weight, long_weight = 0.4, 0.4, 0.2
        
        return (row['short_vwpa'] * short_weight + 
                row['medium_vwpa'] * medium_weight + 
                row['long_vwpa'] * long_weight)
    
    # Apply the regime-adaptive weighting
    data['factor'] = data.apply(calculate_final_factor, axis=1)
    
    return data['factor']
