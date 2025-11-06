import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Efficiency Divergence Component
    # Intraday vs Medium-Term Momentum Divergence
    data['intraday_momentum_efficiency'] = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    data['five_day_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_divergence'] = data['intraday_momentum_efficiency'] - data['five_day_momentum']
    
    # Range Efficiency Momentum
    data['daily_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Calculate rolling correlation between range efficiency and 1-day returns
    data['one_day_returns'] = data['close'].pct_change()
    rolling_corr = pd.Series(index=data.index, dtype=float)
    for i in range(10, len(data)):
        window_data = data.iloc[i-10:i]
        corr = window_data['daily_range_efficiency'].corr(window_data['one_day_returns'])
        rolling_corr.iloc[i] = corr
    data['range_efficiency_corr'] = rolling_corr
    
    # Track correlation momentum as 5-day change in rolling correlation
    data['correlation_momentum'] = data['range_efficiency_corr'] - data['range_efficiency_corr'].shift(5)
    
    # Volume Efficiency Confirmation
    # Volume-Range Relationship Analysis
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['five_day_volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Calculate volume-range correlation
    volume_range_corr = pd.Series(index=data.index, dtype=float)
    for i in range(10, len(data)):
        window_data = data.iloc[i-10:i]
        corr = window_data['volume_efficiency'].corr(window_data['one_day_returns'])
        volume_range_corr.iloc[i] = corr
    data['volume_range_correlation'] = volume_range_corr
    
    # Amount-Weighted Validation
    data['amount_efficiency'] = data['amount'] / (data['high'] - data['low'])
    data['volume_amount_ratio'] = data['volume_efficiency'] / data['amount_efficiency']
    
    # Efficiency-Weighted Signal Integration
    # Volatility-Adjusted Momentum
    data['returns_volatility'] = data['one_day_returns'].rolling(window=20).std()
    data['volatility_adjusted_momentum'] = data['correlation_momentum'] * data['returns_volatility']
    
    # Volume-Confirmed Divergence Synthesis
    data['volume_confirmation_strength'] = data['volume_range_correlation'] * data['five_day_volume_momentum']
    data['volume_confirmed_divergence'] = data['momentum_divergence'] * data['volume_confirmation_strength']
    
    # Multi-Timeframe Validation
    # Short vs Medium-Term Efficiency Comparison
    data['five_day_range_efficiency_avg'] = data['daily_range_efficiency'].rolling(window=5).mean()
    data['twenty_day_range_efficiency_avg'] = data['daily_range_efficiency'].rolling(window=20).mean()
    data['efficiency_momentum_shift'] = data['five_day_range_efficiency_avg'] - data['twenty_day_range_efficiency_avg']
    
    # Volume Pattern Consistency Check
    data['five_day_volume_efficiency_avg'] = data['volume_efficiency'].rolling(window=5).mean()
    data['twenty_day_volume_efficiency_avg'] = data['volume_efficiency'].rolling(window=20).mean()
    data['volume_efficiency_trend'] = data['five_day_volume_efficiency_avg'] - data['twenty_day_volume_efficiency_avg']
    data['volume_amount_stability'] = data['volume_amount_ratio'].rolling(window=10).std()
    
    # Final factor calculation combining all components
    data['factor'] = (
        data['volume_confirmed_divergence'] * 
        data['volatility_adjusted_momentum'] * 
        data['efficiency_momentum_shift'] * 
        (1 / (1 + data['volume_amount_stability']))
    )
    
    return data['factor']
