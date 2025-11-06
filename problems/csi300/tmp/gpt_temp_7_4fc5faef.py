import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Volatility-Weighted Price Reversal
    # Raw Reversal Signal
    data['raw_reversal'] = (data['open'] - data['close']) / data['open']
    
    # Adaptive Volatility Scaling
    # Daily range calculation
    data['daily_range'] = data['high'] - data['low']
    
    # Short-term volatility (5-day average range)
    data['short_term_vol'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    
    # Long-term volatility (20-day average range)
    data['long_term_vol'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    
    # Volatility ratio with square root transformation
    data['vol_ratio'] = np.sqrt(data['short_term_vol'] / data['long_term_vol'])
    
    # Apply volatility-weighted scaling
    data['vol_weighted_reversal'] = data['raw_reversal'] * data['vol_ratio']
    
    # Calculate Liquidity Acceleration
    # Volume momentum - linear slope over 5 days
    def linear_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    # Volume acceleration - second derivative approximation
    def volume_acceleration(volume_series):
        if len(volume_series) < 3:
            return np.nan
        # First differences
        first_diff = volume_series.diff()
        # Second differences (acceleration)
        second_diff = first_diff.diff()
        return np.abs(second_diff.iloc[-1]) if not pd.isna(second_diff.iloc[-1]) else 0
    
    # Calculate volume acceleration
    volume_acc_values = []
    for i in range(len(data)):
        if i >= 4:
            window_vol = data['volume'].iloc[i-4:i+1]
            acc = volume_acceleration(window_vol)
            volume_acc_values.append(acc)
        else:
            volume_acc_values.append(np.nan)
    data['volume_acceleration'] = volume_acc_values
    
    # Price-Volume Efficiency
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['weighted_return'] = data['intraday_return'] * data['volume']
    
    # 5-day average weighted return
    data['avg_weighted_return_5d'] = data['weighted_return'].rolling(window=5, min_periods=3).mean()
    
    # Efficiency deviation
    data['efficiency_deviation'] = data['weighted_return'] / data['avg_weighted_return_5d']
    
    # Combine volume and efficiency signals with cube root
    data['liquidity_acceleration'] = np.cbrt(data['volume_acceleration'] * data['efficiency_deviation'])
    
    # Combine Reversal with Liquidity Signals
    data['combined_signal'] = data['vol_weighted_reversal'] * data['liquidity_acceleration']
    
    # Apply Consistency Filter
    # Volume-Price Alignment check
    def check_alignment(row):
        if pd.isna(row['volume_acceleration']) or pd.isna(row['vol_weighted_reversal']):
            return 1.0
        # Positive correlation: both positive or both negative
        if (row['volume_acceleration'] > 0 and row['vol_weighted_reversal'] > 0) or \
           (row['volume_acceleration'] < 0 and row['vol_weighted_reversal'] < 0):
            return 1.5  # Strengthen signal
        else:
            return 0.7  # Weaken signal
    
    alignment_factors = []
    for idx, row in data.iterrows():
        alignment_factors.append(check_alignment(row))
    data['alignment_factor'] = alignment_factors
    
    # Volatility stability filter
    data['volatility_60d'] = data['daily_range'].rolling(window=60, min_periods=30).std()
    current_vol = data['daily_range'].rolling(window=5, min_periods=3).std()
    
    def volatility_filter(current_vol, historical_vol):
        if pd.isna(current_vol) or pd.isna(historical_vol):
            return 1.0
        vol_ratio = current_vol / historical_vol
        if vol_ratio > 2.0:  # Extreme volatility
            return 0.3
        elif vol_ratio > 1.5:  # High volatility
            return 0.7
        else:  # Normal volatility
            return 1.0
    
    vol_filter_values = []
    for i in range(len(data)):
        if i >= 60:
            cv = current_vol.iloc[i] if not pd.isna(current_vol.iloc[i]) else np.nan
            hv = data['volatility_60d'].iloc[i] if not pd.isna(data['volatility_60d'].iloc[i]) else np.nan
            vol_filter_values.append(volatility_filter(cv, hv))
        else:
            vol_filter_values.append(1.0)
    data['volatility_filter'] = vol_filter_values
    
    # Final alpha factor with filters applied
    data['alpha_factor'] = data['combined_signal'] * data['alignment_factor'] * data['volatility_filter']
    
    return data['alpha_factor']
