import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate all required components
    # Range and absolute change for asymmetric intraday movement
    data['range'] = data['high'] - data['low']
    data['abs_change'] = abs(data['close'] - data['open'])
    
    # Volume-weighted ratio for asymmetric intraday movement
    data['vw_ratio'] = np.where(data['range'] > 0, 
                               data['abs_change'] / data['range'] * data['volume'], 
                               0)
    
    # Prior return and volume change for volume-confirmed reversal
    data['prior_return'] = data['close'].shift(1) / data['close'].shift(2) - 1
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['contrarian_signal'] = -1 * data['prior_return'] * data['volume_change']
    
    # Efficiency momentum components
    n_period = 5
    data['net_change'] = data['close'] - data['close'].shift(n_period)
    
    # Calculate total movement over n_period
    data['close_ret'] = data['close'].pct_change()
    data['total_movement'] = abs(data['close_ret']).rolling(window=n_period+1).sum()
    data['avg_volume'] = data['volume'].rolling(window=n_period).mean()
    data['efficiency_ratio'] = np.where(data['total_movement'] > 0,
                                       (data['net_change'] / data['close'].shift(n_period)) / data['total_movement'] * data['avg_volume'],
                                       0)
    
    # Pressure imbalance components
    data['buying_pressure'] = data['close'] - data['low']
    data['selling_pressure'] = data['high'] - data['close']
    data['pressure_ratio'] = np.where((data['buying_pressure'] > 0) & (data['selling_pressure'] > 0),
                                     np.log(data['buying_pressure'] / data['selling_pressure']) * data['volume'],
                                     0)
    
    # Volatility-normalized return components
    data['daily_return'] = data['close'] / data['close'].shift(1) - 1
    data['volatility'] = data['daily_return'].rolling(window=20).std()
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['vol_norm_return'] = np.where(data['volatility'] > 0,
                                      (data['daily_return'] / data['volatility']) * data['volume_ratio'],
                                      0)
    
    # Gap fade signal components
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_filled'] = np.where(data['overnight_gap'] > 0,
                                 (data['low'] - data['close'].shift(1)) / data['close'].shift(1),
                                 (data['high'] - data['close'].shift(1)) / data['close'].shift(1))
    data['fulfillment'] = np.minimum(1, abs(data['gap_filled']) / abs(data['overnight_gap']))
    data['gap_signal'] = data['overnight_gap'] * (1 - data['fulfillment']) * data['volume_ratio']
    
    # Combine all factors with equal weights
    factors_to_combine = [
        'vw_ratio',
        'contrarian_signal', 
        'efficiency_ratio',
        'pressure_ratio',
        'vol_norm_return',
        'gap_signal'
    ]
    
    # Ensure we have enough data for all calculations
    min_periods = max(20, n_period + 2)
    
    for i in range(len(data)):
        if i >= min_periods:
            # Combine normalized versions of each factor
            combined_value = 0
            valid_factors = 0
            
            for factor_col in factors_to_combine:
                if not pd.isna(data[factor_col].iloc[i]):
                    # Normalize by recent volatility of the factor
                    recent_std = data[factor_col].iloc[i-10:i+1].std()
                    if recent_std > 0:
                        normalized_value = data[factor_col].iloc[i] / recent_std
                        combined_value += normalized_value
                        valid_factors += 1
            
            if valid_factors > 0:
                factor.iloc[i] = combined_value / valid_factors
    
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor
