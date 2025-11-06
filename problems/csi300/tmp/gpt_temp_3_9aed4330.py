import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining asymmetric gap-reversal momentum, 
    volume-clustering price efficiency, bid-ask imbalance momentum, and 
    overnight-momentum carryover patterns.
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate variables
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['abs_gap'] = data['gap'].abs()
    data['gap_direction'] = np.sign(data['gap'])
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['reversal_magnitude'] = data['intraday_return'].abs() * data['abs_gap']
    
    # Gap persistence (consecutive same-direction gaps)
    gap_persistence = pd.Series(index=data.index, dtype=int)
    current_streak = 0
    prev_direction = 0
    
    for i in range(len(data)):
        if pd.isna(data['gap_direction'].iloc[i]) or data['gap_direction'].iloc[i] == 0:
            gap_persistence.iloc[i] = 0
            current_streak = 0
            prev_direction = 0
        elif data['gap_direction'].iloc[i] == prev_direction:
            current_streak += 1
            gap_persistence.iloc[i] = current_streak
        else:
            current_streak = 1
            gap_persistence.iloc[i] = current_streak
            prev_direction = data['gap_direction'].iloc[i]
    
    data['gap_persistence'] = gap_persistence
    
    # Reversal consistency (days with same reversal direction)
    reversal_consistency = pd.Series(index=data.index, dtype=int)
    current_streak = 0
    prev_direction = 0
    
    for i in range(len(data)):
        if pd.isna(data['intraday_return'].iloc[i]) or data['intraday_return'].iloc[i] == 0:
            reversal_consistency.iloc[i] = 0
            current_streak = 0
            prev_direction = 0
        elif np.sign(data['intraday_return'].iloc[i]) == prev_direction:
            current_streak += 1
            reversal_consistency.iloc[i] = current_streak
        else:
            current_streak = 1
            reversal_consistency.iloc[i] = current_streak
            prev_direction = np.sign(data['intraday_return'].iloc[i])
    
    data['reversal_consistency'] = reversal_consistency
    
    # Asymmetric Gap-Reversal Momentum
    gap_reversal_signal = data['reversal_magnitude'] * data['gap_persistence']
    gap_reversal_signal = gap_reversal_signal * data['reversal_consistency']
    gap_reversal_factor = gap_reversal_signal * data['volume']
    
    # Volume-Clustering Price Efficiency
    # Volume percentile over 20 days
    data['volume_percentile'] = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).sum() / len(x[:-1]) if len(x[:-1]) > 0 else 0.5
    )
    
    # Volume clustering (consecutive high-volume days)
    volume_clustering = pd.Series(index=data.index, dtype=int)
    current_cluster = 0
    
    for i in range(len(data)):
        if pd.isna(data['volume_percentile'].iloc[i]):
            volume_clustering.iloc[i] = 0
            current_cluster = 0
        elif data['volume_percentile'].iloc[i] > 0.7:
            current_cluster += 1
            volume_clustering.iloc[i] = current_cluster
        else:
            current_cluster = 0
            volume_clustering.iloc[i] = 0
    
    data['volume_clustering'] = volume_clustering
    
    # Price efficiency calculations
    data['price_noise'] = (data['high'] - data['low']) / data['close']
    data['price_change'] = data['close'].pct_change()
    data['efficiency_ratio'] = data['price_change'].abs() / data['price_noise'].replace(0, np.nan)
    data['efficiency_ratio'] = data['efficiency_ratio'].fillna(0)
    
    # 5-day return variance for trend smoothness
    data['trend_smoothness'] = data['close'].pct_change().rolling(window=5, min_periods=1).std()
    data['trend_smoothness'] = 1 / (1 + data['trend_smoothness'])  # Inverse for smoothness measure
    
    # Volume-clustering efficiency factor
    volume_efficiency_factor = data['volume_clustering'] * data['efficiency_ratio']
    volume_efficiency_factor = volume_efficiency_factor * (data['volume'] / data['volume'].shift(1)).fillna(1)
    
    # Recent return directional bias
    recent_return = data['close'].pct_change(3).fillna(0)
    volume_efficiency_factor = volume_efficiency_factor * np.sign(recent_return)
    
    # Bid-Ask Imbalance Momentum
    # Effective spread estimation
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['effective_spread'] = 2 * (data['close'] - data['mid_price']).abs() / data['close']
    
    # Spread momentum and volatility
    data['spread_momentum'] = data['effective_spread'] / data['effective_spread'].shift(5).replace(0, np.nan)
    data['spread_momentum'] = data['spread_momentum'].fillna(1)
    data['spread_volatility'] = data['effective_spread'].rolling(window=5, min_periods=1).std()
    
    # Buying pressure
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['buying_pressure'] = data['buying_pressure'].fillna(0.5)
    
    # Pressure persistence
    pressure_persistence = pd.Series(index=data.index, dtype=int)
    current_pressure_streak = 0
    prev_pressure_high = False
    
    for i in range(len(data)):
        if pd.isna(data['buying_pressure'].iloc[i]):
            pressure_persistence.iloc[i] = 0
            current_pressure_streak = 0
            prev_pressure_high = False
        elif (data['buying_pressure'].iloc[i] > 0.7 and prev_pressure_high) or \
             (data['buying_pressure'].iloc[i] < 0.3 and not prev_pressure_high):
            current_pressure_streak += 1
            pressure_persistence.iloc[i] = current_pressure_streak
        else:
            current_pressure_streak = 1
            pressure_persistence.iloc[i] = current_pressure_streak
            prev_pressure_high = data['buying_pressure'].iloc[i] > 0.5
    
    data['pressure_persistence'] = pressure_persistence
    
    # Pressure intensity
    data['pressure_intensity'] = data['buying_pressure'] * data['volume']
    
    # Imbalance factor
    imbalance_factor = data['spread_momentum'] * data['pressure_intensity']
    imbalance_factor = imbalance_factor * data['pressure_persistence']
    imbalance_factor = imbalance_factor * data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Overnight-Momentum Carryover
    data['overnight_return'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['overnight_momentum'] = data['overnight_return'].rolling(window=5, min_periods=1).sum()
    data['overnight_volatility'] = data['overnight_return'].rolling(window=10, min_periods=1).std()
    
    # Momentum carryover and persistence
    data['momentum_carryover'] = data['overnight_return'] * data['intraday_return']
    
    momentum_persistence = pd.Series(index=data.index, dtype=int)
    current_momentum_streak = 0
    prev_carryover_direction = 0
    
    for i in range(len(data)):
        if pd.isna(data['momentum_carryover'].iloc[i]) or data['momentum_carryover'].iloc[i] == 0:
            momentum_persistence.iloc[i] = 0
            current_momentum_streak = 0
            prev_carryover_direction = 0
        elif np.sign(data['momentum_carryover'].iloc[i]) == prev_carryover_direction:
            current_momentum_streak += 1
            momentum_persistence.iloc[i] = current_momentum_streak
        else:
            current_momentum_streak = 1
            momentum_persistence.iloc[i] = current_momentum_streak
            prev_carryover_direction = np.sign(data['momentum_carryover'].iloc[i])
    
    data['momentum_persistence'] = momentum_persistence
    
    # Carryover factor
    carryover_factor = data['momentum_carryover'] * data['momentum_persistence']
    carryover_factor = carryover_factor * (1 + data['overnight_volatility'])
    carryover_factor = carryover_factor * data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    
    # Combine all factors with equal weighting
    combined_factor = (
        gap_reversal_factor.fillna(0) + 
        volume_efficiency_factor.fillna(0) + 
        imbalance_factor.fillna(0) + 
        carryover_factor.fillna(0)
    )
    
    # Normalize the final factor
    factor = (combined_factor - combined_factor.rolling(window=20, min_periods=1).mean()) / \
             combined_factor.rolling(window=20, min_periods=1).std().replace(0, 1)
    
    return factor.fillna(0)
