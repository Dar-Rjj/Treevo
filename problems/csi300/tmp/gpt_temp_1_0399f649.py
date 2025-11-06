import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Intraday Momentum Persistence alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize output series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate required components
    # 1. Intraday Price Strength
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Gap classification
    data['gap_magnitude'] = 0
    data.loc[data['gap'].abs() > 0.02, 'gap_magnitude'] = 2  # Large gap
    data.loc[(data['gap'].abs() > 0.005) & (data['gap'].abs() <= 0.02), 'gap_magnitude'] = 1  # Medium gap
    data.loc[data['gap'].abs() <= 0.005, 'gap_magnitude'] = 0  # Small gap
    
    # Morning session momentum
    data['morning_strength'] = (data['high'] - data['open']) / data['open']
    data['morning_weakness'] = (data['open'] - data['low']) / data['open']
    
    # Afternoon session momentum
    data['afternoon_recovery'] = (data['close'] - data['low']) / data['low']
    data['afternoon_decline'] = (data['high'] - data['close']) / data['high']
    
    # Intraday Momentum Score
    data['intraday_momentum'] = (
        np.sign(data['gap']) * data['gap_magnitude'] * 0.3 +
        data['morning_strength'] * 0.4 - 
        data['morning_weakness'] * 0.4 +
        data['afternoon_recovery'] * 0.3 - 
        data['afternoon_decline'] * 0.3
    )
    
    # 2. Volume Distribution Patterns
    # Avoid division by zero
    price_range = data['high'] - data['low']
    price_range = price_range.replace(0, np.nan)
    
    # Hourly volume estimates
    data['morning_volume_est'] = data['volume'] * (data['high'] - data['open']) / price_range
    data['afternoon_volume_est'] = data['volume'] * (data['close'] - data['low']) / price_range
    
    # Fill NaN values with equal distribution
    data['morning_volume_est'] = data['morning_volume_est'].fillna(data['volume'] * 0.5)
    data['afternoon_volume_est'] = data['afternoon_volume_est'].fillna(data['volume'] * 0.5)
    
    # Volume concentration ratio
    data['volume_concentration'] = abs(data['morning_volume_est'] - data['afternoon_volume_est']) / data['volume']
    
    # Volume concentration classification
    data['concentration_level'] = 0
    data.loc[data['volume_concentration'] > 0.6, 'concentration_level'] = 2  # High concentration
    data.loc[(data['volume_concentration'] >= 0.3) & (data['volume_concentration'] <= 0.6), 'concentration_level'] = 1  # Medium
    data.loc[data['volume_concentration'] < 0.3, 'concentration_level'] = 0  # Low concentration
    
    # Volume-Price Divergence
    data['price_direction'] = np.sign(data['close'] - data['open'])
    data['volume_bias'] = np.sign(data['morning_volume_est'] - data['afternoon_volume_est'])
    
    # Divergence strength
    data['divergence_strength'] = 0
    bullish_condition = (data['price_direction'] > 0) & (data['volume_bias'] > 0)
    bearish_condition = (data['price_direction'] < 0) & (data['volume_bias'] < 0)
    
    data.loc[bullish_condition, 'divergence_strength'] = data['concentration_level'] * 0.5
    data.loc[bearish_condition, 'divergence_strength'] = -data['concentration_level'] * 0.5
    
    # 3-day rolling divergence persistence
    data['divergence_persistence'] = data['divergence_strength'].rolling(window=3, min_periods=1).mean()
    
    # Volume Pattern Score
    data['volume_pattern'] = (
        data['divergence_persistence'] * 0.7 +
        data['concentration_level'] * np.sign(data['volume_bias']) * 0.3
    )
    
    # 3. Price Rejection Signals
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_midpoint'] = (data['prev_high'] + data['prev_low']) / 2
    
    # Resistance and support tests
    data['resistance_test'] = data['high'] / data['prev_high'] - 1
    data['support_test'] = data['low'] / data['prev_low'] - 1
    
    # Rejection analysis
    data['rejection_score'] = 0
    
    # Support bounce (positive rejection)
    support_bounce = (data['low'] <= data['prev_low'] * 1.005) & (data['close'] > data['low'] * 1.01)
    data.loc[support_bounce, 'rejection_score'] = (data['close'] - data['low']) / data['low'] * 2
    
    # Resistance rejection (negative rejection)
    resistance_rejection = (data['high'] >= data['prev_high'] * 0.995) & (data['close'] < data['high'] * 0.99)
    data.loc[resistance_rejection, 'rejection_score'] = -(data['high'] - data['close']) / data['high'] * 2
    
    # 4. Combine Multi-Timeframe Signals with time decay
    # Calculate exponential weights (half-life of 3 days)
    weights = np.exp(-np.arange(10) * np.log(2) / 3)
    weights = weights / weights.sum()
    
    # Initialize component arrays
    intraday_scores = []
    volume_scores = []
    rejection_scores = []
    
    # Calculate weighted components
    for i in range(len(data)):
        if i < 10:
            # Use available data with adjusted weights
            available_weights = weights[:i+1] / weights[:i+1].sum()
            intraday_score = np.dot(data['intraday_momentum'].iloc[:i+1].values, available_weights)
            volume_score = np.dot(data['volume_pattern'].iloc[:i+1].values, available_weights)
            rejection_score = np.dot(data['rejection_score'].iloc[:i+1].values, available_weights)
        else:
            intraday_score = np.dot(data['intraday_momentum'].iloc[i-9:i+1].values, weights)
            volume_score = np.dot(data['volume_pattern'].iloc[i-9:i+1].values, weights)
            rejection_score = np.dot(data['rejection_score'].iloc[i-9:i+1].values, weights)
        
        intraday_scores.append(intraday_score)
        volume_scores.append(volume_score)
        rejection_scores.append(rejection_score)
    
    data['weighted_intraday'] = intraday_scores
    data['weighted_volume'] = volume_scores
    data['weighted_rejection'] = rejection_scores
    
    # 5. Generate Composite Alpha Factor
    # Dynamic weights based on component volatility (higher volatility = lower weight)
    intraday_vol = data['weighted_intraday'].rolling(window=20, min_periods=1).std()
    volume_vol = data['weighted_volume'].rolling(window=20, min_periods=1).std()
    rejection_vol = data['weighted_rejection'].rolling(window=20, min_periods=1).std()
    
    # Inverse volatility weighting
    total_vol = intraday_vol + volume_vol + rejection_vol
    intraday_weight = (1 / intraday_vol) / (1 / intraday_vol + 1 / volume_vol + 1 / rejection_vol)
    volume_weight = (1 / volume_vol) / (1 / intraday_vol + 1 / volume_vol + 1 / rejection_vol)
    rejection_weight = (1 / rejection_vol) / (1 / intraday_vol + 1 / volume_vol + 1 / rejection_vol)
    
    # Fill NaN weights with default values
    intraday_weight = intraday_weight.fillna(0.4)
    volume_weight = volume_weight.fillna(0.35)
    rejection_weight = rejection_weight.fillna(0.25)
    
    # Final alpha calculation
    alpha = (
        data['weighted_intraday'] * intraday_weight +
        data['weighted_volume'] * volume_weight +
        data['weighted_rejection'] * rejection_weight
    )
    
    # Scale to 0-3 range based on historical distribution
    rolling_std = alpha.rolling(window=63, min_periods=1).std()
    alpha_scaled = alpha / (rolling_std * 2)  # Scale to approximately ±2.5 standard deviations
    alpha_final = np.clip(alpha_scaled, -3.0, 3.0)
    
    return alpha_final
