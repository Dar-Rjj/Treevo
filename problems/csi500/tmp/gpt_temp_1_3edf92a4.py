import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Convergence Momentum Alpha Factor
    Combines price momentum, volume convergence, and volatility context
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Price Momentum Components
    # Short-Term Momentum (1-3 days)
    data['price_return'] = data['close'] - data['close'].shift(1)
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['price_acceleration'] = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    
    # Medium-Term Momentum (5-10 days)
    data['cumulative_return_5d'] = data['close'] - data['close'].shift(5)
    
    # Trend Consistency: count of positive returns in last 5 days
    data['positive_returns_count'] = data['price_return'].rolling(window=5, min_periods=3).apply(
        lambda x: (x > 0).sum(), raw=True
    )
    
    # Price Stability
    data['price_stability'] = (data['high'] - data['low']) / (data['cumulative_return_5d'].abs() + 1e-8)
    
    # Volume Convergence Analysis
    # Volume Trend Components
    data['volume_direction'] = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['volume_magnitude_change'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    
    # Volume Streak: consecutive days with same direction
    data['volume_streak'] = 0
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            data['volume_streak'].iloc[i] = data['volume_streak'].iloc[i-1] + 1
        else:
            data['volume_streak'].iloc[i] = 1
    
    # Volume-Price Alignment
    data['direction_agreement'] = (
        np.sign(data['price_return']) == np.sign(data['volume'] - data['volume'].shift(1))
    ).astype(int)
    
    # Alignment Persistence: consecutive days with agreement
    data['alignment_persistence'] = 0
    for i in range(1, len(data)):
        if data['direction_agreement'].iloc[i] == 1:
            data['alignment_persistence'].iloc[i] = data['alignment_persistence'].iloc[i-1] + 1
        else:
            data['alignment_persistence'].iloc[i] = 0
    
    # Strength of Agreement
    data['strength_agreement'] = data['price_return'].abs() * (data['volume'] - data['volume'].shift(1)).abs()
    
    # Volume Regime Detection
    data['recent_volume_avg'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_comparison'] = data['volume'] / data['recent_volume_avg'].replace(0, np.nan)
    data['volume_classification'] = (data['volume'] > data['recent_volume_avg']).astype(int)
    
    # Volatility Context
    # Range-Based Volatility
    data['daily_range'] = data['high'] - data['low']
    data['recent_volatility'] = data['daily_range'].rolling(window=3, min_periods=2).mean()
    data['volatility_change'] = data['daily_range'] / data['recent_volatility'].replace(0, np.nan)
    
    # Volatility-Adjusted Momentum
    data['range_normalized_return'] = data['price_return'] / data['daily_range'].replace(0, np.nan)
    data['volatility_scaled_acceleration'] = data['price_acceleration'] / data['recent_volatility'].replace(0, np.nan)
    data['stability_measure'] = data['cumulative_return_5d'] / data['daily_range'].rolling(window=5, min_periods=3).sum().replace(0, np.nan)
    
    # Factor Construction
    for i in range(5, len(data)):
        # Core Momentum Signal
        short_term_weight = data['range_normalized_return'].iloc[i] if not pd.isna(data['range_normalized_return'].iloc[i]) else 0
        medium_term_weight = data['stability_measure'].iloc[i] if not pd.isna(data['stability_measure'].iloc[i]) else 0
        
        # Multi-Timeframe Blend
        combined_momentum = 0.6 * short_term_weight + 0.4 * medium_term_weight
        
        # Acceleration Adjustment
        if not pd.isna(data['price_acceleration'].iloc[i]):
            if data['price_acceleration'].iloc[i] > 0:
                combined_momentum *= 1.2
            elif data['price_acceleration'].iloc[i] < 0:
                combined_momentum *= 0.8
        
        # Volume Confirmation
        volume_boost = 1.0
        if data['direction_agreement'].iloc[i] == 1:
            if data['alignment_persistence'].iloc[i] >= 2:
                # Strong Agreement
                volume_boost = 1.3
            else:
                # Moderate Agreement
                volume_boost = 1.1
            
            # Volume Regime Scaling
            if not pd.isna(data['volume_comparison'].iloc[i]):
                if data['volume_comparison'].iloc[i] > 1.2:
                    volume_boost *= 1.3  # High Volume
                elif data['volume_comparison'].iloc[i] > 0.8:
                    volume_boost *= 1.1  # Normal Volume
                else:
                    volume_boost *= 0.9  # Low Volume
        
        # Volatility Context
        volatility_scaling = 1.0
        if not pd.isna(data['volatility_change'].iloc[i]):
            if data['volatility_change'].iloc[i] < 0.8:
                volatility_scaling = 1.4  # Low Volatility
            elif data['volatility_change'].iloc[i] > 1.2:
                volatility_scaling = 0.7  # High Volatility
        
        # Trend Confidence
        trend_confidence = 1.0
        if not pd.isna(data['positive_returns_count'].iloc[i]) and not pd.isna(data['volatility_change'].iloc[i]):
            if data['positive_returns_count'].iloc[i] >= 4 and data['volatility_change'].iloc[i] < 0.8:
                trend_confidence = 1.2  # High Confidence
            elif data['positive_returns_count'].iloc[i] <= 1:
                trend_confidence = 0.8  # Low Confidence
        
        # Final Composite Signal
        final_signal = combined_momentum * volume_boost * volatility_scaling * trend_confidence
        
        result.iloc[i] = final_signal
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
