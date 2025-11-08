import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum with Volume-Price Convergence alpha factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    data['ultra_short_momentum'] = (data['close'] / data['close'].shift(2)) - 1
    data['short_term_momentum'] = (data['close'] / data['close'].shift(5)) - 1
    data['medium_term_momentum'] = (data['close'] / data['close'].shift(10)) - 1
    data['long_term_momentum'] = (data['close'] / data['close'].shift(20)) - 1
    
    # Adaptive Volatility Scaling
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['rolling_volatility'] = data['daily_range'].rolling(window=5).std()
    data['volatility_regime_median'] = data['rolling_volatility'].rolling(window=20).median()
    
    # Volatility regime classification
    data['volatility_regime'] = np.where(
        data['rolling_volatility'] > data['volatility_regime_median'], 
        'high', 
        'low'
    )
    
    # Regime-specific scaling
    def scale_return(return_series, volatility_series, regime_series):
        scaled_returns = []
        for i in range(len(return_series)):
            if pd.isna(return_series.iloc[i]) or pd.isna(volatility_series.iloc[i]):
                scaled_returns.append(np.nan)
            elif regime_series.iloc[i] == 'high':
                scaling_factor = 1 / volatility_series.iloc[i] if volatility_series.iloc[i] != 0 else 1
                scaled_returns.append(return_series.iloc[i] * scaling_factor)
            else:  # low volatility
                scaling_factor = 1 + volatility_series.iloc[i]
                scaled_returns.append(return_series.iloc[i] * scaling_factor)
        return pd.Series(scaled_returns, index=return_series.index)
    
    data['ultra_short_scaled'] = scale_return(data['ultra_short_momentum'], 
                                            data['rolling_volatility'], 
                                            data['volatility_regime'])
    data['short_term_scaled'] = scale_return(data['short_term_momentum'], 
                                           data['rolling_volatility'], 
                                           data['volatility_regime'])
    data['medium_term_scaled'] = scale_return(data['medium_term_momentum'], 
                                            data['rolling_volatility'], 
                                            data['volatility_regime'])
    data['long_term_scaled'] = scale_return(data['long_term_momentum'], 
                                          data['rolling_volatility'], 
                                          data['volatility_regime'])
    
    # Volume-Price Convergence Analysis
    data['volume_momentum'] = (data['volume'] / data['volume'].shift(3)) - 1
    data['volume_mean_5d'] = data['volume'].rolling(window=5).mean()
    
    # Volume regime classification
    conditions = [
        (data['volume'] > data['volume_mean_5d']) & (data['volume_momentum'] > 0.1),
        (data['volume_momentum'].abs() <= 0.1),
        (data['volume'] < data['volume_mean_5d']) & (data['volume_momentum'] < -0.1)
    ]
    choices = ['high', 'normal', 'low']
    data['volume_regime'] = np.select(conditions, choices, default='normal')
    
    # Convergence Scoring
    data['price_direction'] = np.sign(data['short_term_momentum'])
    data['volume_direction'] = np.sign(data['volume_momentum'])
    data['alignment_score'] = data['price_direction'] * data['volume_direction']
    
    data['price_strength'] = data['short_term_momentum'].abs()
    data['volume_strength'] = data['volume_momentum'].abs()
    
    def calculate_strength_ratio(price_str, vol_str):
        min_val = np.minimum(price_str, vol_str)
        max_val = np.maximum(price_str, vol_str)
        return np.where(max_val != 0, min_val / max_val, 0)
    
    data['strength_ratio'] = calculate_strength_ratio(data['price_strength'], data['volume_strength'])
    
    # Dynamic Factor Integration
    # Momentum Aggregation
    timeframe_weights = {
        'ultra_short_scaled': 0.25,
        'short_term_scaled': 0.30,
        'medium_term_scaled': 0.25,
        'long_term_scaled': 0.20
    }
    
    data['combined_momentum'] = (
        data['ultra_short_scaled'] * timeframe_weights['ultra_short_scaled'] +
        data['short_term_scaled'] * timeframe_weights['short_term_scaled'] +
        data['medium_term_scaled'] * timeframe_weights['medium_term_scaled'] +
        data['long_term_scaled'] * timeframe_weights['long_term_scaled']
    )
    
    # Volume-Price Enhancement
    regime_multipliers = {
        'high': 1.5,
        'normal': 1.0,
        'low': 0.7
    }
    
    data['regime_multiplier'] = data['volume_regime'].map(regime_multipliers)
    data['enhanced_factor'] = data['combined_momentum'] * data['regime_multiplier'] * data['alignment_score']
    
    # Final Alpha Construction
    data['amplified_factor'] = data['enhanced_factor'] * (1 + data['strength_ratio'])
    
    # Volume Momentum Adjustment
    data['final_alpha'] = data['amplified_factor'] * (
        1 + np.sign(data['amplified_factor']) * 0.1 * data['volume_momentum'].abs()
    )
    
    return data['final_alpha']
