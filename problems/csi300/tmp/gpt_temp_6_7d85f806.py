import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Acceleration with Range Efficiency Regime factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Dynamics
    # Short-Term Momentum (2-day)
    short_term_momentum = data['close'] / data['close'].shift(2) - 1
    
    # Medium-Term Momentum (5-day)
    medium_term_momentum = data['close'] / data['close'].shift(5) - 1
    
    # Momentum Acceleration Ratio
    momentum_acceleration = short_term_momentum / (medium_term_momentum + 1e-8)
    
    # Volume Acceleration Analysis
    # 2-day volume change rate
    volume_2day = data['volume'] / (data['volume'].shift(2) + 1e-8) - 1
    
    # 5-day volume change rate
    volume_5day = data['volume'] / (data['volume'].shift(5) + 1e-8) - 1
    
    # Volume acceleration ratio
    volume_acceleration = volume_2day / (volume_5day + 1e-8)
    
    # Volume Sensitivity Ratio by Price Movement
    daily_returns = data['close'].pct_change()
    
    # Create masks for up and down days
    up_days = daily_returns > 0
    down_days = daily_returns < 0
    
    # Calculate rolling average volume for up and down days
    up_volume_avg = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: np.mean(x[up_days.loc[x.index].values]) if up_days.loc[x.index].any() else 1
    )
    
    down_volume_avg = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: np.mean(x[down_days.loc[x.index].values]) if down_days.loc[x.index].any() else 1
    )
    
    volume_sensitivity_ratio = up_volume_avg / (down_volume_avg + 1e-8)
    
    # Range Efficiency Analysis
    # Daily Range Efficiency
    daily_range = data['high'] - data['low']
    daily_range = np.where(daily_range == 0, 1e-8, daily_range)
    range_efficiency = (data['close'] - data['open']) / daily_range
    
    # Range Efficiency Persistence
    range_efficiency_3day = range_efficiency.rolling(window=3, min_periods=1).mean()
    range_efficiency_10day = range_efficiency.rolling(window=10, min_periods=1).mean()
    
    range_efficiency_persistence = range_efficiency_3day / (range_efficiency_10day + 1e-8)
    
    # Volatility Regime Detection
    price_range_5day = (data['high'].rolling(window=5, min_periods=1).max() - 
                       data['low'].rolling(window=5, min_periods=1).min()) / data['close']
    
    price_range_10day = (data['high'].rolling(window=10, min_periods=1).max() - 
                        data['low'].rolling(window=10, min_periods=1).min()) / data['close']
    
    volatility_regime = price_range_5day / (price_range_10day + 1e-8)
    
    # Generate Composite Alpha Factor
    # Combine momentum acceleration with volume sensitivity
    momentum_volume_component = momentum_acceleration * volume_sensitivity_ratio
    
    # Apply directional adjustment based on momentum regime
    momentum_direction = np.sign(short_term_momentum)
    momentum_volume_component = momentum_volume_component * momentum_direction
    
    # Integrate range efficiency with regime context
    regime_adjusted_efficiency = range_efficiency_persistence * volatility_regime
    
    # Final composite factor
    alpha_factor = momentum_volume_component * regime_adjusted_efficiency
    
    # Clean and normalize the factor
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(method='ffill')
    
    return alpha_factor
