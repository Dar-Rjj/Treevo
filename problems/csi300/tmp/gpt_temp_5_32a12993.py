import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Weighted Range Momentum with Volume-Price Divergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Daily High-Low Range Computation
    data['daily_range'] = data['high'] - data['low']
    
    # Range Momentum Signals
    data['range_momentum_5d'] = data['daily_range'] / data['daily_range'].shift(5) - 1
    data['range_momentum_10d'] = data['daily_range'] / data['daily_range'].shift(10) - 1
    data['range_change_ratio'] = data['daily_range'] / data['daily_range'].shift(1)
    
    # Range Momentum Patterns
    data['range_momentum_diff'] = data['range_momentum_5d'] - data['range_momentum_10d']
    data['range_acceleration'] = data['range_momentum_5d'] - data['range_momentum_5d'].shift(1)
    
    # Volatility Measurement
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = np.maximum(np.maximum(tr1, tr2), tr3)
    data['atr_20'] = data['true_range'].rolling(window=20).mean()
    
    # Range Efficiency
    data['range_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['daily_range']
    data['volatility_persistence'] = data['atr_20'] / data['atr_20'].shift(5)
    
    # Volume Trend Analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['volume_ratio'] = data['volume_5d_avg'] / data['volume_10d_avg']
    
    # Volume Slope (5-day)
    data['volume_slope'] = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(5), x, 1)[0] if len(x) == 5 else np.nan
    )
    
    # Volume Stability (Coefficient of Variation)
    data['volume_cv'] = data['volume'].rolling(window=10).std() / data['volume'].rolling(window=10).mean()
    
    # Price Trend Analysis
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_trend_direction'] = np.sign(data['price_momentum_5d'])
    
    # Divergence Detection
    data['bullish_divergence'] = ((data['price_momentum_5d'] < 0) & (data['volume_slope'] > 0)).astype(int)
    data['bearish_divergence'] = ((data['price_momentum_5d'] > 0) & (data['volume_slope'] < 0)).astype(int)
    
    # Volatility-Based Weighting
    data['volatility_weight'] = 1 / (1 + data['volatility_persistence'])
    data['efficiency_weight'] = data['range_efficiency']
    
    # Volatility-Enhanced Momentum
    data['vol_adj_range_momentum_5d'] = data['range_momentum_5d'] * data['volatility_weight'] * data['efficiency_weight']
    data['vol_adj_range_momentum_10d'] = data['range_momentum_10d'] * data['volatility_weight'] * data['efficiency_weight']
    
    # Volume-Momentum Integration
    data['volume_momentum_integration'] = (
        data['vol_adj_range_momentum_5d'] * data['volume_ratio'] * np.sign(data['volume_slope'])
    )
    
    # Divergence-Enhanced Signals
    data['divergence_enhancement'] = (
        data['bullish_divergence'] * 1.0 + 
        data['bearish_divergence'] * (-1.0)
    )
    
    # Range Momentum Strength Assessment
    data['momentum_persistence'] = data['range_momentum_5d'].rolling(window=5).std()
    data['signal_quality'] = 1 / (1 + data['momentum_persistence'])
    
    # Multi-Signal Combination
    data['combined_range_momentum'] = (
        data['vol_adj_range_momentum_5d'] * 0.6 + 
        data['vol_adj_range_momentum_10d'] * 0.4
    )
    
    data['divergence_adjusted_momentum'] = (
        data['combined_range_momentum'] * (1 + 0.3 * data['divergence_enhancement'])
    )
    
    data['volume_integrated_signal'] = (
        data['divergence_adjusted_momentum'] * data['volume_momentum_integration']
    )
    
    # Final alpha factor with non-linear transformation
    data['alpha_factor'] = np.tanh(
        data['volume_integrated_signal'] * data['signal_quality'] * (1 / (1 + data['volume_cv']))
    )
    
    # Return the final alpha factor series
    return data['alpha_factor']
