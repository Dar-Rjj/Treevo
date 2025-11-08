import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency Patterns
    # Intraday Efficiency Component
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_trend_5d'] = data['daily_efficiency'].rolling(window=5).mean()
    
    # Medium-Term Efficiency Confirmation
    data['close_10d_trend'] = data['close'].diff(10) / data['close'].shift(10)
    data['efficiency_momentum'] = data['daily_efficiency'] * np.sign(data['close_10d_trend'])
    
    # Accumulated Trading Pressure
    # Daily Pressure Components
    close_change = data['close'].diff()
    data['buying_pressure'] = np.where(close_change > 0, data['close'] - data['low'], 0)
    data['selling_pressure'] = np.where(close_change < 0, data['high'] - data['close'], 0)
    
    # Net Pressure Accumulation
    data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
    data['net_pressure_5d'] = data['net_pressure'].rolling(window=5).sum()
    
    # Volume regression slope for pressure-volume alignment
    def volume_slope(x):
        if len(x) < 2:
            return 0
        try:
            return linregress(range(len(x)), x).slope
        except:
            return 0
    
    data['volume_slope_5d'] = data['volume'].rolling(window=5).apply(volume_slope, raw=True)
    data['pressure_volume_aligned'] = data['net_pressure_5d'] * data['volume_slope_5d']
    
    # Volatility Regime Adjustment
    data['high_low_range'] = data['high'] - data['low']
    data['volatility_20d'] = data['high_low_range'].rolling(window=20).std()
    
    # Average True Range (10-day)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['atr_10d'] = data['true_range'].rolling(window=10).mean()
    
    # Volume-based regime confirmation
    data['volume_trend_10d'] = data['volume'].rolling(window=10).apply(volume_slope, raw=True)
    data['amount_volume_ratio'] = data['amount'] / (data['volume'] + 1e-8)
    data['av_ratio_consistency'] = data['amount_volume_ratio'].rolling(window=10).std()
    
    # Efficiency-Pressure Divergence Detection
    data['efficiency_pressure_divergence'] = np.where(
        (data['efficiency_momentum'] > 0) & (data['net_pressure_5d'] < 0), -1,
        np.where((data['efficiency_momentum'] < 0) & (data['net_pressure_5d'] > 0), 1, 0)
    )
    
    # Volume-confirmed pattern persistence
    data['consecutive_divergence'] = data['efficiency_pressure_divergence'].rolling(window=3).sum()
    data['range_volume_correlation'] = data['high_low_range'].rolling(window=5).corr(data['volume'])
    
    # Combine Components with Volatility-Adaptive Weighting
    # Efficiency-Pressure Interaction
    efficiency_pressure_interaction = data['net_pressure_5d'] * data['efficiency_momentum']
    
    # Volatility regime strength
    volatility_regime = (data['volatility_20d'] + data['atr_10d']) / 2
    volatility_weight = 1 / (volatility_regime + 1e-8)
    
    # Volume confirmation multiplier
    volume_confirmation = np.where(
        data['volume_slope_5d'] * data['efficiency_momentum'] > 0, 
        1.2,  # Enhanced signal when aligned
        np.where(data['volume_slope_5d'] * data['efficiency_momentum'] < 0, 0.8, 1.0)  # Reduced when divergent
    )
    
    # Amount validation for liquidity
    liquidity_factor = np.where(
        data['amount_volume_ratio'] > data['amount_volume_ratio'].rolling(window=20).mean(),
        1.1,  # Higher liquidity
        0.9   # Lower liquidity
    )
    
    # Pattern convergence adjustment
    pattern_convergence = np.where(
        data['consecutive_divergence'].abs() >= 2,
        1.3,  # Strong pattern persistence
        np.where(data['range_volume_correlation'] > 0, 1.1, 1.0)  # Range-volume alignment
    )
    
    # High volatility period reduction
    high_vol_reduction = np.where(
        data['volatility_20d'] > data['volatility_20d'].rolling(window=50).quantile(0.8),
        0.7,  # Reduce weight during high volatility
        1.0
    )
    
    # Final factor calculation
    base_factor = efficiency_pressure_interaction * volatility_weight
    enhanced_factor = base_factor * volume_confirmation * liquidity_factor
    final_factor = enhanced_factor * pattern_convergence * high_vol_reduction
    
    # Return the factor series
    return final_factor
