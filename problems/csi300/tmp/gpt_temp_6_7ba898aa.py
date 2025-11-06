import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap-Based Momentum Signal
    # Calculate Overnight Price Gap
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Calculate Intraday Momentum Persistence
    data['daily_range'] = data['high'] - data['low']
    data['intraday_persistence'] = (data['close'] - data['open']) / np.where(data['daily_range'] == 0, 1, data['daily_range'])
    
    # Generate Gap Momentum Factor
    data['gap_momentum'] = data['overnight_gap'] * data['intraday_persistence']
    
    # Volume Acceleration Confirmation
    # Calculate Volume Momentum
    data['volume_ma_3'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(2) - 1
    
    # Calculate Volume-Range Efficiency
    data['volume_range_efficiency'] = data['daily_range'] / np.where(data['volume'] == 0, 1, data['volume'])
    
    # Apply Volume Confirmation
    data['volume_confirmed_momentum'] = data['gap_momentum'] * data['volume_acceleration'] * data['volume_range_efficiency']
    
    # Market Regime Detection
    # Identify Volatility Regime
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR and volatility regime
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    data['atr_20_median'] = data['true_range'].rolling(window=20, min_periods=1).median()
    data['volatility_regime'] = np.where(data['atr_10'] > data['atr_20_median'], 1.2, 0.8)  # High vol: amplify, low vol: dampen
    
    # Detect Trend Environment
    # Calculate 15-day price slope using linear regression
    def calculate_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    data['close_15_slope'] = data['close'].rolling(window=15, min_periods=1).apply(calculate_slope, raw=False)
    data['trend_strength'] = np.where(abs(data['close_15_slope']) > data['close'].rolling(window=15, min_periods=1).std() * 0.1, 1.3, 0.9)
    
    # Regime-Adaptive Weighting
    data['regime_adjusted_factor'] = data['volume_confirmed_momentum'] * data['volatility_regime'] * data['trend_strength']
    
    # Clean up intermediate columns
    result = data['regime_adjusted_factor'].copy()
    
    return result
