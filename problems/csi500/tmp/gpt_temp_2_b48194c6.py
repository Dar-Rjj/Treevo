import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['abs_momentum_5d'] = data['momentum_5d'].abs()
    data['abs_momentum_10d'] = data['momentum_10d'].abs()
    
    # Momentum Acceleration
    data['momentum_ratio'] = (data['close'] / data['close'].shift(5)) / (data['close'] / data['close'].shift(10))
    data['momentum_acceleration'] = (data['momentum_ratio'] - 
                                   2 * data['momentum_ratio'].shift(1) + 
                                   data['momentum_ratio'].shift(2))
    
    # Intraday Efficiency Measurement
    data['range_hl'] = data['high'] - data['low']
    data['range_hc'] = (data['high'] - data['close'].shift(1)).abs()
    data['range_lc'] = (data['low'] - data['close'].shift(1)).abs()
    data['true_range'] = pd.concat([data['range_hl'], data['range_hc'], data['range_lc']], axis=1).max(axis=1)
    data['range_efficiency'] = (data['close'] - data['open']).abs() / data['true_range']
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume-Range Divergence Analysis
    data['volume_ma5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma10'] = data['volume'].rolling(window=10).mean()
    data['volume_divergence'] = (data['volume'] / data['volume'].shift(5)) - (data['volume_ma5'] / data['volume_ma10'])
    
    # Volume persistence streak
    data['volume_persistence'] = 0
    for i in range(1, len(data)):
        if data['volume'].iloc[i] > data['volume'].iloc[i-1]:
            data['volume_persistence'].iloc[i] = data['volume_persistence'].iloc[i-1] + 1
        else:
            data['volume_persistence'].iloc[i] = 0
    
    # Range Divergence
    data['true_range_ma5'] = data['true_range'].rolling(window=5).mean()
    data['true_range_ma10'] = data['true_range'].rolling(window=10).mean()
    data['range_divergence'] = (data['true_range'] / data['true_range_ma5']) - (data['true_range'] / data['true_range_ma10'])
    
    # Divergence Confirmation
    data['divergence_confirmation'] = np.sign(data['volume_divergence']) * np.sign(data['range_divergence'])
    data['divergence_magnitude'] = (data['volume_divergence'].abs() + data['range_divergence'].abs()) / 2
    
    # Market Regime Context
    data['daily_return'] = data['close'].pct_change()
    data['volatility_20d'] = data['daily_return'].rolling(window=20).std()
    data['volatility_regime'] = (data['volatility_20d'] > data['volatility_20d'].rolling(window=50).mean()).astype(int)
    
    data['daily_amplitude'] = (data['high'] - data['low']) / data['close']
    data['amplitude_regime'] = (data['daily_amplitude'] > data['daily_amplitude'].rolling(window=20).mean()).astype(int)
    
    # Trend Regime using linear regression slope
    def linear_trend(series, window=10):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window-1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['price_slope_10d'] = linear_trend(data['close'], 10)
    data['trend_regime'] = (data['price_slope_10d'] > 0).astype(int)
    
    # Gap and Breakout Integration
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    data['volume_breakout'] = (data['volume'] > 1.5 * data['volume'].rolling(window=20).mean()).astype(int)
    
    data['range_breakout'] = ((data['true_range'] / data['close'].shift(1)) > 
                             1.5 * (data['true_range'] / data['close'].shift(1)).rolling(window=20).mean()).astype(int)
    
    # Composite Alpha Signal Generation
    # Core Momentum-Efficiency Component
    data['momentum_efficiency'] = data['momentum_acceleration'] * data['range_efficiency']
    
    # Volume-Range Divergence Integration
    data['volume_weighted_acceleration'] = data['momentum_acceleration'] * data['volume_divergence']
    
    # Regime-Adaptive Adjustment
    data['volatility_scaling'] = np.where(data['volatility_regime'] == 1, 1.2, 0.8)
    data['trend_scaling'] = np.where(data['trend_regime'] == 1, 1.1, 0.9)
    data['amplitude_scaling'] = np.where(data['amplitude_regime'] == 1, 1.15, 0.85)
    
    # Gap and Breakout Enhancement
    data['gap_enhancement'] = np.where(data['overnight_gap'] * data['momentum_5d'] > 0, 1.1, 1.0)
    data['breakout_enhancement'] = np.where((data['volume_breakout'] == 1) | (data['range_breakout'] == 1), 1.2, 1.0)
    
    # Final Alpha Factor Calculation
    data['alpha_factor'] = (
        data['momentum_efficiency'] * 
        data['volume_weighted_acceleration'] * 
        data['volatility_scaling'] * 
        data['trend_scaling'] * 
        data['amplitude_scaling'] * 
        data['gap_enhancement'] * 
        data['breakout_enhancement'] * 
        (1 + data['divergence_confirmation'] * 0.1) * 
        (1 + data['volume_persistence'] * 0.05)
    )
    
    # Clean up and return the alpha factor series
    alpha_series = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
