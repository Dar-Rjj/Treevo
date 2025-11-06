import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Volume Asymmetry Calculation
    # Calculate 5-day rolling average of volume (current day + previous 4 days)
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Up-Day Volume Intensity
    data['up_day_volume_intensity'] = np.where(
        data['close'] > data['close'].shift(1),
        data['volume'] / data['volume_ma_5'],
        0
    )
    
    # Down-Day Volume Intensity
    data['down_day_volume_intensity'] = np.where(
        data['close'] < data['close'].shift(1),
        data['volume'] / data['volume_ma_5'],
        0
    )
    
    # Volume Asymmetry Ratio
    data['volume_asymmetry_ratio'] = (
        data['up_day_volume_intensity'] - data['down_day_volume_intensity']
    )
    
    # Regime Detection
    # True Range calculation
    data['true_range'] = np.maximum(
        np.maximum(
            data['high'] - data['low'],
            abs(data['high'] - data['close'].shift(1))
        ),
        abs(data['low'] - data['close'].shift(1))
    )
    
    # Volatility Regime (70th percentile of True Range over 20 days)
    def calc_percentile(series):
        if len(series) >= 20:
            return np.percentile(series, 70)
        else:
            return np.nan
    
    data['true_range_percentile_70'] = (
        data['true_range'].rolling(window=20, min_periods=1).apply(calc_percentile, raw=False)
    )
    
    data['volatility_regime'] = np.where(
        data['true_range'] > data['true_range_percentile_70'],
        1,
        -1
    )
    
    # Trend Regime
    data['close_ma_10'] = data['close'].rolling(window=10, min_periods=1).mean()
    data['trend_regime'] = np.where(data['close'] > data['close_ma_10'], 1, -1)
    
    # Regime Convergence
    data['regime_convergence'] = data['volatility_regime'] * data['trend_regime']
    
    # Gap Analysis
    # Overnight Gap
    data['overnight_gap'] = (data['open'] / data['close'].shift(1)) - 1
    
    # Gap Filled
    data['gap_filled'] = np.where(
        (data['open'] > data['close'].shift(1)) & (data['close'] < data['close'].shift(1)),
        -1,
        np.where(
            (data['open'] < data['close'].shift(1)) & (data['close'] > data['close'].shift(1)),
            1,
            0
        )
    )
    
    # Gap Fill Ratio
    data['gap_fill_ratio'] = data['gap_filled'] * abs(data['overnight_gap'])
    
    # Alpha Construction
    # Core Factor
    data['core_factor'] = data['volume_asymmetry_ratio'] * data['regime_convergence']
    
    # Gap Adjusted
    data['gap_adjusted'] = data['core_factor'] * data['gap_fill_ratio']
    
    # Final Alpha
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['final_alpha'] = data['gap_adjusted'] * data['volume'] * data['price_change']
    
    # Return the final alpha factor series
    return data['final_alpha']
