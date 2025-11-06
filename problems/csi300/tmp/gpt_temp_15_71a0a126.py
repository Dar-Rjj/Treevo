import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Residual Momentum with Volatility-Regime Adaptive Volume Acceleration
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Volatility Regime Identification
    # Short-term volatility measures
    data['high_low_5d'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    data['close_vol_5d'] = data['close'].pct_change().rolling(window=5).std()
    
    # Long-term volatility measures  
    data['high_low_20d'] = data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()
    data['close_vol_20d'] = data['close'].pct_change().rolling(window=20).std()
    
    # Volatility ratios (short-term vs long-term)
    data['vol_ratio_hl'] = data['high_low_5d'] / data['high_low_20d']
    data['vol_ratio_close'] = data['close_vol_5d'] / data['close_vol_20d']
    
    # Regime classification
    data['high_vol_regime'] = (data['vol_ratio_hl'] > 1.2) & (data['vol_ratio_close'] > 1.2)
    data['low_vol_regime'] = (data['vol_ratio_hl'] < 0.8) & (data['vol_ratio_close'] < 0.8)
    data['normal_vol_regime'] = ~data['high_vol_regime'] & ~data['low_vol_regime']
    
    # Residual Momentum Calculation (using 20-day period)
    # For market return proxy, use average return across all stocks in the dataset
    # In practice, you would use a proper market index
    market_return = data['close'].pct_change(20)
    stock_return = data['close'].pct_change(20)
    data['residual_momentum'] = stock_return - market_return
    
    # Volume Acceleration Analysis
    # Volume momentum component
    data['volume_change_5d'] = data['volume'].pct_change(5)
    data['volume_momentum_dir'] = np.sign(data['volume_change_5d'])
    
    # Volume slope calculation (10-day linear regression slope)
    def volume_slope(volume_series):
        if len(volume_series) < 10:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    data['volume_slope'] = data['volume'].rolling(window=10).apply(volume_slope, raw=True)
    data['volume_acceleration'] = data['volume_slope'] / data['volume'].rolling(window=10).mean()
    
    # Adaptive Signal Generation
    # Define thresholds
    vol_acc_threshold_high = data['volume_acceleration'].quantile(0.7)
    vol_acc_threshold_low = data['volume_acceleration'].quantile(0.3)
    res_mom_threshold = data['residual_momentum'].quantile(0.5)
    
    # Initialize factor values
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # High Volatility Regime Signals (reversal focus)
    high_vol_mask = data['high_vol_regime']
    # Bullish: Negative residual momentum with strong volume acceleration
    bullish_high_vol = (data['residual_momentum'] < -res_mom_threshold) & (data['volume_acceleration'] > vol_acc_threshold_high)
    # Bearish: Positive residual momentum with strong volume acceleration  
    bearish_high_vol = (data['residual_momentum'] > res_mom_threshold) & (data['volume_acceleration'] > vol_acc_threshold_high)
    
    factor_values[high_vol_mask & bullish_high_vol] = 1.0
    factor_values[high_vol_mask & bearish_high_vol] = -1.0
    
    # Low Volatility Regime Signals (breakout focus)
    low_vol_mask = data['low_vol_regime']
    # Bullish: Positive residual momentum with confirming volume acceleration
    bullish_low_vol = (data['residual_momentum'] > res_mom_threshold) & (data['volume_acceleration'] > vol_acc_threshold_low)
    # Bearish: Negative residual momentum with confirming volume acceleration
    bearish_low_vol = (data['residual_momentum'] < -res_mom_threshold) & (data['volume_acceleration'] > vol_acc_threshold_low)
    
    factor_values[low_vol_mask & bullish_low_vol] = 1.0
    factor_values[low_vol_mask & bearish_low_vol] = -1.0
    
    # Normal Regime Signals (balanced approach)
    normal_vol_mask = data['normal_vol_regime']
    # Weighted combination - stronger signals get higher weights
    normal_signal = (
        data['residual_momentum'] * 0.6 + 
        data['volume_acceleration'] * 0.4
    )
    # Scale to reasonable range
    normal_signal_scaled = normal_signal / normal_signal.abs().rolling(window=50, min_periods=1).mean()
    factor_values[normal_vol_mask] = normal_signal_scaled[normal_vol_mask]
    
    # Fill any remaining NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
