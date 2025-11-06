import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Regime-Adaptive Price-Volume Divergence with Liquidity Adjustment
    """
    data = df.copy()
    
    # Calculate Volatility Regime
    # Compute ATR (Average True Range)
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Short-term volatility (5-day ATR)
    data['short_vol'] = data['tr'].rolling(window=5, min_periods=3).mean()
    
    # Long-term volatility (20-day ATR)
    data['long_vol'] = data['tr'].rolling(window=20, min_periods=10).mean()
    
    # Volatility regime (Short/Long Volatility Ratio)
    data['vol_regime'] = data['short_vol'] / data['long_vol']
    
    # Calculate Price-Volume Trend Components
    def linear_regression_slope(series, window):
        slopes = np.full(len(series), np.nan)
        for i in range(window-1, len(series)):
            if not np.any(np.isnan(series[i-window+1:i+1])):
                y = series[i-window+1:i+1].values
                x = np.arange(window)
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes[i] = slope
        return slopes
    
    # Price trend: 10-day linear regression slope of Close
    data['price_trend'] = linear_regression_slope(data['close'], 10)
    
    # Volume trend: 10-day linear regression slope of Volume
    data['volume_trend'] = linear_regression_slope(data['volume'], 10)
    
    # Detect Divergence (Compare trend directions)
    data['trend_divergence'] = np.sign(data['price_trend']) * np.sign(data['volume_trend'])
    
    # Assess Market Liquidity Conditions
    # Compute Amihud Illiquidity Ratio (10-day average)
    data['daily_return'] = data['close'].pct_change()
    data['amihud_ratio'] = abs(data['daily_return']) / data['amount']
    data['avg_amihud'] = data['amihud_ratio'].rolling(window=10, min_periods=5).mean()
    
    # Classify as Liquid/Illiquid based on ratio threshold (median as threshold)
    liquidity_threshold = data['avg_amihud'].expanding().median()
    data['liquidity_regime'] = (data['avg_amihud'] <= liquidity_threshold).astype(int)
    
    # Generate Regime-Adaptive Signals
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_10d'] = data['close'].pct_change(10)
    
    # High Volatility Regime (vol_regime > 1)
    high_vol_mask = data['vol_regime'] > 1
    data['high_vol_signal'] = np.nan
    data.loc[high_vol_mask, 'high_vol_signal'] = (
        data.loc[high_vol_mask, 'momentum_3d'] * 
        data.loc[high_vol_mask, 'trend_divergence'] * 
        (1 + data.loc[high_vol_mask, 'vol_regime'])
    )
    
    # Low Volatility Regime (vol_regime <= 1)
    low_vol_mask = data['vol_regime'] <= 1
    data['low_vol_signal'] = np.nan
    data.loc[low_vol_mask, 'low_vol_signal'] = (
        data.loc[low_vol_mask, 'momentum_10d'] * 
        data.loc[low_vol_mask, 'trend_divergence'] * 
        (0.5 + 0.5 * data.loc[low_vol_mask, 'vol_regime'])
    )
    
    # Combine volatility regime signals
    data['vol_regime_signal'] = data['high_vol_signal'].fillna(0) + data['low_vol_signal'].fillna(0)
    
    # Apply Liquidity-Based Adjustment
    # Liquid Conditions (liquidity_regime == 1)
    liquid_mask = data['liquidity_regime'] == 1
    data['liquid_adjusted'] = np.nan
    data.loc[liquid_mask, 'liquid_adjusted'] = (
        data.loc[liquid_mask, 'vol_regime_signal'] * 
        (1 + abs(data.loc[liquid_mask, 'volume_trend'])) *
        (1 + data.loc[liquid_mask, 'trend_divergence'])
    )
    
    # Illiquid Conditions (liquidity_regime == 0)
    illiquid_mask = data['liquidity_regime'] == 0
    data['illiquid_adjusted'] = np.nan
    data.loc[illiquid_mask, 'illiquid_adjusted'] = (
        data.loc[illiquid_mask, 'vol_regime_signal'] * 
        (0.5 + 0.5 * abs(data.loc[illiquid_mask, 'volume_trend'])) *
        (0.7 + 0.3 * data.loc[illiquid_mask, 'trend_divergence'])
    )
    
    # Final factor combining all components
    data['factor'] = data['liquid_adjusted'].fillna(0) + data['illiquid_adjusted'].fillna(0)
    
    # Normalize the final factor
    data['factor_normalized'] = (data['factor'] - data['factor'].rolling(window=20, min_periods=10).mean()) / data['factor'].rolling(window=20, min_periods=10).std()
    
    return data['factor_normalized']
