import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Price-Volume Divergence factor
    """
    data = df.copy()
    
    # Volatility Regime Classification
    # Short-term volatility (5-day range-based)
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close']
    data['short_term_vol'] = data['daily_range_vol'].rolling(window=5, min_periods=3).mean()
    
    # Medium-term volatility (20-day close-to-close)
    data['close_to_close_ret'] = data['close'].pct_change().abs()
    data['medium_term_vol'] = data['close_to_close_ret'].rolling(window=20, min_periods=10).mean()
    
    # Volatility regime
    data['high_vol_regime'] = (data['short_term_vol'] > data['medium_term_vol']).astype(int)
    data['vol_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Price Momentum Signals
    data['price_3d'] = data['close'].pct_change(3)
    data['price_5d'] = data['close'].pct_change(5)
    data['price_10d'] = data['close'].pct_change(10)
    data['price_20d'] = data['close'].pct_change(20)
    
    # Volume Anomaly Detection
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_20d_std'] = data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_spike'] = data['volume'] / data['volume_20d_avg']
    data['volume_zscore'] = (data['volume'] - data['volume_20d_avg']) / data['volume_20d_std']
    
    # Volume trend slopes
    def calc_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window and not np.isnan(y).any():
                    slope = stats.linregress(x, y).slope
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['volume_slope_5d'] = calc_slope(data['volume'], 5)
    data['volume_slope_10d'] = calc_slope(data['volume'], 10)
    
    # Volume-Price Correlation
    data['abs_return'] = data['close'].pct_change().abs()
    
    def rolling_corr(series1, series2, window):
        corrs = []
        for i in range(len(series1)):
            if i >= window - 1:
                s1 = series1.iloc[i-window+1:i+1]
                s2 = series2.iloc[i-window+1:i+1]
                if len(s1) == window and not (s1.isna().any() or s2.isna().any()):
                    corr = s1.corr(s2)
                    corrs.append(corr if not np.isnan(corr) else 0)
                else:
                    corrs.append(0)
            else:
                corrs.append(0)
        return pd.Series(corrs, index=series1.index)
    
    data['vol_price_corr_5d'] = rolling_corr(data['volume'], data['abs_return'], 5)
    data['vol_price_corr_10d'] = rolling_corr(data['volume'], data['abs_return'], 10)
    
    # High Volatility Regime Patterns
    # Breakout confirmation
    data['breakout_confirmation'] = (
        (data['price_5d'] > 0) & 
        (data['volume_spike'] > 1.5) & 
        (data['vol_price_corr_5d'] > 0.3)
    ).astype(int)
    
    # False breakout warning
    data['false_breakout_warning'] = (
        (data['price_5d'] > 0) & 
        (data['volume_slope_5d'] < 0) & 
        (data['vol_price_corr_5d'] < 0.1)
    ).astype(int)
    
    # Reversal signals
    data['price_exhaustion'] = (
        (data['price_3d'] < 0) & 
        (data['volume_spike'] > 1.8) & 
        (data['price_10d'].abs() > data['medium_term_vol'] * 8)
    ).astype(int)
    
    data['volume_divergence_extremes'] = (
        ((data['high'] == data['high'].rolling(10).max()) | 
         (data['low'] == data['low'].rolling(10).min())) & 
        (data['volume_spike'] < 0.8)
    ).astype(int)
    
    # Low Volatility Regime Patterns
    # Accumulation detection
    data['accumulation_signal'] = (
        (data['price_10d'] > 0) & 
        (data['volume_slope_10d'] > 0) & 
        (data['vol_price_corr_10d'] > 0.2)
    ).astype(int)
    
    # Distribution warning
    data['distribution_warning'] = (
        (data['price_5d'].abs() < data['medium_term_vol'] * 2) & 
        (data['volume_spike'] > 1.2) & 
        (data['vol_price_corr_5d'] < -0.1)
    ).astype(int)
    
    # Breakout anticipation
    data['breakout_anticipation'] = (
        (data['volume_spike'] < 0.7) & 
        (data['volume_slope_5d'] < 0) & 
        (data['short_term_vol'] < data['medium_term_vol'] * 0.8)
    ).astype(int)
    
    # Mean reversion opportunities
    data['mean_reversion_signal'] = (
        (data['close_to_close_ret'] > data['medium_term_vol'] * 3) & 
        (data['volume_spike'] < 1.2) & 
        (data['high_vol_regime'] == 0)
    ).astype(int)
    
    # Volume-supported reversals
    data['volume_reversal'] = (
        (data['price_3d'] * data['price_5d'] < 0) &  # Direction change
        (data['volume_spike'] > 1.5)
    ).astype(int)
    
    # Adaptive Signal Generation
    # Volume Confirmation Score
    data['volume_confirmation'] = (
        data['volume_spike'].clip(0, 3) * 0.4 +
        np.tanh(data['volume_slope_10d'] * 10) * 0.3 +
        data['vol_price_corr_10d'].clip(-1, 1) * 0.3
    )
    
    # Price Momentum Quality
    data['momentum_quality'] = (
        np.sign(data['price_3d']) * np.sign(data['price_5d']) * 0.3 +
        np.sign(data['price_5d']) * np.sign(data['price_10d']) * 0.3 +
        np.sign(data['price_10d']) * np.sign(data['price_20d']) * 0.4
    )
    
    # Composite Signal Construction
    # High volatility regime signal
    high_vol_signal = (
        data['momentum_quality'] * 
        data['volume_confirmation'] * 
        data['vol_ratio']
    )
    
    # Low volatility regime signal  
    low_vol_signal = (
        data['price_20d'] * 
        np.tanh(data['volume_slope_10d'] * 5) / 
        (data['vol_ratio'] + 0.1)
    )
    
    # Regime-specific pattern signals
    high_vol_patterns = (
        data['breakout_confirmation'] * 0.4 -
        data['false_breakout_warning'] * 0.3 -
        data['price_exhaustion'] * 0.2 -
        data['volume_divergence_extremes'] * 0.1
    )
    
    low_vol_patterns = (
        data['accumulation_signal'] * 0.3 -
        data['distribution_warning'] * 0.2 +
        data['breakout_anticipation'] * 0.3 -
        data['mean_reversion_signal'] * 0.1 +
        data['volume_reversal'] * 0.1
    )
    
    # Final composite signal
    final_signal = (
        data['high_vol_regime'] * (high_vol_signal * 0.7 + high_vol_patterns * 0.3) +
        (1 - data['high_vol_regime']) * (low_vol_signal * 0.7 + low_vol_patterns * 0.3)
    )
    
    # Clean up and return
    factor = final_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
