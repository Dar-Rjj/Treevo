import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    
    # Calculate Multi-Period Momentum
    data['mom_3d'] = data['close'].pct_change(3)
    data['mom_5d'] = data['close'].pct_change(5)
    
    # Derive Acceleration Signal
    data['momentum_acceleration'] = data['intraday_momentum'] - data['mom_3d']
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Compute Price Efficiency
    data['price_efficiency'] = abs(data['intraday_momentum']) / data['true_range']
    data['price_efficiency'] = data['price_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Analyze Volatility Regime
    data['daily_return'] = data['close'].pct_change()
    data['vol_5d'] = data['daily_return'].rolling(window=5, min_periods=3).std()
    data['vol_20d'] = data['daily_return'].rolling(window=20, min_periods=10).std()
    data['vol_ratio'] = data['vol_5d'] / data['vol_20d']
    
    # Calculate Volume Acceleration
    data['vol_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_acceleration'] = data['volume'] / data['vol_5d_avg']
    
    # Assess Volume-Price Alignment
    data['volume_momentum_alignment'] = np.sign(data['momentum_acceleration']) * np.sign(data['volume_acceleration'] - 1)
    
    # Identify Momentum Quality
    data['momentum_quality'] = data['price_efficiency'] * data['volume_momentum_alignment']
    
    # Regress Momentum on Volatility Structure
    # Extract volatility-independent momentum acceleration using rolling regression
    def rolling_regression_residuals(y, x, window=10):
        residuals = pd.Series(index=y.index, dtype=float)
        for i in range(window, len(y)):
            if not (pd.isna(y.iloc[i]) or pd.isna(x.iloc[i])):
                y_window = y.iloc[i-window:i]
                x_window = x.iloc[i-window:i]
                valid_mask = ~(y_window.isna() | x_window.isna())
                if valid_mask.sum() >= 5:
                    y_clean = y_window[valid_mask]
                    x_clean = x_window[valid_mask]
                    if len(x_clean) > 0:
                        beta = np.cov(y_clean, x_clean)[0, 1] / np.var(x_clean)
                        alpha = np.mean(y_clean) - beta * np.mean(x_clean)
                        residuals.iloc[i] = y.iloc[i] - (alpha + beta * x.iloc[i])
        return residuals
    
    data['vol_independent_momentum'] = rolling_regression_residuals(
        data['momentum_acceleration'], data['vol_ratio'], window=10
    )
    
    # Apply Volume Confirmation Filter
    data['volume_confirmed_momentum'] = data['vol_independent_momentum'] * data['volume_acceleration']
    
    # Incorporate Efficiency Assessment
    data['efficiency_adjusted_signal'] = data['volume_confirmed_momentum'] * data['price_efficiency']
    
    # Generate Final Predictive Signal with Regime-Adaptive Scaling
    def get_vol_regime_weight(vol_ratio):
        if vol_ratio < 0.8:
            return 1.5  # Volatility compression - stronger weights
        elif vol_ratio > 1.2:
            return 0.7  # High volatility - reduced weights
        else:
            return 1.0  # Normal volatility - moderate weights
    
    data['vol_regime_weight'] = data['vol_ratio'].apply(get_vol_regime_weight)
    data['final_signal'] = data['efficiency_adjusted_signal'] * data['vol_regime_weight']
    
    # Clean up intermediate columns
    result = data['final_signal'].copy()
    
    return result
