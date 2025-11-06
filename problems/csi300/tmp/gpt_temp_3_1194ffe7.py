import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Price-Volume Entropy factor
    """
    data = df.copy()
    
    # Calculate daily true range
    prev_close = data['close'].shift(1)
    high_low_range = data['high'] - data['low']
    high_prev_close = abs(data['high'] - prev_close)
    low_prev_close = abs(data['low'] - prev_close)
    data['true_range'] = np.maximum(high_low_range, np.maximum(high_prev_close, low_prev_close))
    
    # Calculate 10-day volatility (standard deviation of daily returns)
    daily_returns = data['close'].pct_change()
    data['volatility_10d'] = daily_returns.rolling(window=10, min_periods=5).std()
    
    # Calculate volatility ratio
    data['volatility_ratio'] = data['true_range'] / data['volatility_10d']
    
    # Classify volatility regime
    data['volatility_regime'] = 1  # Normal regime
    data.loc[data['volatility_ratio'] > 1.2, 'volatility_regime'] = 2  # High volatility
    data.loc[data['volatility_ratio'] < 0.8, 'volatility_regime'] = 0  # Low volatility
    
    # Calculate price direction changes
    price_changes = np.sign(data['close'].diff())
    price_changes_binary = (price_changes > 0).astype(int)
    
    # Calculate 5-day price entropy (Shannon entropy of price direction changes)
    def calculate_entropy(series, window=5):
        entropy_values = []
        for i in range(len(series)):
            if i < window:
                entropy_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window:i]
            value_counts = window_data.value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
            entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=series.index)
    
    data['price_entropy_5d'] = calculate_entropy(price_changes_binary, window=5)
    
    # Calculate volume direction changes
    volume_changes = np.sign(data['volume'].diff())
    volume_changes_binary = (volume_changes > 0).astype(int)
    
    # Calculate 5-day volume entropy
    data['volume_entropy_5d'] = calculate_entropy(volume_changes_binary, window=5)
    
    # Calculate price-volume divergence
    data['price_volume_divergence'] = abs(data['price_entropy_5d'] - data['volume_entropy_5d'])
    
    # Calculate entropy persistence (autocorrelation of price-volume divergence at lag 3)
    data['divergence_lag1'] = data['price_volume_divergence'].shift(1)
    data['divergence_lag2'] = data['price_volume_divergence'].shift(2)
    data['divergence_lag3'] = data['price_volume_divergence'].shift(3)
    
    def rolling_autocorr(series, window=10, lag=3):
        autocorr_values = []
        for i in range(len(series)):
            if i < window + lag:
                autocorr_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window:i]
            if len(window_data) < window:
                autocorr_values.append(np.nan)
                continue
                
            autocorr = window_data.autocorr(lag=lag)
            autocorr_values.append(autocorr if not pd.isna(autocorr) else 0)
        
        return pd.Series(autocorr_values, index=series.index)
    
    data['entropy_persistence'] = rolling_autocorr(data['price_volume_divergence'], window=10, lag=3)
    
    # Calculate entropy trend (slope of 5-day price entropy)
    def rolling_slope(series, window=5):
        slope_values = []
        x = np.arange(window)
        
        for i in range(len(series)):
            if i < window:
                slope_values.append(np.nan)
                continue
                
            y = series.iloc[i-window:i].values
            if len(y) < window or np.isnan(y).any():
                slope_values.append(np.nan)
                continue
                
            slope = np.polyfit(x, y, 1)[0]
            slope_values.append(slope)
        
        return pd.Series(slope_values, index=series.index)
    
    data['entropy_trend'] = rolling_slope(data['price_entropy_5d'], window=5)
    
    # Generate base signal
    data['base_signal'] = data['price_volume_divergence'] * data['entropy_persistence']
    
    # Apply regime-specific scaling
    data['regime_signal'] = data['base_signal'].copy()
    
    # High volatility regime: multiply by volatility ratio
    high_vol_mask = data['volatility_regime'] == 2
    data.loc[high_vol_mask, 'regime_signal'] = data.loc[high_vol_mask, 'base_signal'] * data.loc[high_vol_mask, 'volatility_ratio']
    
    # Low volatility regime: divide by volatility ratio
    low_vol_mask = data['volatility_regime'] == 0
    data.loc[low_vol_mask, 'regime_signal'] = data.loc[low_vol_mask, 'base_signal'] / data.loc[low_vol_mask, 'volatility_ratio']
    
    # Incorporate entropy trend
    data['final_signal'] = data['regime_signal'] * (1 + data['entropy_trend'])
    
    # Return the final factor values
    return data['final_signal']
