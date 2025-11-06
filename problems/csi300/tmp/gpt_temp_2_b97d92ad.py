import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Hierarchical Price-Volume Entropy Alpha Factor
    Combines entropy-based regime detection with microstructure signals
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price change entropy across multiple timeframes
    for window in [5, 10, 20]:
        # Calculate price changes
        price_changes = data['close'].pct_change().rolling(window=window)
        
        # Discretize price changes into 5 bins for entropy calculation
        def calc_price_entropy(series):
            if len(series.dropna()) < 5:
                return np.nan
            hist, _ = np.histogram(series.dropna(), bins=5, density=True)
            hist = hist[hist > 0]  # Remove zero bins for entropy calculation
            return entropy(hist)
        
        data[f'price_entropy_{window}'] = price_changes.apply(calc_price_entropy, raw=False)
    
    # 2. Volume distribution entropy shifts
    for window in [5, 10]:
        # Calculate volume changes
        volume_changes = data['volume'].pct_change().rolling(window=window)
        
        def calc_volume_entropy(series):
            if len(series.dropna()) < 5:
                return np.nan
            hist, _ = np.histogram(series.dropna(), bins=5, density=True)
            hist = hist[hist > 0]
            return entropy(hist)
        
        data[f'volume_entropy_{window}'] = volume_changes.apply(calc_volume_entropy, raw=False)
    
    # 3. Combined price-volume entropy states
    data['combined_entropy_5'] = (data['price_entropy_5'] + data['volume_entropy_5']) / 2
    data['combined_entropy_10'] = (data['price_entropy_10'] + data['volume_entropy_10']) / 2
    
    # 4. OHLC-based spread and impact estimation
    data['relative_spread'] = (data['high'] - data['low']) / data['close']
    data['price_impact'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # 5. Volume-price relationship for order flow
    data['vwp'] = data['amount'] / data['volume']  # Volume-weighted price approximation
    data['volume_price_corr'] = data['volume'].rolling(10).corr(data['close'])
    
    # 6. Range-based liquidity measures
    data['efficiency_ratio'] = abs(data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(5).max() - data['low'].rolling(5).min()).replace(0, np.nan)
    
    # 7. Price series scaling properties (fractal dimension approximation)
    def hurst_exponent(series, max_lag=20):
        if len(series.dropna()) < max_lag:
            return np.nan
        lags = range(2, min(max_lag, len(series.dropna())))
        tau = [np.std(np.subtract(series.dropna()[lag:], series.dropna()[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['hurst_20'] = data['close'].rolling(50).apply(hurst_exponent, raw=False)
    
    # 8. Multi-scale volatility patterns
    data['volatility_ratio'] = data['close'].pct_change().rolling(5).std() / \
                              data['close'].pct_change().rolling(20).std().replace(0, np.nan)
    
    # 9. Regime-Adaptive Signal Generation
    # High entropy regime (mean-reversion)
    mean_reversion_signal = -data['close'].pct_change(5)
    
    # Low entropy regime (momentum)
    momentum_signal = data['close'].pct_change(5)
    
    # Regime classification based on combined entropy
    high_entropy_threshold = data['combined_entropy_10'].quantile(0.7)
    low_entropy_threshold = data['combined_entropy_10'].quantile(0.3)
    
    # Signal weighting by regime confidence
    regime_weight = np.where(data['combined_entropy_10'] > high_entropy_threshold, 
                           data['combined_entropy_10'], 0)
    regime_weight = np.where(data['combined_entropy_10'] < low_entropy_threshold, 
                           1 - data['combined_entropy_10'], regime_weight)
    
    # Combine signals based on regime
    final_signal = np.where(data['combined_entropy_10'] > high_entropy_threshold,
                           mean_reversion_signal * regime_weight,
                           momentum_signal * regime_weight)
    
    # Add microstructure adjustments
    microstructure_adjustment = (data['efficiency_ratio'] * data['volume_price_corr']).fillna(0)
    final_signal = final_signal * (1 + 0.1 * microstructure_adjustment)
    
    # Normalize final signal
    final_signal_series = pd.Series(final_signal, index=data.index)
    final_signal_series = (final_signal_series - final_signal_series.rolling(20).mean()) / \
                         final_signal_series.rolling(20).std()
    
    return final_signal_series.fillna(0)
