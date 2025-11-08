import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Composite Liquidity Momentum Factor combining return persistence, volume stability, 
    and multiple liquidity proxies to generate adaptive trading signals.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Measure Persistence Metrics
    # Return autocorrelation (1-day and 5-day lags over 20 days)
    data['return_autocorr_1d'] = data['returns'].rolling(window=20).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    data['return_autocorr_5d'] = data['returns'].rolling(window=20).apply(
        lambda x: x.autocorr(lag=5), raw=False
    )
    
    # Volume stability metrics
    data['volume_cv_10d'] = data['volume'].rolling(window=10).std() / data['volume'].rolling(window=10).mean()
    data['volume_autocorr_5d'] = data['volume'].rolling(window=20).apply(
        lambda x: x.autocorr(lag=5), raw=False
    )
    
    # 2. Assess Composite Liquidity
    # Calculate typical price
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Effective spread (High-Low)/Typical Price
    data['effective_spread'] = (data['high'] - data['low']) / data['typical_price']
    
    # Volume concentration ratio (current volume vs 5-day average)
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_concentration'] = data['volume'] / data['volume_5d_avg']
    
    # Amount volatility (5-day std)
    data['amount_volatility'] = data['amount'].rolling(window=5).std()
    
    # Normalize liquidity proxies
    liquidity_proxies = ['effective_spread', 'volume_concentration', 'amount_volatility']
    for proxy in liquidity_proxies:
        data[f'{proxy}_norm'] = (data[proxy] - data[proxy].rolling(window=50).mean()) / data[proxy].rolling(window=50).std()
    
    # Calculate composite liquidity score (weighted average)
    weights = [0.4, 0.3, 0.3]  # Weights for spread, concentration, amount volatility
    data['liquidity_score'] = (
        weights[0] * data['effective_spread_norm'] + 
        weights[1] * data['volume_concentration_norm'] + 
        weights[2] * data['amount_volatility_norm']
    )
    
    # Apply exponential smoothing (10-day span)
    alpha = 2 / (10 + 1)
    data['liquidity_score_smooth'] = data['liquidity_score'].ewm(alpha=alpha, adjust=False).mean()
    
    # 3. Generate Adaptive Signal
    # Combine persistence and liquidity
    # Use average of 1-day and 5-day return autocorrelation
    data['return_persistence'] = (data['return_autocorr_1d'] + data['return_autocorr_5d']) / 2
    
    # Multiply return persistence by liquidity score
    data['raw_signal'] = data['return_persistence'] * data['liquidity_score_smooth']
    
    # Adjust for market conditions using volume stability
    volume_stability = (1 - data['volume_cv_10d']) * data['volume_autocorr_5d']
    data['adjusted_signal'] = data['raw_signal'] * volume_stability
    
    # 4. Apply Robust Validation
    # Cross-validate with regime indicators
    # Calculate volatility regime (high volatility when range > median)
    data['daily_range'] = (data['high'] - data['low']) / data['typical_price']
    data['range_median_60d'] = data['daily_range'].rolling(window=60).median()
    data['volatility_regime'] = data['daily_range'] > data['range_median_60d']
    
    # Adjust signal based on volatility regime
    # In high volatility, reduce signal strength
    regime_adjustment = np.where(data['volatility_regime'], 0.7, 1.0)
    data['regime_adjusted_signal'] = data['adjusted_signal'] * regime_adjustment
    
    # Filter for signal consistency using rolling z-score
    signal_mean = data['regime_adjusted_signal'].rolling(window=20).mean()
    signal_std = data['regime_adjusted_signal'].rolling(window=20).std()
    data['final_signal'] = (data['regime_adjusted_signal'] - signal_mean) / signal_std
    
    # Handle NaN values
    data['final_signal'] = data['final_signal'].fillna(0)
    
    return data['final_signal']
