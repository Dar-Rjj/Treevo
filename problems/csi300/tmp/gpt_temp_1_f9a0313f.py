import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Fractal Market State Identification
    # Compute Fractal Dimension (5-day price path length / straight-line distance)
    data['price_path_length'] = (data['high'] - data['low']).rolling(window=5).sum()
    data['straight_line_distance'] = data['close'] - data['close'].shift(5)
    data['fractal_dimension'] = data['price_path_length'] / (abs(data['straight_line_distance']) + 1e-8)
    
    # Compute Hurst Exponent (R/S analysis over 20-day window)
    def hurst_exponent(series):
        if len(series) < 20:
            return np.nan
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['hurst'] = data['close'].rolling(window=20).apply(hurst_exponent, raw=True)
    
    # Market State Classification
    data['market_state'] = 0  # Transition
    data.loc[data['hurst'] > 0.6, 'market_state'] = 1  # Trending
    data.loc[data['hurst'] < 0.4, 'market_state'] = -1  # Mean-reverting
    
    # Momentum Decay Characterization
    # Compute Momentum Half-Life (autocorrelation decay of 10-day returns)
    data['returns_10d'] = data['close'].pct_change(10)
    
    def momentum_half_life(returns):
        if len(returns) < 10:
            return np.nan
        try:
            lag1_autocorr = returns.autocorr(lag=1)
            if lag1_autocorr <= 0:
                return 10  # Default to window length if no persistence
            return -np.log(2) / np.log(abs(lag1_autocorr))
        except:
            return 10
    
    data['momentum_half_life'] = data['returns_10d'].rolling(window=20).apply(momentum_half_life, raw=True)
    
    # Calculate Decay-Adjusted Momentum
    data['momentum_decay'] = data['returns_10d'] * np.exp(-1 / (data['momentum_half_life'] + 1e-8))
    
    # Volume-Weighted Decay
    data['volume_10d_median'] = data['volume'].rolling(window=10).median()
    data['volume_weighted_decay'] = data['momentum_decay'] * (data['volume'] / (data['volume_10d_median'] + 1e-8))
    
    # Flow Asymmetry Measurement
    # Up-Flow Intensity
    data['up_flow'] = ((data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    
    # Down-Flow Resistance
    data['down_flow'] = ((data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    
    # Asymmetry Ratio
    data['asymmetry_ratio'] = (data['up_flow'] - data['down_flow']) / (data['up_flow'] + data['down_flow'] + 1e-8)
    
    # Regime-Adaptive Signal Construction
    # Trending Markets: Emphasize Momentum Decay with 2-day persistence
    trending_signal = data['volume_weighted_decay'].rolling(window=2).mean()
    
    # Mean-Reverting Markets: Emphasize Flow Asymmetry with 5-day reversal
    mean_reverting_signal = -data['asymmetry_ratio'].rolling(window=5).mean()
    
    # Transition States: Equal weighting with 3-day smoothing
    transition_signal = (data['volume_weighted_decay'] + data['asymmetry_ratio']).rolling(window=3).mean() / 2
    
    # Combine based on market state
    data['regime_signal'] = np.where(
        data['market_state'] == 1, trending_signal,
        np.where(data['market_state'] == -1, mean_reverting_signal, transition_signal)
    )
    
    # Volatility-Constrained Enhancement
    # Compute Realized Volatility (10-day average true range)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['realized_volatility'] = data['true_range'].rolling(window=10).mean()
    
    # Volatility-Adjusted Signal
    volatility_ratio = data['realized_volatility'] / data['close'].rolling(window=10).std()
    data['vol_adjusted_signal'] = data['regime_signal'] / (1 + volatility_ratio + 1e-8)
    
    # Volume Confirmation
    data['volume_20d_q30'] = data['volume'].rolling(window=20).quantile(0.3)
    data['volume_confirmed_signal'] = data['vol_adjusted_signal'] * (data['volume'] / (data['volume_20d_q30'] + 1e-8))
    
    # Multi-Timeframe Integration
    # Short-Term Component (3-day): Flow Asymmetry × Momentum Decay
    short_term = (data['asymmetry_ratio'] * data['momentum_decay']).rolling(window=3).mean()
    
    # Medium-Term Component (10-day): Regime-Adaptive Signal
    medium_term = data['volume_confirmed_signal'].rolling(window=10).mean()
    
    # Long-Term Component (20-day): Fractal Dimension × Hurst Exponent
    long_term = (data['fractal_dimension'] * data['hurst']).rolling(window=20).mean()
    
    # Final Alpha: (Short × 0.4) + (Medium × 0.4) + (Long × 0.2)
    alpha = (short_term * 0.4) + (medium_term * 0.4) + (long_term * 0.2)
    
    return alpha
