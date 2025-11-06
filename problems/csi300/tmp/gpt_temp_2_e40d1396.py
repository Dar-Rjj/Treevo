import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Fractal Market State Identification
    # Fractal Dimension (5-day price path length / straight-line distance)
    data['path_length'] = (data['high'] - data['low']).rolling(window=5).sum()
    data['straight_distance'] = data['close'] - data['close'].shift(4)
    data['fractal_dim'] = data['path_length'] / (abs(data['straight_distance']) + 1e-8)
    
    # Hurst Exponent (simplified R/S analysis over 20-day window)
    def hurst_exponent(series):
        if len(series) < 20:
            return 0.5
        lags = range(2, 10)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['hurst'] = data['close'].rolling(window=20).apply(hurst_exponent, raw=True)
    
    # Momentum Decay Characterization
    # Compute 10-day returns
    data['ret_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum Half-Life (autocorrelation decay)
    def momentum_half_life(returns):
        if len(returns) < 15:
            return 10
        try:
            autocorr = [returns.autocorr(lag=i) for i in range(1, 6)]
            valid_acf = [acf for acf in autocorr if not np.isnan(acf) and acf > 0]
            if len(valid_acf) == 0:
                return 10
            decay_rate = -np.log(valid_acf[0]) if valid_acf[0] > 0 else 1
            half_life = max(1, min(20, np.log(2) / decay_rate))
            return half_life
        except:
            return 10
    
    data['half_life'] = data['ret_10d'].rolling(window=15).apply(momentum_half_life, raw=False)
    data['decay_adj_momentum'] = data['ret_10d'] * np.exp(-1 / (data['half_life'] + 1e-8))
    
    # Flow Asymmetry Measurement
    data['up_flow'] = ((data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['down_flow'] = ((data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['asymmetry_ratio'] = (data['up_flow'] - data['down_flow']) / (data['up_flow'] + data['down_flow'] + 1e-8)
    
    # Regime-Adaptive Signal Construction
    # Trending Markets (Hurst > 0.6)
    trending_signal = data['decay_adj_momentum'] * data['ret_10d'].rolling(window=2).mean()
    
    # Mean-Reverting Markets (Hurst < 0.4)
    mean_rev_signal = data['asymmetry_ratio'] * (-data['ret_10d'].rolling(window=5).mean())
    
    # Transition States
    transition_signal = (data['decay_adj_momentum'] + data['asymmetry_ratio']) / 2
    transition_signal = transition_signal.rolling(window=3).mean()
    
    # Combine regimes
    data['raw_signal'] = np.where(
        data['hurst'] > 0.6, trending_signal,
        np.where(data['hurst'] < 0.4, mean_rev_signal, transition_signal)
    )
    
    # Volatility-Constrained Enhancement
    # Realized Volatility (10-day average true range)
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['realized_vol'] = data['tr'].rolling(window=10).mean()
    data['volatility_ratio'] = data['realized_vol'] / (data['close'].rolling(window=10).mean() + 1e-8)
    
    # Volatility-Adjusted Signal
    data['final_signal'] = data['raw_signal'] / (1 + data['volatility_ratio'])
    
    return data['final_signal']
