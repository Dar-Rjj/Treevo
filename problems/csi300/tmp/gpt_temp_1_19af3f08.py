import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic components
    df['TrueRange'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Fractal Microstructure Analysis
    tick_scale = 0.01  # assuming minimum price movement
    df['PriceFractal1'] = np.log(df['high'] - df['low']) / np.log(tick_scale)
    df['PriceFractal2'] = np.log(df['TrueRange']) / np.log(df['high'] - df['low'])
    df['PriceFractalDim'] = (df['PriceFractal1'] + df['PriceFractal2']) / 2
    
    # Volume Fractal Patterns
    df['VolumeRatio1'] = df['volume'] / df['volume'].shift(1)
    df['VolumeRatio2'] = df['volume'].shift(1) / df['volume'].shift(2)
    df['VolumeAsymmetry'] = abs(df['VolumeRatio1'] - df['VolumeRatio2']) / (df['VolumeRatio1'] + df['VolumeRatio2'])
    
    # Bid-Ask Proxy
    df['BidAskProxy'] = abs((df['high'] + df['low']) / 2 - df['close']) / (df['high'] - df['low'])
    
    # Nonlinear Momentum Dynamics
    df['MomentumAcceleration'] = (df['close'] - df['close'].shift(3)) - (df['close'].shift(3) - df['close'].shift(6))
    df['MomentumCurvature'] = (df['close'] - 2 * df['close'].shift(3) + df['close'].shift(6)) / df['close'].shift(6)
    
    # Volatility-Adjusted Momentum
    df['TrueRangeStd5'] = df['TrueRange'].rolling(window=5, min_periods=3).std()
    df['VolAdjMomentum'] = (df['close'] - df['close'].shift(5)) / df['TrueRangeStd5']
    
    # Regime-Dependent Microstructure
    df['FractalDim5'] = df['PriceFractalDim'].rolling(window=5, min_periods=3).mean()
    df['FractalDim20'] = df['PriceFractalDim'].rolling(window=20, min_periods=10).mean()
    df['FractalRegime'] = df['FractalDim5'] / df['FractalDim20']
    
    df['MomentumFractalAlignment'] = df['MomentumAcceleration'] * df['PriceFractalDim']
    
    # Microstructure Persistence (3-day autocorrelation of bid-ask proxy)
    df['BidAskProxy_autocorr'] = df['BidAskProxy'].rolling(window=3, min_periods=2).apply(
        lambda x: x.autocorr() if len(x) > 1 and not np.isnan(x.autocorr()) else 0, raw=False
    )
    
    # Volume-Microstructure Interaction
    df['FractalVolume'] = df['volume'] * df['PriceFractalDim']
    df['MicrostructureEfficiency'] = abs(df['close'] - df['open']) / (df['PriceFractalDim'] * df['TrueRange'])
    df['VolumeFractalMomentum'] = df['VolumeAsymmetry'] * df['MomentumCurvature']
    
    # Adaptive Signal Synthesis
    # Fractal Regime Weighting
    high_fractal_mask = df['FractalRegime'] > 1.0
    low_fractal_mask = df['FractalRegime'] <= 1.0
    
    # High fractal dimension regime
    high_regime_signal = (
        0.35 * df['MomentumAcceleration'] +
        0.40 * df['MicrostructureEfficiency'] +
        0.25 * df['VolumeFractalMomentum']
    )
    
    # Low fractal dimension regime
    low_regime_signal = (
        0.45 * df['MomentumCurvature'] +
        0.30 * df['BidAskProxy'] +
        0.25 * df['BidAskProxy_autocorr']
    )
    
    # Final Signal with regime weighting
    final_signal = pd.Series(index=df.index, dtype=float)
    final_signal[high_fractal_mask] = high_regime_signal[high_fractal_mask]
    final_signal[low_fractal_mask] = low_regime_signal[low_fractal_mask]
    
    # Apply fractal-microstructure alignment
    final_signal = final_signal * df['MomentumFractalAlignment']
    
    # Clean up and return
    final_signal = final_signal.replace([np.inf, -np.inf], np.nan)
    return final_signal
