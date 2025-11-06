import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Scale Volatility Momentum
    # Micro: (High_t - Low_t)/(High_{t-1} - Low_{t-1}) × (Close_t - Close_{t-1})/(High_t - Low_t)
    high_low_ratio_micro = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    close_change_ratio_micro = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    micro_vol_momentum = high_low_ratio_micro * close_change_ratio_micro
    
    # Meso: (High_t - Low_t)/(High_{t-5} - Low_{t-5}) × (Close_t - Close_{t-5})/(High_t - Low_t)
    high_low_ratio_meso = (df['high'] - df['low']) / (df['high'].shift(5) - df['low'].shift(5))
    close_change_ratio_meso = (df['close'] - df['close'].shift(5)) / (df['high'] - df['low'])
    meso_vol_momentum = high_low_ratio_meso * close_change_ratio_meso
    
    # Macro: (High_t - Low_t)/(High_{t-20} - Low_{t-20}) × (Close_t - Close_{t-20})/(High_t - Low_t)
    high_low_ratio_macro = (df['high'] - df['low']) / (df['high'].shift(20) - df['low'].shift(20))
    close_change_ratio_macro = (df['close'] - df['close'].shift(20)) / (df['high'] - df['low'])
    macro_vol_momentum = high_low_ratio_macro * close_change_ratio_macro
    
    # Multi-Scale Momentum Patterns
    # Micro: (Close_t - Close_{t-1}) / (Close_{t-1} - Close_{t-2})
    micro_momentum = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) - df['close'].shift(2))
    
    # Meso: STD(Close_{t-4:t} - Close_{t-5:t-1}) / STD(Close_{t-8:t-4} - Close_{t-9:t-5})
    close_diff_meso_recent = df['close'].rolling(window=5).apply(lambda x: np.std(x.diff().dropna()), raw=False)
    close_diff_meso_prior = df['close'].shift(4).rolling(window=5).apply(lambda x: np.std(x.diff().dropna()), raw=False)
    meso_momentum = close_diff_meso_recent / close_diff_meso_prior
    
    # Macro: STD(Close_{t-10:t} - Close_{t-11:t-1}) / STD(Close_{t-20:t-10} - Close_{t-21:t-11})
    close_diff_macro_recent = df['close'].rolling(window=11).apply(lambda x: np.std(x.diff().dropna()), raw=False)
    close_diff_macro_prior = df['close'].shift(10).rolling(window=11).apply(lambda x: np.std(x.diff().dropna()), raw=False)
    macro_momentum = close_diff_macro_recent / close_diff_macro_prior
    
    # Fractal Divergence Identification
    # Volatility Divergence: ABS(Micro Volatility Momentum - Meso Volatility Momentum) × ABS(Meso Volatility Momentum - Macro Volatility Momentum)
    vol_divergence = np.abs(micro_vol_momentum - meso_vol_momentum) * np.abs(meso_vol_momentum - macro_vol_momentum)
    
    # Momentum Divergence: ABS(Micro Momentum Ratio - Meso Momentum Ratio) × ABS(Meso Momentum Ratio - Macro Momentum Ratio)
    mom_divergence = np.abs(micro_momentum - meso_momentum) * np.abs(meso_momentum - macro_momentum)
    
    # Cross-Fractal Divergence: Volatility Divergence × Momentum Divergence
    cross_fractal_divergence = vol_divergence * mom_divergence
    
    # Volume-Volatility Coupling
    # Volume Momentum: (Close_t - Close_{t-1}) × Volume_t/Amount_t × (High_t - Low_t)/Close_{t-1}
    volume_momentum = (df['close'] - df['close'].shift(1)) * (df['volume'] / df['amount']) * ((df['high'] - df['low']) / df['close'].shift(1))
    
    # Volume Alignment: Volume Momentum × Micro Volatility Momentum × sign(Close_t - Open_t) × -1
    volume_alignment = volume_momentum * micro_vol_momentum * np.sign(df['close'] - df['open']) * -1
    
    # Alpha Synthesis
    # Core Divergence: Cross-Fractal Divergence × Volume Alignment
    core_divergence = cross_fractal_divergence * volume_alignment
    
    # Final Alpha: Core Divergence × (1 + Volume Momentum)
    alpha = core_divergence * (1 + volume_momentum)
    
    return alpha
