import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive Momentum-Volume Volatility Convergence Factor
    
    Economic intuition: Identifies stocks with strong momentum signals that are supported by
    increasing volume participation while exhibiting favorable volatility characteristics.
    The factor combines short-term price acceleration, medium-term volume confirmation,
    and long-term volatility stability to detect sustainable price movements.
    
    Key innovations:
    - Momentum acceleration across multiple asymmetric horizons (2, 5, 13 days)
    - Volume confirmation through dollar-volume weighted efficiency
    - Volatility regime detection using high-low range patterns
    - Adaptive smoothing based on recent volatility conditions
    - Geometric combination for robust multi-timeframe signals
    """
    
    # Asymmetric momentum horizons for capturing different market regimes
    mom_horizons = [2, 5, 13]
    
    # Momentum acceleration components
    mom_components = []
    for horizon in mom_horizons:
        # Price momentum with acceleration term
        returns = df['close'] / df['close'].shift(horizon) - 1
        prev_returns = df['close'].shift(horizon) / df['close'].shift(horizon * 2) - 1
        momentum_accel = returns - prev_returns
        mom_components.append(momentum_accel)
    
    # Volume efficiency with dollar volume weighting
    volume_efficiency = []
    for horizon in [3, 8]:
        # Dollar volume normalized efficiency
        dollar_volume = df['close'] * df['volume']
        avg_dollar_volume = dollar_volume.rolling(window=horizon).mean()
        volume_expansion = dollar_volume / (avg_dollar_volume + 1e-7)
        
        # Price efficiency relative to volume expansion
        price_change = df['close'] / df['close'].shift(horizon) - 1
        vol_efficiency = price_change / (volume_expansion + 1e-7)
        volume_efficiency.append(vol_efficiency)
    
    # Volatility regime detection
    volatility_components = []
    for window in [5, 13]:
        # True range based volatility
        true_range = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        
        avg_true_range = true_range.rolling(window=window).mean()
        vol_regime = true_range / (avg_true_range + 1e-7)
        volatility_components.append(vol_regime)
    
    # Geometric combination of momentum components
    momentum_geo = (mom_components[0].abs() * mom_components[1].abs() * mom_components[2].abs()) ** (1/3)
    momentum_geo = momentum_geo * np.sign(mom_components[1])  # Use medium-term direction
    
    # Geometric combination of volume efficiency
    volume_geo = (volume_efficiency[0] * volume_efficiency[1]) ** 0.5
    
    # Geometric combination of volatility regimes
    volatility_geo = (volatility_components[0] * volatility_components[1]) ** 0.5
    
    # Core factor: Strong momentum with volume confirmation in favorable volatility regime
    raw_factor = momentum_geo * volume_geo / (volatility_geo + 1e-7)
    
    # Adaptive smoothing based on recent volatility
    recent_vol = raw_factor.rolling(window=5).std()
    smooth_window = np.where(recent_vol > recent_vol.rolling(window=21).median(), 3, 8)
    
    # Apply adaptive smoothing
    final_factor = pd.Series(index=raw_factor.index, dtype=float)
    for i in range(len(raw_factor)):
        if i >= max(smooth_window):
            window = int(smooth_window[i])
            final_factor.iloc[i] = raw_factor.iloc[i-window+1:i+1].mean()
        else:
            final_factor.iloc[i] = raw_factor.iloc[i]
    
    return final_factor
