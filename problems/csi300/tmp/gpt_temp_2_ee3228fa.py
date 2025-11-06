import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Volatility-scaled momentum acceleration with volume divergence and regime-aware weighting.
    
    Interpretation:
    - Momentum acceleration captures rate of change in price movement across multiple timeframes
    - Volatility scaling adapts signal strength to current market conditions
    - Volume divergence identifies when trading activity confirms or contradicts price momentum
    - Regime-aware weighting adjusts factor emphasis based on volatility environment
    - Multi-timeframe alignment ensures consistent signals across different horizons
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest bearish momentum acceleration with volume divergence
    """
    
    # Momentum acceleration components
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Momentum acceleration (rate of change of momentum)
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    daily_accel = daily_return - daily_return.shift(1)
    
    # Volatility estimation using daily range
    daily_range = (df['high'] - df['low']) / df['open']
    short_term_vol = daily_range.rolling(window=5).std()
    medium_term_vol = daily_range.rolling(window=15).std()
    
    # Volatility regime classification
    vol_ratio = short_term_vol / (medium_term_vol + 1e-7)
    vol_regime = vol_ratio.apply(lambda x: 'high' if x > 1.3 else 'low' if x < 0.7 else 'medium')
    
    # Volume divergence (trading activity relative to price movement)
    volume_ma = df['volume'].rolling(window=5).mean()
    volume_divergence = (df['volume'] - volume_ma) / (volume_ma + 1e-7)
    price_volume_alignment = np.sign(intraday_return) * np.sign(volume_divergence)
    
    # Regime-aware weighting
    regime_weights = {
        'high': {'intraday': 0.4, 'overnight': 0.2, 'daily': 0.3, 'volume': 0.1},
        'medium': {'intraday': 0.3, 'overnight': 0.3, 'daily': 0.3, 'volume': 0.1},
        'low': {'intraday': 0.2, 'overnight': 0.4, 'daily': 0.2, 'volume': 0.2}
    }
    
    # Volatility scaling factor
    vol_scale = vol_ratio.apply(lambda x: 1.5 if x > 1.3 else 0.7 if x < 0.7 else 1.0)
    
    # Multi-timeframe momentum alignment
    momentum_alignment = (
        np.sign(intraday_accel) * np.sign(overnight_accel) * 
        np.sign(daily_accel) * np.sign(price_volume_alignment)
    )
    
    # Combine components with regime-aware weighting and volatility scaling
    alpha_components = []
    
    for idx in df.index:
        regime = vol_regime.loc[idx]
        weights = regime_weights[regime]
        scale = vol_scale.loc[idx]
        align = momentum_alignment.loc[idx]
        
        component = (
            weights['intraday'] * intraday_accel.loc[idx] +
            weights['overnight'] * overnight_accel.loc[idx] +
            weights['daily'] * daily_accel.loc[idx] +
            weights['volume'] * volume_divergence.loc[idx] * price_volume_alignment.loc[idx]
        ) * scale * align
        
        alpha_components.append(component)
    
    alpha_factor = pd.Series(alpha_components, index=df.index)
    
    return alpha_factor
