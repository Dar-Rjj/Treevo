import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a novel alpha factor combining momentum, volatility-normalized range, 
    and price-volume relationship signals with regime-aware scaling.
    """
    df = data.copy()
    
    # Price Momentum with Volume Weighting
    # Multi-horizon momentum signals
    df['momentum_short'] = df['close'].pct_change(3)  # 3-day momentum
    df['momentum_medium'] = df['close'].pct_change(10)  # 10-day momentum  
    df['momentum_long'] = df['close'].pct_change(40)  # 40-day momentum
    
    # Volume-based weighting components
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_trend'] = df['volume'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    df['volume_volatility'] = df['volume'].pct_change().rolling(20).std()
    
    # Volume-weighted momentum combination
    volume_weight = np.tanh(df['volume_ratio']) * (1 + 0.5 * np.sign(df['volume_trend']))
    df['momentum_weighted'] = (
        0.4 * df['momentum_short'] + 
        0.35 * df['momentum_medium'] + 
        0.25 * df['momentum_long']
    ) * volume_weight / (1 + df['volume_volatility'])
    
    # Volatility-Normalized Range Factors
    # True range components
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Volatility estimation
    df['returns_vol'] = df['close'].pct_change().rolling(20).std()
    df['atr'] = df['true_range'].rolling(20).mean()
    df['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * ((df['high']/df['low']).apply(np.log)**2).rolling(20).mean())
    
    # Range efficiency measures
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volatility_normalized_range'] = (df['high'] - df['low']) / df['atr']
    
    # Price-Volume Relationship Factors
    # Volume-price divergence
    price_up = df['close'] > df['close'].shift(1)
    volume_down = df['volume'] < df['volume'].shift(1)
    df['divergence_bearish'] = (price_up & volume_down).astype(int)
    
    price_down = df['close'] < df['close'].shift(1) 
    volume_up = df['volume'] > df['volume'].shift(1)
    df['divergence_bullish'] = (price_down & volume_up).astype(int)
    
    # VWAP-based signals
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']
    df['vwap_trend'] = df['vwap'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Accumulation/distribution patterns
    df['up_volume_ratio'] = (df['volume'].where(df['close'] > df['close'].shift(1), 0).rolling(10).sum() / 
                            df['volume'].rolling(10).sum())
    
    # Regime-Aware Factor Scaling
    # Market condition detection
    df['volatility_regime'] = df['returns_vol'].rolling(60).rank(pct=True)
    df['trend_strength'] = df['close'].rolling(20).apply(lambda x: abs(np.polyfit(range(len(x)), x, 1)[0]) / np.std(x))
    
    # Dynamic factor weighting based on regimes
    momentum_regime_weight = np.where(df['trend_strength'] > df['trend_strength'].rolling(60).quantile(0.7), 1.2, 0.8)
    volatility_regime_weight = np.where(df['volatility_regime'] > 0.7, 0.7, 1.3)
    
    # Final factor combination with regime-aware scaling
    momentum_component = df['momentum_weighted'] * momentum_regime_weight
    range_component = df['volatility_normalized_range'] * (1 / (1 + df['volatility_regime']))
    volume_component = (df['up_volume_ratio'] - 0.5) * df['vwap_trend'] * volatility_regime_weight
    
    # Combine all components
    alpha_factor = (
        0.45 * momentum_component +
        0.35 * range_component + 
        0.20 * volume_component
    )
    
    # Remove divergence signals (used for interpretation only)
    alpha_factor = alpha_factor - 0.1 * df['divergence_bearish'] + 0.1 * df['divergence_bullish']
    
    return alpha_factor
