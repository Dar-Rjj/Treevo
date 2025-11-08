import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate adaptive regime momentum factor with volume persistence
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Identify Current Trading Regime
    # Volatility regime: (High[t]-Low[t]) / Close[t]
    data['volatility_regime'] = (data['high'] - data['low']) / data['close']
    
    # Volume regime: Volume[t] / Volume[t-5]
    data['volume_regime'] = data['volume'] / data['volume'].shift(5)
    
    # Regime classification based on volatility and volume percentiles
    data['volatility_percentile'] = data['volatility_regime'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    data['volume_percentile'] = data['volume_regime'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Define regimes
    data['regime'] = 'transition'
    data.loc[(data['volatility_percentile'] > 0.7) & (data['volume_percentile'] > 0.7), 'regime'] = 'high_volatility'
    data.loc[(data['volatility_percentile'] < 0.3) & (data['volume_percentile'] < 0.3), 'regime'] = 'low_volatility'
    
    # Calculate Regime-Specific Momentum Signals
    # High volatility regime: Mean reversion momentum
    high_vol_momentum = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    high_vol_volume = data['volume'] / data['volume'].shift(1)
    high_vol_signal = high_vol_momentum * high_vol_volume
    
    # Low volatility regime: Breakout momentum
    low_vol_momentum = (data['close'] - data['open']) / (data['high'] - data['low'])
    low_vol_volume = data['volume'] / data['volume'].shift(2)
    low_vol_signal = low_vol_momentum * low_vol_volume
    
    # Transition regime: Acceleration momentum
    trans_momentum = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    trans_volume = data['volume'] / data['volume'].shift(3)
    trans_signal = trans_momentum * trans_volume
    
    # Generate Adaptive Composite Factor
    # Apply regime-specific signals
    data['regime_signal'] = 0.0
    data.loc[data['regime'] == 'high_volatility', 'regime_signal'] = high_vol_signal
    data.loc[data['regime'] == 'low_volatility', 'regime_signal'] = low_vol_signal
    data.loc[data['regime'] == 'transition', 'regime_signal'] = trans_signal
    
    # Regime weighting based on persistence
    regime_persistence = data['regime'].ne(data['regime'].shift(1)).rolling(window=5).sum()
    regime_weight = 1 / (1 + regime_persistence)
    
    # Volume-momentum alignment adjustment
    price_momentum = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    volume_momentum = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    alignment = np.sign(price_momentum) * np.sign(volume_momentum)
    
    # Multi-timeframe smoothing
    short_term_signal = data['regime_signal'].rolling(window=3, min_periods=2).mean()
    medium_term_signal = data['regime_signal'].rolling(window=5, min_periods=3).mean()
    
    # Final adaptive momentum factor
    adaptive_factor = (
        regime_weight * 
        data['regime_signal'] * 
        alignment * 
        (0.6 * short_term_signal + 0.4 * medium_term_signal)
    )
    
    # Clean and return
    factor = adaptive_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
