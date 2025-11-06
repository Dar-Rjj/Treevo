import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor using multi-timeframe regime dynamics
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Price-Volume Convergence with Intraday Extremes
    # Short-term Price-Volume Alignment (3-day)
    data['price_change_3d'] = data['close'].pct_change(periods=3)
    data['volume_change_3d'] = data['volume'].pct_change(periods=3)
    data['alignment_score'] = (np.sign(data['price_change_3d']) * 
                              np.sign(data['volume_change_3d']) * 
                              np.abs(data['price_change_3d'] * data['volume_change_3d']))
    
    # Medium-term Price-Volume Divergence (10-day)
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['divergence_score'] = (np.sign(data['price_momentum_10d']) * 
                               np.sign(data['volume_momentum_10d']) * 
                               (np.abs(data['price_momentum_10d']) - np.abs(data['volume_momentum_10d'])))
    
    # Multi-timeframe Interaction
    data['convergence_divergence'] = data['alignment_score'] * data['divergence_score']
    
    # 2. Dynamic Regime Detection
    # Volatility Regime
    data['intraday_range'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    
    # Liquidity Regime
    data['price_per_volume'] = data['amount'] / data['volume']
    
    # Market Microstructure
    data['overnight_return'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['opening_pressure'] = data['overnight_return'] - data['intraday_return']
    
    # 3. Regime-Adaptive Factor Construction
    # Volatility regime scaling (inverse relationship - higher volatility reduces signal strength)
    volatility_scaling = 1 / (1 + data['intraday_range'].rolling(window=20, min_periods=10).std())
    
    # Liquidity enhancement (higher liquidity increases signal strength)
    liquidity_enhancement = data['price_per_volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Regime-weighted convergence-divergence
    data['regime_weighted_conv_div'] = (data['convergence_divergence'] * 
                                       volatility_scaling * 
                                       (1 + 0.1 * liquidity_enhancement))
    
    # 4. Cross-Timeframe Price Pattern Synthesis
    # Gap Reversion with Regime Context
    data['gap_reversion'] = (-data['overnight_return'] * 
                            np.sign(data['intraday_return']) * 
                            data['volume'] * 
                            volatility_scaling)
    
    # Extreme Price Recovery
    high_to_close = (data['high'] - data['close']) / data['high']
    low_to_close = (data['close'] - data['low']) / data['close']
    data['extreme_recovery'] = ((high_to_close + low_to_close) * 
                               data['volume'] * 
                               liquidity_enhancement)
    
    # Multi-timeframe Directional Agreement
    short_term_trend = np.sign(data['close'].pct_change(periods=3))
    medium_term_trend = np.sign(data['close'].pct_change(periods=10))
    directional_agreement = (short_term_trend == medium_term_trend).astype(int)
    
    # 5. Final Factor Integration
    # Combine all components with regime-adaptive weights
    factor = (data['regime_weighted_conv_div'] * 0.4 +
              data['gap_reversion'] * 0.3 +
              data['extreme_recovery'] * 0.3) * directional_agreement
    
    # Normalize the factor using rolling z-score
    factor_normalized = factor.rolling(window=60, min_periods=30).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    return factor_normalized
