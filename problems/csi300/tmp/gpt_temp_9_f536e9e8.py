import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Momentum with Structural Break Detection
    """
    data = df.copy()
    
    # Micro-regime detection (1-3 days)
    # Intraday volatility clustering
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['prev_daily_range'] = data['daily_range'].shift(1)
    data['range_expansion'] = data['daily_range'] / data['prev_daily_range'].replace(0, np.nan)
    data['gap_volatility'] = abs(data['open'] - data['close'].shift(1)) / data['prev_daily_range'].replace(0, np.nan)
    data['micro_regime_score'] = data['range_expansion'].fillna(0) + data['gap_volatility'].fillna(0)
    
    # Volume-volatility relationship
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1).replace(0, np.nan)
    data['volume_spike_volatility'] = data['volume_ratio'] * data['daily_range']
    data['low_volume_consolidation'] = data['volume_ratio'] / data['daily_range'].replace(0, np.nan)
    data['volume_volatility_regime'] = np.where(
        data['volume_spike_volatility'] > data['volume_spike_volatility'].rolling(5).mean(),
        data['volume_spike_volatility'],
        -data['low_volume_consolidation']
    )
    
    # Price momentum under micro-regimes
    data['price_change'] = data['close'].pct_change()
    data['micro_momentum'] = data['price_change'].rolling(3).mean()
    data['micro_momentum_regime'] = np.where(
        data['micro_regime_score'] > data['micro_regime_score'].rolling(5).mean(),
        data['micro_momentum'],
        -abs(data['micro_momentum'])
    )
    
    # Meso-regime detection (5-10 days)
    # Volatility trend analysis
    data['volatility_5d'] = (data['high'].rolling(5).max() - data['low'].rolling(5).min()) / data['close']
    data['volatility_momentum'] = data['daily_range'] - data['volatility_5d'].shift(5)
    data['volatility_acceleration'] = data['daily_range'] / data['daily_range'].shift(2).replace(0, np.nan)
    data['meso_volatility_regime'] = data['volatility_momentum'] + data['volatility_acceleration'].fillna(0)
    
    # Volume regime transitions
    data['volume_trend'] = data['volume'].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data['volume_trend_break'] = abs(data['volume_trend'] - data['volume_trend'].shift(5))
    data['meso_volume_regime'] = data['volume_trend'] * data['volume_trend_break']
    
    # Price behavior in meso-regimes
    data['meso_momentum'] = data['close'].pct_change(10)
    data['meso_price_regime'] = data['meso_momentum'] * data['meso_volatility_regime']
    
    # Macro-regime detection (15-30 days)
    # Structural volatility shifts
    data['volatility_20d'] = (data['high'].rolling(20).max() - data['low'].rolling(20).min()) / data['close']
    data['volatility_regime_change'] = data['volatility_20d'] - data['volatility_20d'].shift(20)
    data['macro_volatility_regime'] = data['volatility_regime_change'].rolling(5).mean()
    
    # Volume cycle analysis
    data['volume_cycle'] = data['volume'].rolling(30).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    data['macro_volume_regime'] = data['volume_cycle'] * data['volume_cycle'].shift(5)
    
    # Long-term price regime characteristics
    data['macro_momentum'] = data['close'].pct_change(20)
    data['macro_price_regime'] = data['macro_momentum'] * data['macro_volatility_regime']
    
    # Structural Break Detection System
    # Price level structural breaks
    data['resistance_break'] = (data['close'] > data['high'].rolling(20).max().shift(1)).astype(int)
    data['support_break'] = (data['close'] < data['low'].rolling(20).min().shift(1)).astype(int)
    data['price_break_score'] = data['resistance_break'] - data['support_break']
    
    # Volume structural breaks
    data['volume_spike'] = (data['volume'] > data['volume'].rolling(20).mean() * 1.5).astype(int)
    data['volume_trend_change'] = (data['volume_trend'] * data['volume_trend'].shift(5) < 0).astype(int)
    data['volume_break_score'] = data['volume_spike'] + data['volume_trend_change']
    
    # Volatility structural breaks
    data['volatility_jump'] = (data['daily_range'] > data['daily_range'].rolling(20).mean() * 1.5).astype(int)
    data['volatility_break_score'] = data['volatility_jump']
    
    # Composite break score
    data['composite_break_score'] = (
        data['price_break_score'] + 
        data['volume_break_score'] + 
        data['volatility_break_score']
    )
    
    # Regime-Adaptive Momentum Construction
    # Micro-regime momentum factors
    data['volatile_regime_momentum'] = np.where(
        data['micro_regime_score'] > data['micro_regime_score'].rolling(10).mean(),
        data['micro_momentum'],
        0
    )
    
    data['consolidation_momentum'] = np.where(
        data['micro_regime_score'] < data['micro_regime_score'].rolling(10).mean(),
        -data['micro_momentum'],
        0
    )
    
    # Break-enhanced momentum signals
    data['break_momentum'] = data['composite_break_score'] * data['price_change']
    data['volume_break_momentum'] = data['volume_break_score'] * data['price_change']
    
    # Adaptive momentum weighting
    # Regime-specific weights
    volatile_weight = data['micro_regime_score'].rolling(10).rank(pct=True)
    consolidation_weight = 1 - volatile_weight
    
    # Break confirmation weights
    break_weight = data['composite_break_score'].rolling(5).mean()
    
    # Timeframe integration
    micro_weight = 0.4
    meso_weight = 0.35
    macro_weight = 0.25
    
    # Final momentum construction
    micro_component = (
        volatile_weight * data['volatile_regime_momentum'] +
        consolidation_weight * data['consolidation_momentum']
    )
    
    meso_component = data['meso_price_regime']
    macro_component = data['macro_price_regime']
    
    # Break-enhanced components
    break_component = break_weight * (data['break_momentum'] + data['volume_break_momentum'])
    
    # Final alpha factor
    volatility_regime_adaptive_momentum_break = (
        micro_weight * micro_component +
        meso_weight * meso_component +
        macro_weight * macro_component +
        0.1 * break_component
    )
    
    # Signal validation and filtering
    # Regime consistency
    regime_alignment = (
        np.sign(micro_component.fillna(0)) * np.sign(meso_component.fillna(0)) * 
        np.sign(macro_component.fillna(0))
    )
    
    # Apply consistency filter
    final_factor = volatility_regime_adaptive_momentum_break * (regime_alignment >= 0)
    
    return final_factor
