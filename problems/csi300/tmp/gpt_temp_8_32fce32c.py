import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Asymmetric Momentum with Liquidity-Regime Quality Dynamics
    """
    data = df.copy()
    
    # Multi-Timeframe Asymmetric Acceleration Framework
    # Micro-Scale Asymmetry (Intraday)
    data['micro_price_acc'] = ((data['close'] - data['open']) / (data['high'] - data['low'])).fillna(0)
    data['micro_price_acc_change'] = data['micro_price_acc'] - data['micro_price_acc'].shift(1)
    
    vol_ma5 = data['volume'].rolling(window=5, min_periods=1).mean()
    data['micro_vol_acc'] = data['volume'] / vol_ma5
    data['micro_vol_acc_change'] = data['micro_vol_acc'] - data['micro_vol_acc'].shift(1)
    
    # Asymmetry persistence calculation
    def calc_persistence(series):
        signs = np.sign(series)
        persistence = signs * 0
        current_streak = 0
        current_sign = 0
        for i in range(len(series)):
            if i == 0 or signs.iloc[i] == 0:
                persistence.iloc[i] = 0
                current_streak = 0
                current_sign = signs.iloc[i]
            elif signs.iloc[i] == current_sign:
                current_streak += 1
                persistence.iloc[i] = current_streak * current_sign
            else:
                current_streak = 1
                current_sign = signs.iloc[i]
                persistence.iloc[i] = current_streak * current_sign
        return persistence
    
    data['micro_acc_persistence'] = calc_persistence(data['micro_price_acc_change'])
    
    # Meso-Scale Asymmetry (Daily)
    data['meso_price_acc'] = (np.log(data['close'] / data['close'].shift(1)) - 
                             np.log(data['close'].shift(1) / data['close'].shift(2))).fillna(0)
    data['meso_vol_acc'] = (np.log(data['volume'] / data['volume'].shift(1)) - 
                           np.log(data['volume'].shift(1) / data['volume'].shift(2))).fillna(0)
    
    # Macro-Scale Asymmetry (Multi-day)
    data['macro_price_acc'] = data['close'].pct_change(periods=3).fillna(0)
    data['macro_vol_acc'] = data['volume'].pct_change(periods=3).fillna(0)
    
    # Cross-timeframe coherence
    data['acc_coherence'] = (np.sign(data['micro_price_acc_change']) + 
                            np.sign(data['meso_price_acc']) + 
                            np.sign(data['macro_price_acc'])) / 3
    
    # Asymmetry strength composite
    data['asymmetry_strength'] = (0.4 * data['micro_price_acc_change'] + 
                                 0.35 * data['meso_price_acc'] + 
                                 0.25 * data['macro_price_acc'])
    
    # Liquidity-Regime Quality Assessment
    # Price Impact Efficiency
    data['price_impact_eff'] = (abs(data['close'] - data['close'].shift(1)) / data['amount']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume Distribution Quality
    data['vol_dist_quality'] = data['volume'] / vol_ma5
    
    # Volume Efficiency
    data['vol_efficiency'] = (data['amount'] / (data['volume'] * (data['high'] - data['low']))).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Liquidity Regime Classification
    vol_concentration = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['liquidity_regime'] = np.where(vol_concentration > 1.1, 1, 
                                       np.where(vol_concentration < 0.9, -1, 0))
    
    # Liquidity Quality Dynamics
    eff_ma5 = data['vol_efficiency'].rolling(window=5, min_periods=1).mean()
    data['efficiency_momentum'] = data['vol_efficiency'] - eff_ma5
    
    # Quality persistence
    data['quality_persistence'] = data['vol_efficiency'].rolling(window=3, min_periods=1).mean()
    
    # Regime-Adaptive Momentum with Microstructure Switching
    # Multi-Scale Momentum Analysis
    data['ultra_short_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(4) - 1
    data['momentum_divergence'] = data['ultra_short_momentum'] - data['short_term_momentum']
    
    # Microstructure Regime Detection
    data['price_impact_micro'] = (abs(data['open'] - data['close'].shift(1)) / 
                                 (data['high'] - data['low'])).replace([np.inf, -np.inf], 0).fillna(0)
    
    vol_prev_avg = data['volume'].rolling(window=2, min_periods=1).mean()
    data['vol_regime_persistence'] = data['volume'] / vol_prev_avg.shift(1)
    
    # Regime-Switching Momentum Quality
    def calc_regime_momentum(row):
        if row['liquidity_regime'] == 1:  # High liquidity
            return (0.6 * row['ultra_short_momentum'] + 
                   0.4 * row['short_term_momentum']) * row['vol_efficiency']
        elif row['liquidity_regime'] == -1:  # Low liquidity
            return (0.4 * row['ultra_short_momentum'] + 
                   0.6 * row['short_term_momentum']) * (1 + row['quality_persistence'])
        else:  # Neutral
            return 0.5 * row['ultra_short_momentum'] + 0.5 * row['short_term_momentum']
    
    data['regime_momentum'] = data.apply(calc_regime_momentum, axis=1)
    
    # Asymmetric Pattern Integration with Quality Enhancement
    # Cross-Asymmetry Pattern Recognition
    def count_patterns(series, window=5):
        return series.rolling(window=window, min_periods=1).apply(
            lambda x: np.sum((x > 0) & (data.loc[x.index, 'meso_vol_acc'] > 0)), raw=False
        )
    
    data['positive_asymmetry_days'] = count_patterns(data['meso_price_acc'])
    
    # Quality-Enhanced Asymmetry Synthesis
    data['quality_enhanced_asymmetry'] = (
        data['asymmetry_strength'] * data['vol_efficiency'] * 
        (1 + data['acc_coherence'])
    )
    
    # Regime-Adaptive Composite Factor Construction
    def calc_composite_factor(row):
        if row['liquidity_regime'] == 1:  # High Liquidity Regime
            return (0.35 * row['quality_enhanced_asymmetry'] +
                   0.35 * row['regime_momentum'] +
                   0.2 * row['positive_asymmetry_days'] +
                   0.1 * (1 - row['price_impact_micro']))
        elif row['liquidity_regime'] == -1:  # Low Liquidity Regime
            return (0.4 * row['quality_enhanced_asymmetry'] * (1 + row['quality_persistence']) +
                   0.3 * row['regime_momentum'] +
                   0.2 * row['vol_efficiency'] * row['asymmetry_strength'] +
                   0.1 * row['vol_regime_persistence'])
        else:  # Neutral Regime
            return (0.3 * row['quality_enhanced_asymmetry'] +
                   0.3 * row['regime_momentum'] +
                   0.2 * row['positive_asymmetry_days'] +
                   0.2 * row['vol_efficiency'])
    
    data['composite_factor'] = data.apply(calc_composite_factor, axis=1)
    
    # Final output with regime transition smoothing
    regime_changes = data['liquidity_regime'].diff().abs()
    transition_smoothing = 1 / (1 + regime_changes.rolling(window=3, min_periods=1).mean())
    
    final_factor = data['composite_factor'] * transition_smoothing
    
    return final_factor
