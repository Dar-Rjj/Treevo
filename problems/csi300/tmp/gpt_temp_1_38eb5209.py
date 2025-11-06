import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Efficiency Convergence factor
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum & Efficiency Calculation
    for n in [5, 10, 20]:
        data[f'price_momentum_{n}'] = (data['close'] - data['close'].shift(n)) / data['close'].shift(n)
        data[f'volume_momentum_{n}'] = data['volume'] / data['volume'].shift(n) - 1
    
    # Intraday Efficiency
    data['intraday_efficiency'] = (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 1e-8)
    
    # Price Path Efficiency for different timeframes
    for n in [5, 10, 20]:
        path_efficiency = []
        for i in range(len(data)):
            if i < n:
                path_efficiency.append(np.nan)
                continue
            price_changes = abs(data['close'].iloc[i-n+1:i+1].diff().dropna())
            if price_changes.sum() == 0:
                path_efficiency.append(0)
            else:
                net_move = abs(data['close'].iloc[i] - data['close'].iloc[i-n])
                path_efficiency.append(net_move / price_changes.sum())
        data[f'path_efficiency_{n}'] = path_efficiency
    
    # Volatility Regime Detection
    # True Range Calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Rolling statistics for regime detection
    data['atr_20'] = data['true_range'].rolling(window=20).mean()
    data['atr_60_median'] = data['true_range'].rolling(window=60).median()
    
    # Volatility regime (1 = high, 0 = low)
    data['vol_regime'] = (data['atr_20'] > data['atr_60_median']).astype(int)
    
    # Regime persistence
    data['regime_persistence'] = data['vol_regime'].rolling(window=5).sum()
    
    # Momentum-Efficiency Convergence Analysis
    # Direction alignment assessment
    momentum_signs = []
    for i in range(len(data)):
        if i < 20:
            momentum_signs.append(0)
            continue
        
        # Check momentum sign consistency across timeframes
        mom_5 = data[f'price_momentum_5'].iloc[i]
        mom_10 = data[f'price_momentum_10'].iloc[i]
        mom_20 = data[f'price_momentum_20'].iloc[i]
        
        # Count positive momentum signals
        pos_count = sum([1 for mom in [mom_5, mom_10, mom_20] if mom > 0])
        neg_count = sum([1 for mom in [mom_5, mom_10, mom_20] if mom < 0])
        
        if pos_count > neg_count:
            momentum_signs.append(1)
        elif neg_count > pos_count:
            momentum_signs.append(-1)
        else:
            momentum_signs.append(0)
    
    data['momentum_direction'] = momentum_signs
    
    # Convergence strength scoring
    convergence_scores = []
    for i in range(len(data)):
        if i < 20:
            convergence_scores.append(0)
            continue
        
        score = 0
        
        # Multi-timeframe momentum consistency
        mom_vals = [data[f'price_momentum_{n}'].iloc[i] for n in [5, 10, 20]]
        mom_std = np.std(mom_vals) if len(mom_vals) > 1 else 0
        mom_consistency = 1 / (1 + abs(mom_std)) if mom_std != 0 else 1
        
        # Volume momentum confirmation
        vol_mom_vals = [data[f'volume_momentum_{n}'].iloc[i] for n in [5, 10, 20]]
        vol_mom_aligned = sum([1 for j in range(3) if np.sign(mom_vals[j]) == np.sign(vol_mom_vals[j])]) / 3
        
        # Efficiency-momentum alignment
        eff_alignment = 0
        for n in [5, 10, 20]:
            if data[f'path_efficiency_{n}'].iloc[i] > 0.6 and data['momentum_direction'].iloc[i] != 0:
                eff_alignment += 0.33
        
        score = (mom_consistency * 0.4 + vol_mom_aligned * 0.3 + eff_alignment * 0.3) * data['momentum_direction'].iloc[i]
        convergence_scores.append(score)
    
    data['convergence_strength'] = convergence_scores
    
    # Order Flow Confirmation
    # Buying/Selling Pressure
    data['buying_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['selling_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume concentration
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Breakout confirmation
    data['resistance_20'] = data['high'].rolling(window=20).max()
    data['support_20'] = data['low'].rolling(window=20).min()
    data['near_resistance'] = (data['close'] >= 0.98 * data['resistance_20']).astype(int)
    data['near_support'] = (data['close'] <= 1.02 * data['support_20']).astype(int)
    
    # Regime-Adaptive Signal Generation
    signals = []
    for i in range(len(data)):
        if i < 20:
            signals.append(0)
            continue
        
        base_signal = data['convergence_strength'].iloc[i]
        regime = data['vol_regime'].iloc[i]
        persistence = data['regime_persistence'].iloc[i] / 5  # Normalize to 0-1
        
        # Order flow adjustments
        flow_strength = (data['buying_pressure'].iloc[i] - data['selling_pressure'].iloc[i]) * data['volume_ratio'].iloc[i]
        
        if regime == 1:  # High volatility
            # Mean-reversion enhanced, volatility-normalized
            volatility_adjustment = 1 / (1 + data['atr_20'].iloc[i] / data['close'].iloc[i])
            signal = base_signal * volatility_adjustment * (1 + 0.2 * flow_strength)
            
        else:  # Low volatility
            # Efficiency-confirmed momentum with breakout potential
            breakout_boost = 1 + 0.3 * (data['near_resistance'].iloc[i] + data['near_support'].iloc[i])
            efficiency_boost = 1 + 0.2 * np.mean([data[f'path_efficiency_{n}'].iloc[i] for n in [5, 10, 20]])
            signal = base_signal * breakout_boost * efficiency_boost * (1 + 0.3 * flow_strength)
        
        # Final signal quality weighting
        regime_weight = 0.7 + 0.3 * persistence
        volume_quality = min(2.0, data['volume_ratio'].iloc[i])
        efficiency_score = np.mean([data[f'path_efficiency_{n}'].iloc[i] for n in [5, 10, 20]])
        
        final_signal = signal * regime_weight * volume_quality * (0.8 + 0.2 * efficiency_score)
        signals.append(final_signal)
    
    result = pd.Series(signals, index=data.index, name='regime_adaptive_convergence')
    return result
