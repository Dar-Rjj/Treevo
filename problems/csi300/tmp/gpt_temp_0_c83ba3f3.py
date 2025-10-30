import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor with dynamic signal effectiveness and regime adaptation
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols_required = ['open', 'high', 'low', 'close', 'amount', 'volume']
    if not all(col in df.columns for col in cols_required):
        return result
    
    # Copy data to avoid modifying original
    data = df[cols_required].copy()
    
    # Calculate returns for effectiveness measurement
    data['ret_1'] = data['close'].pct_change()
    data['ret_5'] = data['close'].pct_change(5)
    data['ret_20'] = data['close'].pct_change(20)
    
    # Dynamic Effectiveness-Weighted Momentum
    # Multi-timeframe raw momentum
    data['mom_5'] = data['close'] / data['close'].shift(5) - 1
    data['mom_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum direction alignment
    data['mom_alignment'] = np.sign(data['mom_5']) * np.sign(data['mom_20'])
    
    # Recent signal effectiveness measurement
    effectiveness_5 = []
    effectiveness_20 = []
    
    for i in range(len(data)):
        if i >= 40:  # Need enough data for calculations
            # 5-day momentum effectiveness (20-day rolling correlation)
            start_idx = max(0, i-19)
            mom_5_window = data['mom_5'].iloc[start_idx:i+1]
            ret_5_window = data['ret_5'].shift(-5).iloc[start_idx:i+1]
            corr_5 = mom_5_window.corr(ret_5_window) if len(mom_5_window) > 5 else 0
            
            # 20-day momentum effectiveness (40-day rolling correlation)
            start_idx_20 = max(0, i-39)
            mom_20_window = data['mom_20'].iloc[start_idx_20:i+1]
            ret_20_window = data['ret_20'].shift(-20).iloc[start_idx_20:i+1]
            corr_20 = mom_20_window.corr(ret_20_window) if len(mom_20_window) > 10 else 0
            
            effectiveness_5.append(corr_5 if not np.isnan(corr_5) else 0)
            effectiveness_20.append(corr_20 if not np.isnan(corr_20) else 0)
        else:
            effectiveness_5.append(0)
            effectiveness_20.append(0)
    
    data['eff_5'] = effectiveness_5
    data['eff_20'] = effectiveness_20
    
    # Effectiveness-weighted combination
    data['eff_weight_5'] = np.abs(data['eff_5']) / (np.abs(data['eff_5']) + np.abs(data['eff_20']) + 1e-8)
    data['eff_weight_20'] = np.abs(data['eff_20']) / (np.abs(data['eff_5']) + np.abs(data['eff_20']) + 1e-8)
    
    data['weighted_momentum'] = (
        data['eff_weight_5'] * data['mom_5'] + 
        data['eff_weight_20'] * data['mom_20']
    )
    
    # Effectiveness confidence
    data['eff_confidence'] = (
        np.sign(data['eff_5']) * np.sign(data['eff_20']) * 
        (np.abs(data['eff_5']) + np.abs(data['eff_20'])) / 2
    )
    
    # Regime-Specific Price-Volume Efficiency
    # Core efficiency metrics
    data['price_range_eff'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['volume_concentration'] = data['amount'] / (data['high'] - data['low'] + 1e-8)
    
    # Efficiency momentum
    data['eff_mom_5'] = data['price_range_eff'] / data['price_range_eff'].shift(5) - 1
    data['eff_mom_20'] = data['price_range_eff'] / data['price_range_eff'].shift(20) - 1
    
    # Regime detection
    # Volatility regime
    vol_20 = data['close'].pct_change().rolling(20).std()
    vol_60 = data['close'].pct_change().rolling(60).std()
    data['vol_regime'] = np.where(
        vol_20 > 1.5 * vol_60, 'high',
        np.where(vol_20 < 0.67 * vol_60, 'low', 'normal')
    )
    
    # Volume regime
    vol_20_amt = data['volume'].rolling(20).mean()
    vol_60_amt = data['volume'].rolling(60).mean()
    data['volume_regime'] = np.where(
        vol_20_amt > 1.5 * vol_60_amt, 'high',
        np.where(vol_20_amt < 0.67 * vol_60_amt, 'low', 'normal')
    )
    
    # Trend regime
    ret_20 = data['close'].pct_change(20)
    data['trend_regime'] = np.where(
        ret_20 > 0.05, 'uptrend',
        np.where(ret_20 < -0.05, 'downtrend', 'sideways')
    )
    
    # Regime-optimized efficiency signals
    efficiency_signals = []
    
    for i in range(len(data)):
        if i < 20:
            efficiency_signals.append(0)
            continue
            
        vol_regime = data['vol_regime'].iloc[i]
        volume_regime = data['volume_regime'].iloc[i]
        trend_regime = data['trend_regime'].iloc[i]
        
        base_signal = (
            data['price_range_eff'].iloc[i] + 
            data['eff_mom_5'].iloc[i] * 0.3 +
            data['eff_mom_20'].iloc[i] * 0.7
        )
        
        # Regime adjustments
        if vol_regime == 'high':
            signal = base_signal * 1.2 + data['eff_mom_5'].iloc[i] * 0.5
        elif vol_regime == 'low':
            signal = base_signal * 0.8 + data['volume_concentration'].iloc[i] * 0.4
        else:
            signal = base_signal
        
        if volume_regime == 'high':
            signal = signal * 1.1 + data['volume_concentration'].iloc[i] * 0.3
        elif volume_regime == 'low':
            signal = signal * 0.9
        
        if trend_regime == 'uptrend':
            signal = signal * 1.15
        elif trend_regime == 'downtrend':
            signal = signal * 0.85
        
        efficiency_signals.append(signal)
    
    data['efficiency_signal'] = efficiency_signals
    
    # Multi-Timeframe Signal Integration
    # Calculate effectiveness for efficiency signals
    eff_efficiency = []
    for i in range(len(data)):
        if i >= 40:
            start_idx = max(0, i-39)
            eff_window = data['efficiency_signal'].iloc[start_idx:i+1]
            ret_window = data['ret_5'].shift(-5).iloc[start_idx:i+1]
            corr_eff = eff_window.corr(ret_window) if len(eff_window) > 10 else 0
            eff_efficiency.append(corr_eff if not np.isnan(corr_eff) else 0)
        else:
            eff_efficiency.append(0)
    
    data['eff_efficiency'] = eff_efficiency
    
    # Dynamic weight rebalancing
    data['weight_momentum'] = np.abs(data['eff_confidence']) / (np.abs(data['eff_confidence']) + np.abs(data['eff_efficiency']) + 1e-8)
    data['weight_efficiency'] = np.abs(data['eff_efficiency']) / (np.abs(data['eff_confidence']) + np.abs(data['eff_efficiency']) + 1e-8)
    
    # Timeframe consistency scoring
    data['timeframe_consistency'] = (
        np.sign(data['mom_5']) * np.sign(data['mom_20']) * 0.5 +
        np.sign(data['mom_5']) * np.sign(data['efficiency_signal']) * 0.3 +
        np.sign(data['mom_20']) * np.sign(data['efficiency_signal']) * 0.2
    )
    
    # Final composite alpha
    for i in range(len(data)):
        if i < 40:
            result.iloc[i] = 0
        else:
            composite = (
                data['weight_momentum'].iloc[i] * data['weighted_momentum'].iloc[i] +
                data['weight_efficiency'].iloc[i] * data['efficiency_signal'].iloc[i]
            )
            
            # Apply timeframe consistency and regime adjustments
            final_signal = composite * (1 + 0.2 * data['timeframe_consistency'].iloc[i])
            
            # Volatility regime final adjustment
            if data['vol_regime'].iloc[i] == 'high':
                final_signal = final_signal * 0.8  # Reduce magnitude in high vol
            elif data['vol_regime'].iloc[i] == 'low':
                final_signal = final_signal * 1.1  # Amplify in low vol
            
            result.iloc[i] = final_signal
    
    # Clean infinite values and handle NaNs
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.fillna(0)
    
    return result
