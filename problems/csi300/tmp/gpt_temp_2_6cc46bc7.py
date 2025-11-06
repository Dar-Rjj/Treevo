import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility Asymmetry Analysis
    # Directional Volatility Components
    data['up_volatility'] = np.where(data['close'] > data['open'], data['high'] - data['open'], 0)
    data['down_volatility'] = np.where(data['close'] < data['open'], data['open'] - data['low'], 0)
    data['neutral_volatility'] = np.where(data['close'] == data['open'], np.abs(data['close'] - data['open']), 0)
    
    # Volume-Volatility Coupling
    data['up_volume_efficiency'] = np.where(data['close'] > data['open'], data['up_volatility'] / data['volume'], 0)
    data['down_volume_efficiency'] = np.where(data['close'] < data['open'], data['down_volatility'] / data['volume'], 0)
    data['neutral_efficiency'] = np.where(data['close'] == data['open'], data['neutral_volatility'] / data['volume'], 0)
    
    # Asymmetry Metrics
    volatility_sum = data['up_volatility'] + data['down_volatility']
    data['volatility_skew'] = np.where(volatility_sum > 0, 
                                     (data['up_volatility'] - data['down_volatility']) / volatility_sum, 0)
    
    data['volume_efficiency_ratio'] = np.where(data['down_volume_efficiency'] > 0, 
                                             data['up_volume_efficiency'] / data['down_volume_efficiency'], 1)
    
    # Directional persistence (5-day window)
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            up_count = (window['close'] > window['open']).sum()
            down_count = (window['close'] < window['open']).sum()
            data.loc[data.index[i], 'directional_persistence'] = up_count - down_count
        else:
            data.loc[data.index[i], 'directional_persistence'] = 0
    
    # 2. Regime Detection via Price Efficiency
    # Efficiency Spectrum Analysis
    data['opening_efficiency'] = np.abs(data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['intraday_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['closing_efficiency'] = np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'])
    
    # Replace infinite values
    data['opening_efficiency'] = data['opening_efficiency'].replace([np.inf, -np.inf], 0)
    data['intraday_efficiency'] = data['intraday_efficiency'].replace([np.inf, -np.inf], 0)
    data['closing_efficiency'] = data['closing_efficiency'].replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    
    # Efficiency Regime Classification
    data['efficiency_regime'] = 'mixed'
    data.loc[(data['intraday_efficiency'] > 0.7) & (data['opening_efficiency'] > 0.5), 'efficiency_regime'] = 'high'
    data.loc[(data['intraday_efficiency'] < 0.3) | (data['opening_efficiency'] < 0.2), 'efficiency_regime'] = 'low'
    
    # Regime Transition Signals
    data['efficiency_momentum'] = data['intraday_efficiency'] - data['intraday_efficiency'].shift(1)
    data['opening_strength'] = data['opening_efficiency'] * np.sign(data['open'] - data['close'].shift(1))
    data['closing_pressure'] = data['closing_efficiency'] * np.sign(data['close'] - data['open'])
    data = data.fillna(0)
    
    # 3. Volume-Price Divergence Dynamics
    # Divergence Detection
    data['price_volume_divergence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['amount_price_divergence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['amount'] - data['amount'].shift(1))
    data['efficiency_volume_divergence'] = np.sign(data['intraday_efficiency'] - data['intraday_efficiency'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data = data.fillna(0)
    
    # Divergence Strength
    data['price_divergence_magnitude'] = np.abs(data['close'] - data['close'].shift(1)) * data['price_volume_divergence']
    data['volume_divergence_magnitude'] = np.abs(data['volume'] - data['volume'].shift(1)) * data['price_volume_divergence']
    data['efficiency_divergence'] = data['intraday_efficiency'] * data['efficiency_volume_divergence']
    data = data.fillna(0)
    
    # Convergence Patterns
    data['positive_convergence'] = ((data['price_volume_divergence'] > 0) & (data['amount_price_divergence'] > 0)).astype(int)
    data['negative_convergence'] = ((data['price_volume_divergence'] < 0) & (data['amount_price_divergence'] < 0)).astype(int)
    data['mixed_signals'] = (~(data['positive_convergence'] | data['negative_convergence'])).astype(int)
    
    # 4. Regime-Adaptive Signal Construction
    data['regime_signal'] = 0
    
    # High Efficiency Regime
    high_mask = data['efficiency_regime'] == 'high'
    data.loc[high_mask, 'regime_signal'] = (
        data.loc[high_mask, 'volatility_skew'] * 
        data.loc[high_mask, 'directional_persistence'] * 
        data.loc[high_mask, 'opening_strength'] * 
        data.loc[high_mask, 'positive_convergence']
    )
    
    # Low Efficiency Regime
    low_mask = data['efficiency_regime'] == 'low'
    data.loc[low_mask, 'regime_signal'] = (
        data.loc[low_mask, 'volume_efficiency_ratio'] * 
        data.loc[low_mask, 'efficiency_divergence'] * 
        data.loc[low_mask, 'closing_pressure'] * 
        data.loc[low_mask, 'negative_convergence']
    )
    
    # Mixed Efficiency Regime
    mixed_mask = data['efficiency_regime'] == 'mixed'
    asymmetry_component = data.loc[mixed_mask, 'volatility_skew'] * data.loc[mixed_mask, 'volume_efficiency_ratio']
    divergence_component = data.loc[mixed_mask, 'price_divergence_magnitude'] * data.loc[mixed_mask, 'efficiency_divergence']
    data.loc[mixed_mask, 'regime_signal'] = asymmetry_component * divergence_component * data.loc[mixed_mask, 'efficiency_momentum']
    
    # 5. Multi-scale Validation Framework
    # Micro-scale Validation
    data['opening_gap_validation'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    data['volume_spike_confirmation'] = (data['volume'] / data['volume'].shift(1)) * data['price_volume_divergence']
    data['efficiency_reversal'] = np.sign(data['intraday_efficiency'] - data['intraday_efficiency'].shift(1)) * np.sign(data['close'] - data['open'])
    data = data.fillna(0)
    
    # Short-term Validation (5-day windows)
    data['volatility_clustering'] = 0
    data['volume_persistence'] = 0
    data['efficiency_trend'] = 0
    
    for i in range(len(data)):
        if i >= 4:
            window = data.iloc[i-4:i+1]
            # Volatility clustering
            vol_range = window['high'] - window['low']
            data.loc[data.index[i], 'volatility_clustering'] = vol_range.std() / vol_range.mean() if vol_range.mean() > 0 else 0
            
            # Volume persistence
            volume_persist = 0
            for j in range(i-3, i+1):
                if j > 0 and data.iloc[j]['volume'] > data.iloc[j-1]['volume']:
                    volume_persist += 1
            data.loc[data.index[i], 'volume_persistence'] = volume_persist
            
            # Efficiency trend
            eff_trend = 0
            for j in range(i-3, i+1):
                if j > 0 and data.iloc[j]['intraday_efficiency'] > data.iloc[j-1]['intraday_efficiency']:
                    eff_trend += 1
            data.loc[data.index[i], 'efficiency_trend'] = eff_trend
    
    # Medium-term Alignment (10-day windows)
    data['directional_consistency'] = 0
    data['efficiency_regime_stability'] = 0
    
    for i in range(len(data)):
        if i >= 9:
            window = data.iloc[i-9:i+1]
            # Directional consistency
            up_days = (window['close'] > window['close'].shift(1)).sum()
            down_days = (window['close'] < window['close'].shift(1)).sum()
            data.loc[data.index[i], 'directional_consistency'] = up_days - down_days
            
            # Efficiency regime stability
            regime_stability = 0
            for j in range(i-8, i+1):
                if j > 0 and data.iloc[j]['efficiency_regime'] == data.iloc[j-1]['efficiency_regime']:
                    regime_stability += 1
            data.loc[data.index[i], 'efficiency_regime_stability'] = regime_stability
    
    # 6. Composite Alpha Generation
    # Multi-scale Confidence Weighting
    data['micro_confidence'] = data['opening_gap_validation'] * data['volume_spike_confirmation']
    data['short_term_confidence'] = data['volatility_clustering'] * data['volume_persistence']
    data['medium_term_confidence'] = data['directional_consistency'] * data['efficiency_regime_stability']
    
    # Apply confidence weighting
    confidence_weighted_signal = (
        data['regime_signal'] * 
        (1 + data['micro_confidence'] * 0.1) *
        (1 + data['short_term_confidence'] * 0.05) *
        (1 + data['medium_term_confidence'] * 0.02)
    )
    
    # Divergence-Convergence Filtering
    filtered_signal = confidence_weighted_signal.copy()
    filtered_signal *= (1 + data['positive_convergence'] * 0.15)
    filtered_signal *= (1 - np.abs(data['negative_convergence']) * 0.1)
    filtered_signal *= (1 + data['efficiency_reversal'] * 0.05)
    
    # Volatility-Volume Coupling Enhancement
    volume_adjusted_signal = filtered_signal * data['volume_efficiency_ratio']
    volatility_enhanced_signal = volume_adjusted_signal * (1 + data['volatility_skew'] * 0.1)
    efficiency_momentum_enhanced = volatility_enhanced_signal * (1 + data['efficiency_momentum'] * 0.08)
    
    # Final Factor Output
    final_factor = efficiency_momentum_enhanced * (1 + data['directional_persistence'] * 0.05)
    final_factor *= (1 + data['efficiency_trend'] * 0.03)
    
    # Clean infinite values and return
    final_factor = final_factor.replace([np.inf, -np.inf], 0)
    final_factor = final_factor.fillna(0)
    
    return pd.Series(final_factor, index=data.index)
