import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Efficiency Momentum
    # Intraday Efficiency Dynamics
    data['range_utilization'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_efficiency'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['closing_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Efficiency persistence
    data['efficiency_persistence'] = data['range_utilization'].rolling(window=3, min_periods=1).apply(
        lambda x: (x > 0.5).sum(), raw=True
    )
    
    # Gap-Driven Efficiency Patterns
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, np.nan)
    data['gap_range_efficiency'] = (data['high'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    data['post_gap_development'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    data['gap_resolution'] = data['close'] - data['open']
    
    # Efficiency Momentum Structure
    data['efficiency_momentum'] = data['range_utilization'] / data['range_utilization'].shift(3).replace(0, np.nan)
    data['efficiency_volatility'] = data['range_utilization'].rolling(window=5, min_periods=3).std()
    data['range_expansion'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    
    # Efficiency-consistency (simplified correlation)
    data['efficiency_consistency'] = data['range_utilization'].rolling(window=3, min_periods=2).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0, raw=False
    )
    
    # Multi-Timeframe Momentum Elasticity
    # Multi-Period Momentum
    data['ultra_short_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(8) - 1
    data['long_term_momentum'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Elasticity Framework
    data['short_medium_elasticity'] = data['short_term_momentum'] / data['medium_term_momentum'].replace(0, np.nan)
    data['medium_long_elasticity'] = data['medium_term_momentum'] / data['long_term_momentum'].replace(0, np.nan)
    data['elasticity_curvature'] = (data['short_medium_elasticity'] - data['medium_long_elasticity']) / abs(data['medium_long_elasticity']).replace(0, np.nan)
    
    # Elasticity persistence
    def count_consistent_elasticity(x):
        if len(x) < 2:
            return 0
        signs = np.sign(x)
        return (signs == signs[0]).sum() if signs[0] != 0 else 0
    
    data['elasticity_persistence'] = data['elasticity_curvature'].rolling(window=3, min_periods=2).apply(
        count_consistent_elasticity, raw=True
    )
    
    # Quality-Adjusted Elasticity
    data['intraday_momentum_persistence'] = data['ultra_short_momentum'].rolling(window=3, min_periods=2).apply(
        lambda x: (np.sign(x) == np.sign(x[0])).sum() if len(x) > 0 and x[0] != 0 else 0, raw=True
    )
    data['efficiency_weighted_elasticity'] = data['elasticity_curvature'] * data['efficiency_persistence']
    data['range_confirmed_elasticity'] = data['elasticity_curvature'] * data['range_expansion']
    data['gap_sustained_elasticity'] = data['elasticity_curvature'] * data['gap_resolution']
    
    # Volume-Price Efficiency Dynamics
    # Volume-Driven Efficiency
    data['volume_weighted_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)) * data['volume']
    data['pure_price_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_divergence'] = data['volume_weighted_efficiency'] - data['pure_price_efficiency']
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Participation Structure
    up_days = data['close'] > data['open']
    data['up_volume'] = np.where(up_days, data['volume'], 0)
    data['up_volume_concentration'] = data['up_volume'].rolling(window=5, min_periods=3).sum() / data['volume'].rolling(window=5, min_periods=3).sum().replace(0, np.nan)
    
    down_days_volume_increase = (data['close'] < data['open']) & (data['volume'] > data['volume'].shift(1))
    data['volume_asymmetry'] = data['up_volume_concentration'] / down_days_volume_increase.rolling(window=5, min_periods=3).sum().replace(0, np.nan)
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) / (data['volume'].shift(1) / data['volume'].shift(2)).replace(0, np.nan)
    
    # Liquidity Efficiency
    data['liquidity_momentum'] = data['amount'] / data['amount'].shift(5) - 1
    data['volume_range_alignment'] = data['volume_momentum'] * data['range_expansion']
    data['liquidity_persistence'] = (data['amount'] > data['amount'].shift(1)).rolling(window=3, min_periods=2).sum()
    data['dynamic_liquidity_adjustment'] = data['volume'] / data['volume'].shift(20)
    
    # Dynamic Regime Classification
    # Volume Regime Analysis
    data['volume_regime'] = np.where(
        data['volume'] > data['volume'].shift(5) * 1.2, 'high',
        np.where(data['volume'] < data['volume'].shift(5) * 0.8, 'low', 'normal')
    )
    
    # Efficiency-Volume Elasticity (simplified)
    data['efficiency_volume_correlation'] = data['range_utilization'].rolling(window=5, min_periods=3).corr(data['volume'])
    data['elasticity_persistence_vol'] = (data['efficiency_volume_correlation'] > 0).rolling(window=3, min_periods=2).sum()
    data['elasticity_momentum_vol'] = data['efficiency_volume_correlation'] / data['efficiency_volume_correlation'].shift(3).replace(0, np.nan)
    
    # Price Level Classification
    data['price_level_20'] = data['close'] / data['close'].rolling(window=20, min_periods=10).mean()
    data['regime_stability'] = (data['volume_regime'] == data['volume_regime'].shift(1)).rolling(window=3, min_periods=2).sum()
    
    # Regime-Adaptive Signal Integration
    # Core Efficiency-Momentum Integration
    data['efficiency_weighted_elasticity_final'] = data['efficiency_weighted_elasticity'] * data['efficiency_persistence']
    data['volume_confirmed_momentum'] = data['short_term_momentum'] * data['volume_momentum']
    data['gap_sustained_efficiency_final'] = data['gap_resolution'] * data['gap_range_efficiency']
    data['range_aligned_elasticity'] = data['elasticity_curvature'] * data['range_expansion']
    
    # Regime-Specific Weighting
    high_volume_mask = data['volume_regime'] == 'high'
    low_volume_mask = data['volume_regime'] == 'low'
    normal_volume_mask = data['volume_regime'] == 'normal'
    
    # Enhanced Predictive Signals
    data['high_volume_signal'] = data['volume_confirmed_momentum'] * data['efficiency_volume_correlation']
    data['low_volume_signal'] = data['efficiency_weighted_elasticity_final'] * data['gap_sustained_efficiency_final']
    data['normal_volume_signal'] = data['efficiency_weighted_elasticity_final'] * data['volume_acceleration']
    
    # Combine signals based on regime
    data['enhanced_signal'] = np.where(
        high_volume_mask, data['high_volume_signal'],
        np.where(low_volume_mask, data['low_volume_signal'], data['normal_volume_signal'])
    )
    
    # Sustainability and Persistence Analysis
    data['momentum_sustainability'] = (
        data['elasticity_persistence'] * 
        data['efficiency_persistence'] * 
        data['range_expansion'] * 
        data['volume_acceleration'] * 
        data['liquidity_persistence']
    )
    
    # Mean Reversion Framework
    data['short_term_overextension'] = -data['ultra_short_momentum'].rolling(window=3, min_periods=2).mean()
    data['momentum_exhaustion'] = -data['elasticity_curvature']
    data['efficiency_reversion'] = -data['efficiency_momentum']
    data['volume_saturation'] = -data['volume_acceleration']
    
    data['mean_reversion_factor'] = (
        data['short_term_overextension'] + 
        data['momentum_exhaustion'] + 
        data['efficiency_reversion'] + 
        data['volume_saturation']
    ) / 4
    
    # Final Alpha Scoring
    data['core_alpha'] = data['enhanced_signal'] * data['momentum_sustainability']
    data['regime_adaptive_composite'] = data['core_alpha'] * (1 + data['mean_reversion_factor'])
    data['sustainability_confirmation'] = data['regime_adaptive_composite'] * data['momentum_sustainability']
    data['volume_credibility_adjustment'] = data['sustainability_confirmation'] * data['volume_acceleration']
    data['final_alpha'] = data['volume_credibility_adjustment'] * data['efficiency_consistency']
    
    # Clean up and return
    alpha_series = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
