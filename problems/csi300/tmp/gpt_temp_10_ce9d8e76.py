import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Basic calculations
    data['range'] = data['high'] - data['low']
    data['directional_efficiency'] = (data['close'] - data['open']) / data['range']
    data['gap_absorption'] = abs(data['open'] - data['close'].shift(1)) / data['range']
    
    # Volatility Regime & Efficiency Analysis
    # Short-term volatility efficiency
    data['range_5d_avg'] = data['range'].rolling(window=5).mean()
    data['range_persistence'] = (data['range'] > data['range_5d_avg']).rolling(window=5).sum() / 5
    
    # Medium-term volatility classification
    data['range_20d_avg'] = data['range'].rolling(window=20).mean()
    data['volatility_ratio'] = data['range_5d_avg'] / data['range_20d_avg']
    
    # Efficiency persistence (autocorrelation of directional efficiency)
    data['efficiency_persistence'] = data['directional_efficiency'].rolling(window=20).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 20 and not x.isna().any() else np.nan, raw=False
    )
    
    # Multi-scale efficiency products
    data['intraday_efficiency'] = data['directional_efficiency'] * data['gap_absorption']
    data['volatility_adjusted_efficiency'] = data['directional_efficiency'] / data['volatility_ratio']
    data['efficiency_momentum'] = data['directional_efficiency'] / data['directional_efficiency'].shift(4) - 1
    
    # Multi-Scale Momentum Divergence
    # Price momentum analysis
    data['momentum_5d'] = data['close'] / data['close'].shift(4) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(9) - 1
    data['momentum_20d'] = data['close'] / data['close'].shift(19) - 1
    data['momentum_acceleration'] = (data['momentum_5d'] / data['momentum_20d']) - 1
    
    # Volume-price divergence
    data['intraday_divergence'] = data['directional_efficiency'] * (data['volume'] / data['range'])
    data['divergence_5d'] = data['momentum_5d'] * (data['volume'] / data['volume'].shift(4) - 1)
    data['divergence_20d'] = data['momentum_20d'] * (data['volume'] / data['volume'].shift(19) - 1)
    
    # Cross-scale divergence products
    data['short_term_divergence'] = data['intraday_divergence'] * data['divergence_5d']
    data['multi_scale_divergence'] = data['intraday_divergence'] * data['divergence_5d'] * data['divergence_20d']
    data['divergence_momentum'] = data['divergence_5d'] / data['divergence_20d'] - 1
    
    # Amount-Based Efficiency Confirmation
    # Amount flow characteristics
    data['amount_persistence'] = data['amount'] / data['amount'].shift(1) - 1
    data['amount_volatility'] = abs(data['amount'] - data['amount'].shift(1)) / data['amount'].shift(1)
    data['amount_efficiency'] = abs(data['close'] - data['open']) / data['amount']
    
    # Volume-amount interaction
    data['volume_to_amount_ratio'] = data['volume'] / data['amount']
    data['ratio_momentum'] = (data['volume_to_amount_ratio'] / data['volume_to_amount_ratio'].shift(1)) - 1
    data['ratio_persistence'] = np.sign(data['volume_to_amount_ratio'] - data['volume_to_amount_ratio'].shift(1))
    
    # Amount-enhanced efficiency
    data['efficiency_amount_product'] = data['directional_efficiency'] * data['amount_efficiency']
    data['volatility_amount_alignment'] = data['volatility_ratio'] * data['amount_volatility']
    data['amount_weighted_divergence'] = data['intraday_divergence'] * data['amount_persistence']
    
    # Session-Based Momentum Alignment
    # Opening session dynamics
    data['opening_gap_efficiency'] = abs(data['open'] - data['close'].shift(1)) / data['range']
    data['opening_volume_intensity'] = data['volume'] / data['range']
    data['opening_alignment'] = np.sign(data['open'] - data['close'].shift(1)) * np.sign(data['close'] - data['open'])
    
    # Closing session efficiency
    data['closing_momentum'] = abs(data['close'] - (data['high'] + data['low'])/2) / data['range']
    data['closing_volume_concentration'] = data['volume'] / abs(data['close'] - data['open'])
    data['closing_efficiency'] = np.sign(data['close'] - data['open']) * np.sign(data['close'] - (data['high'] + data['low'])/2)
    
    # Session consistency
    data['session_consistency'] = data['opening_alignment'] * data['closing_efficiency']
    
    # Regime classification
    data['regime'] = 'transition'
    data.loc[data['volatility_ratio'] > 1.2, 'regime'] = 'high'
    data.loc[data['volatility_ratio'] < 0.8, 'regime'] = 'low'
    
    # Regime-Adaptive Composite Construction
    # High volatility regime signals
    data['strong_convergence_factor'] = (data['short_term_divergence'] * data['amount_volatility']) * data['opening_alignment']
    data['efficiency_momentum_premium'] = data['efficiency_momentum'] * data['volume_to_amount_ratio']
    data['volatility_enhanced_momentum'] = data['momentum_5d'] * data['volatility_adjusted_efficiency']
    
    # Low volatility regime signals
    data['emerging_divergence_factor'] = (data['divergence_20d'] * data['amount_efficiency']) * data['closing_efficiency']
    data['persistence_weighted_momentum'] = data['momentum_5d'] * data['ratio_persistence']
    data['efficiency_amount_alignment'] = data['efficiency_amount_product'] * data['opening_gap_efficiency']
    
    # Transition regime signals
    data['cross_scale_integration'] = data['multi_scale_divergence'] * data['session_consistency']
    data['volatility_efficiency_momentum'] = data['volatility_ratio'] * data['efficiency_momentum']
    data['amount_flow_divergence'] = data['amount_weighted_divergence'] * data['ratio_momentum']
    
    # Final composite factor construction
    high_vol_signal = data['strong_convergence_factor'] + data['efficiency_momentum_premium']
    low_vol_signal = data['emerging_divergence_factor'] + data['persistence_weighted_momentum']
    transition_signal = data['cross_scale_integration'] + data['volatility_efficiency_momentum']
    
    # Regime-weighted signal combination
    data['composite_factor'] = np.nan
    data.loc[data['regime'] == 'high', 'composite_factor'] = high_vol_signal
    data.loc[data['regime'] == 'low', 'composite_factor'] = low_vol_signal
    data.loc[data['regime'] == 'transition', 'composite_factor'] = transition_signal
    
    # Volume-amount confirmation layer
    data['final_factor'] = data['composite_factor'] * data['ratio_persistence'] * (1 + data['amount_persistence'])
    
    # Clean up and return
    result = data['final_factor'].copy()
    return result
