import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate returns for various timeframes
    data['ret_1'] = data['close'] / data['close'].shift(1) - 1
    data['ret_2'] = data['close'] / data['close'].shift(2) - 1
    data['ret_3'] = data['close'] / data['close'].shift(3) - 1
    data['ret_8'] = data['close'] / data['close'].shift(8) - 1
    data['ret_10'] = data['close'] / data['close'].shift(10) - 1
    
    # Multi-timeframe momentum asymmetry
    data['short_term_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    data['momentum_asymmetry'] = (data['short_term_momentum'] - data['medium_term_momentum']) / (
        abs(data['short_term_momentum']) + abs(data['medium_term_momentum']) + 0.001)
    
    # Price reversal acceleration
    data['reversal_acceleration'] = data['ret_2'] - data['ret_10'] / 5
    data['reversal_strength'] = np.sign(data['momentum_asymmetry']) * data['reversal_acceleration']
    data['directional_consistency'] = np.sign(data['short_term_momentum']) * np.sign(data['medium_term_momentum'])
    
    # Asymmetric Volatility Weighting
    # Calculate rolling volatility metrics
    data['positive_returns'] = data['ret_1'].where(data['ret_1'] > 0, 0)
    data['negative_returns'] = data['ret_1'].where(data['ret_1'] < 0, 0)
    
    data['upside_vol'] = data['positive_returns'].rolling(window=10, min_periods=5).std()
    data['downside_vol'] = abs(data['negative_returns'].rolling(window=10, min_periods=5).std())
    data['volatility_asymmetry'] = data['upside_vol'] / (data['downside_vol'] + 0.001)
    
    # Volatility regime classification
    conditions = [
        data['volatility_asymmetry'] > 1.5,
        data['volatility_asymmetry'] < 0.67
    ]
    choices = ['bull', 'bear']
    data['volatility_regime'] = np.select(conditions, choices, default='balanced')
    
    # Regime-adaptive scaling
    data['regime_scaled_momentum_reversal'] = np.where(
        data['volatility_regime'] == 'bull',
        data['reversal_strength'] * (1 + 1 / data['volatility_asymmetry']),
        np.where(
            data['volatility_regime'] == 'bear',
            data['reversal_strength'] * data['volatility_asymmetry'],
            data['reversal_strength'] * (1 + data['directional_consistency'])
        )
    )
    
    # Liquidity-Price Convergence Analysis
    # Volume acceleration framework
    data['volume_momentum_3'] = (data['volume'] - data['volume'].shift(3)) / (data['volume'].shift(3) + 0.001)
    data['volume_momentum_8'] = (data['volume'] - data['volume'].shift(8)) / (data['volume'].shift(8) + 0.001)
    data['volume_acceleration'] = data['volume_momentum_3'] - data['volume_momentum_8']
    
    # Calculate volume persistence (consecutive increases)
    data['volume_increase'] = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['volume_persistence'] = data['volume_increase'].rolling(window=5, min_periods=1).apply(
        lambda x: x[::-1].cumprod().sum(), raw=False
    )
    
    # Amount-based liquidity metrics
    data['amount_volume_ratio'] = data['amount'] / (data['volume'] + 0.001)
    data['trade_size_trend'] = data['amount_volume_ratio'] / data['amount_volume_ratio'].shift(5) - 1
    
    # Large order concentration
    data['amount_mean_5'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['amount_std_5'] = data['amount'].rolling(window=5, min_periods=3).std()
    data['large_order_concentration'] = data['amount_std_5'] / (data['amount_mean_5'] + 0.001)
    
    # Liquidity efficiency
    data['liquidity_efficiency'] = data['ret_1'] / (data['amount'] + 0.001)
    
    # Price-liquidity convergence
    data['price_acceleration'] = data['ret_3'] - data['ret_8']
    data['direction_alignment'] = np.sign(data['price_acceleration']) * np.sign(data['volume_acceleration'])
    data['magnitude_convergence'] = abs(data['price_acceleration']) / (abs(data['volume_acceleration']) + 0.001)
    data['efficiency_weighted_confirmation'] = data['direction_alignment'] * data['liquidity_efficiency']
    
    # Multi-Timeframe Regime Integration
    # Volatility regime consistency
    data['vol_5day'] = data['ret_1'].rolling(window=5, min_periods=3).std()
    data['vol_15day'] = data['ret_1'].rolling(window=15, min_periods=8).std()
    data['vol_regime_5day'] = np.where(data['vol_5day'] > data['vol_15day'], 'high', 'low')
    data['vol_regime_15day'] = np.where(data['vol_15day'] > data['vol_15day'].shift(15), 'rising', 'falling')
    
    # Calculate regime stability (simplified)
    data['regime_stability'] = data['volatility_regime'].eq(data['volatility_regime'].shift(1)).rolling(
        window=10, min_periods=5
    ).mean()
    
    # Convergence confidence weighting
    data['timeframe_alignment'] = np.sign(data['short_term_momentum']) * np.sign(data['medium_term_momentum'])
    data['high_confidence'] = data['timeframe_alignment'] * data['regime_stability']
    data['medium_confidence'] = data['direction_alignment'] * data['volume_persistence']
    data['low_confidence'] = data['efficiency_weighted_confirmation']
    
    data['convergence_confidence_weight'] = (
        data['high_confidence'] * 0.5 + data['medium_confidence'] * 0.3 + data['low_confidence'] * 0.2
    )
    
    # Dynamic Factor Synthesis
    # Core momentum-reversal construction
    data['price_liquidity_convergence_score'] = (
        data['direction_alignment'] * data['magnitude_convergence'] * data['efficiency_weighted_confirmation']
    )
    
    # Regime-optimized enhancement
    data['core_factor'] = (
        data['regime_scaled_momentum_reversal'] * 
        data['price_liquidity_convergence_score'] * 
        data['convergence_confidence_weight']
    )
    
    data['enhanced_factor'] = np.where(
        data['volatility_regime'] == 'bull',
        data['core_factor'] * (1 + data['trade_size_trend']),
        np.where(
            data['volatility_regime'] == 'bear',
            data['core_factor'] * (1 - data['large_order_concentration']),
            data['core_factor'] * data['regime_stability']
        )
    )
    
    # Adaptive final scaling
    data['final_factor'] = data['enhanced_factor']
    
    # Volatility dampening during high concentration
    high_concentration = data['large_order_concentration'] > data['large_order_concentration'].rolling(20).quantile(0.8)
    data.loc[high_concentration, 'final_factor'] = data.loc[high_concentration, 'final_factor'] * 0.7
    
    # Volume acceleration during high persistence
    high_persistence = data['volume_persistence'] > 3
    data.loc[high_persistence, 'final_factor'] = data.loc[high_persistence, 'final_factor'] * 1.2
    
    # Efficiency enhancement during stable convergence
    stable_convergence = (data['regime_stability'] > 0.7) & (abs(data['direction_alignment']) > 0.5)
    data.loc[stable_convergence, 'final_factor'] = data.loc[stable_convergence, 'final_factor'] * 1.1
    
    # Predictive Alpha Generation
    alpha = data['final_factor'].copy()
    
    return alpha
