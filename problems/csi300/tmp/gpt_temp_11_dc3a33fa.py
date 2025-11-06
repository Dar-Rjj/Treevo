import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Regime-Adaptive Cross-Association System
    """
    data = df.copy()
    
    # Cross-Association Momentum & Divergence
    # Short-term Price-Volume Association (3-day)
    data['price_change_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['volume_change_3d'] = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    data['alignment_score'] = np.sign(data['price_change_3d']) * np.sign(data['volume_change_3d']) * \
                             np.abs(data['price_change_3d'] * data['volume_change_3d'])
    
    # 3-day return × Volume correlation
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['volume_corr_3d'] = data['close'].rolling(window=3).corr(data['volume'])
    data['association_momentum'] = data['return_3d'] * data['volume_corr_3d']
    
    # Medium-term Association Divergence (5-day vs 10-day)
    data['corr_5d'] = data['close'].rolling(window=5).corr(data['volume'])
    data['corr_10d'] = data['close'].rolling(window=10).corr(data['volume'])
    data['corr_divergence'] = data['corr_5d'] - data['corr_10d']
    data['corr_magnitude_change'] = data['corr_10d'].diff(10)
    
    # Multi-timeframe Integration
    data['cross_association_product'] = data['alignment_score'] * data['corr_divergence']
    data['volume_avg_10d'] = data['volume'].rolling(window=10).mean()
    data['volume_confirmation'] = data['volume'] / data['volume_avg_10d']
    data['cross_association_signal'] = np.tanh(data['cross_association_product'] * data['volume_confirmation'])
    
    # Dynamic Regime Detection & Efficiency Analysis
    # Volatility-Efficiency Regime
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['price_efficiency'] = np.abs(data['close'] - data['prev_close']) / data['true_range']
    data['efficiency_ratio'] = data['price_efficiency'].rolling(window=5).sum() / \
                              data['price_efficiency'].rolling(window=20).sum()
    
    # Volatility Persistence
    data['true_range_20d'] = data['true_range'].rolling(window=20).mean()
    data['vol_persistence'] = data['true_range_20d'].diff().rolling(window=2).corr()
    
    # Regime Classification
    data['trending_regime'] = (data['vol_persistence'] > 0.2).astype(int)
    data['mean_reversion_regime'] = (data['vol_persistence'] <= 0.2).astype(int)
    
    # Liquidity-Microstructure Regime
    data['price_per_volume'] = data['amount'] / data['volume']
    data['info_asymmetry'] = (data['high'] + data['low'] - 2 * data['close'])**2 / \
                            ((data['high'] - data['low']) * (data['volume']**(1/3)))
    data['info_asymmetry'] = data['info_asymmetry'].replace([np.inf, -np.inf], np.nan)
    
    # Order Flow Entropy (simplified)
    data['signed_volume_change'] = data['volume'].diff(3)
    data['volume_entropy'] = -data['signed_volume_change'].rolling(window=3).apply(
        lambda x: np.sum(x * np.log(np.abs(x) + 1e-10)) if np.all(x != 0) else 0
    )
    
    # Volume-Spread Correlation
    data['spread'] = data['high'] - data['low']
    data['volume_spread_corr'] = data['spread'].rolling(window=20).corr(data['volume'])
    
    # Market Microstructure Patterns
    data['opening_pressure'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['closing_stability'] = (data['close'] - data['open']) / data['open']
    data['execution_quality'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    
    # Regime-Adaptive Signal Construction
    # Volatility-Efficiency Adjustment
    data['volume_bias'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    data['trending_signal'] = data['cross_association_signal'] * data['efficiency_ratio'] * data['volume_bias']
    data['mean_reversion_signal'] = -data['cross_association_signal'] * data['vol_persistence'] * (1 - data['efficiency_ratio'])
    
    data['vol_efficiency_signal'] = data['trending_regime'] * data['trending_signal'] + \
                                   data['mean_reversion_regime'] * data['mean_reversion_signal']
    
    # Liquidity-Microstructure Enhancement
    data['liquidity_threshold'] = data['price_per_volume'].rolling(window=20).quantile(0.7)
    data['high_liquidity'] = (data['price_per_volume'] < data['liquidity_threshold']).astype(int)
    data['low_liquidity'] = (data['price_per_volume'] >= data['liquidity_threshold']).astype(int)
    
    data['liquidity_signal'] = data['high_liquidity'] * data['cross_association_signal'] + \
                              data['low_liquidity'] * data['info_asymmetry']
    
    # Regime Confidence
    data['regime_confidence'] = np.abs(data['corr_magnitude_change']) * (1 - data['volume_entropy'])
    
    # Multi-Regime Integration
    data['regime_signal'] = data['vol_efficiency_signal'] * data['liquidity_signal'] * data['regime_confidence']
    
    # Cross-Timeframe Pattern Synthesis
    # Momentum-Efficiency Dynamics
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_acceleration'] = data['momentum_5d'] - data['momentum_5d'].shift(1)
    
    data['efficiency_divergence'] = np.abs(
        (data['close'] / data['close'].shift(5) - 1) - 
        (data['close'] / data['close'].shift(20) - 1)
    )
    
    data['momentum_factor'] = data['momentum_acceleration'] * data['efficiency_divergence'] * data['regime_confidence']
    
    # Price Recovery & Microstructure Confirmation
    data['gap_reversion'] = (data['open'] / data['prev_close'] - 1) * (data['close'] / data['open'] - 1)
    data['recovery_strength'] = (data['high'] - data['close'] + data['close'] - data['low']) / data['close']
    data['combined_recovery'] = data['recovery_strength'] * (1 - data['price_efficiency']) * data['info_asymmetry']
    
    # Final Factor Integration
    data['microstructure_scaling'] = data['regime_signal'] * data['info_asymmetry'] / (data['volume_entropy'] + 1e-10)
    
    # Multi-timeframe Consistency
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['directional_agreement'] = (
        np.sign(data['momentum_3d']) * np.sign(data['momentum_5d']) * np.sign(data['momentum_10d'])
    )
    
    # Final Factor
    data['final_factor'] = (
        data['microstructure_scaling'] * 
        data['momentum_factor'] * 
        data['combined_recovery'] * 
        data['directional_agreement']
    )
    
    return data['final_factor']
