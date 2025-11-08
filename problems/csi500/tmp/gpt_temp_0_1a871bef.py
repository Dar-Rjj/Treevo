import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Momentum-Liquidity Regime Factor v2
    Multi-scale regime-adaptive factor combining volatility, volume, momentum, and microstructure signals
    """
    data = df.copy()
    
    # Volatility Spectrum Analysis
    data['ultra_short_vol'] = abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['medium_term_vol'] = data['close'].rolling(window=5).std()
    data['volatility_regime'] = (data['ultra_short_vol'] > data['medium_term_vol']).astype(int)
    
    # Volume Dynamics Classification
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_trend'] = data['volume'] / data['volume'].rolling(window=5).mean() - 1
    data['volume_regime'] = (data['volume_acceleration'] * data['volume_trend'] > 0).astype(int)
    
    # Asymmetric Momentum Patterns
    # Directional Momentum Strength
    up_mask = data['close'] > data['close'].shift(1)
    down_mask = data['close'] < data['close'].shift(1)
    
    data['up_momentum_intensity'] = 0
    data.loc[up_mask, 'up_momentum_intensity'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    data['down_momentum_resistance'] = 0
    data.loc[down_mask, 'down_momentum_resistance'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    
    data['momentum_asymmetry'] = data['up_momentum_intensity'] - data['down_momentum_resistance']
    
    # Gap Behavior Analysis
    data['opening_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_momentum_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['gap_persistence'] = data['opening_momentum'] * data['intraday_momentum_efficiency']
    
    # Microstructure Liquidity Signals
    # Price Impact Dynamics
    data['volume_price_impact'] = abs(data['close'] - data['close'].shift(1)) / (data['amount'] + 1e-8)
    
    # Large trade impact calculation
    data['amount_rolling_mean'] = data['amount'].rolling(window=5).mean()
    large_trade_mask = data['amount'] > data['amount_rolling_mean']
    data['large_trade_impact'] = (
        data['amount'].rolling(window=3).apply(
            lambda x: np.sum(x[x > x.mean()]) / (np.sum(x) + 1e-8) if len(x) == 3 else np.nan
        )
    )
    data['impact_asymmetry'] = data['volume_price_impact'] * data['large_trade_impact']
    
    # Execution Quality Metrics
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap_deviation'] = abs(data['close'] - data['vwap']) / (data['close'] + 1e-8)
    data['execution_efficiency'] = 1 - data['vwap_deviation']
    data['quality_momentum'] = data['execution_efficiency'] / data['execution_efficiency'].shift(3) - 1
    
    # Price Discovery Patterns
    # Information Absorption
    data['early_session_info'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['late_session_adjustment'] = abs(data['close'] - (data['high'] + data['low'])/2) / (data['close'] + 1e-8)
    data['information_efficiency'] = data['early_session_info'] / (data['late_session_adjustment'] + 0.001)
    
    # Trend Quality Assessment
    data['trend_consistency'] = (
        (data['close'] > data['close'].shift(1)).rolling(window=3).sum() / 3
    )
    data['volatility_adjusted_trend'] = data['trend_consistency'] / (data['medium_term_vol'] + 0.001)
    data['quality_score'] = data['volatility_adjusted_trend'] * data['execution_efficiency']
    
    # Regime-Adaptive Signal Generation
    # Volatility-Regime Signals
    data['high_vol_signal'] = data['momentum_asymmetry'] * data['information_efficiency']
    data['low_vol_signal'] = data['gap_persistence'] * data['quality_score']
    data['volatility_adaptive_core'] = np.where(
        data['volatility_regime'] == 1, 
        data['high_vol_signal'], 
        data['low_vol_signal']
    )
    
    # Volume-Regime Enhancement
    data['high_volume_signal'] = data['impact_asymmetry'] * data['quality_momentum']
    data['low_volume_signal'] = data['execution_efficiency'] * data['trend_consistency']
    data['volume_adaptive_component'] = np.where(
        data['volume_regime'] == 1,
        data['high_volume_signal'],
        data['low_volume_signal']
    )
    
    # Multi-Factor Integration
    # Core Factor Construction
    data['base_integration'] = data['volatility_adaptive_core'] * data['volume_adaptive_component']
    data['microstructure_adjusted'] = data['base_integration'] * data['execution_efficiency']
    data['trend_confirmed_factor'] = data['microstructure_adjusted'] * data['volatility_adjusted_trend']
    
    # Signal Refinement
    data['momentum_alignment'] = np.sign(data['opening_momentum']) * np.sign(data['intraday_momentum_efficiency'])
    data['volume_confirmation'] = np.where(data['volume_acceleration'] > 0, 1, -1)
    
    # Final factor
    data['final_factor'] = (
        data['trend_confirmed_factor'] * 
        data['momentum_alignment'] * 
        data['volume_confirmation']
    )
    
    return data['final_factor']
