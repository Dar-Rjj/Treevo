import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volatility-regime adaptive momentum,
    price-volume fractal efficiency, extreme-tail risk momentum, and microstructure-informed reversal.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Regime Adaptive Momentum Component
    # Calculate true range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1) / data['prev_close']
    
    # Volatility regime
    data['vol_regime_5'] = data['true_range'].rolling(window=5).mean()
    data['vol_regime_20'] = data['true_range'].rolling(window=20).mean()
    data['regime_strength'] = np.where(data['vol_regime_5'] > data['vol_regime_20'], 1, -1)
    
    # Momentum quality measures
    data['momentum_accel'] = (data['close'] - data['close'].shift(5)) - (data['close'].shift(5) - data['close'].shift(10))
    
    # 5-day return autocorrelation for smoothness
    returns_5d = data['close'].pct_change(periods=5)
    data['momentum_smoothness'] = returns_5d.rolling(window=10).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Adaptive momentum signal
    data['adaptive_momentum'] = (data['momentum_accel'] * data['regime_strength'] * 
                                data['momentum_smoothness'].fillna(0) * data['volume'])
    
    # Price-Volume Fractal Efficiency Component
    # Volume fractal analysis
    data['volume_fractal'] = data['volume'] / data['volume'].shift(1)
    data['volume_momentum_div'] = (data['volume'].rolling(window=5).sum() / 
                                  data['volume'].rolling(window=10).sum() - 1)
    
    # Price path efficiency
    data['fractal_dim'] = np.log(data['high'] - data['low']) / np.log(data['volume'].replace(0, 1))
    price_changes = abs(data['close'].diff())
    data['path_efficiency'] = abs(data['close'] - data['close'].shift(5)) / price_changes.rolling(window=5).sum().replace(0, 1)
    
    # Fractal signal
    data['fractal_signal'] = (data['fractal_dim'] * data['path_efficiency'] * 
                             data['volume_momentum_div'].fillna(0))
    
    # Extreme-Tail Risk Momentum Component
    # Tail risk measures
    returns = data['close'].pct_change()
    vol_5d = returns.rolling(window=5).std()
    extreme_threshold = 2 * vol_5d
    data['extreme_move'] = abs(returns) > extreme_threshold
    data['tail_risk_exp'] = data['extreme_move'].rolling(window=20).sum() / 20
    
    # Risk-adjusted momentum
    data['risk_adj_momentum'] = returns.rolling(window=5).sum() / vol_5d.replace(0, 1)
    
    # Tail risk signal
    data['tail_risk_signal'] = (data['risk_adj_momentum'].fillna(0) * data['tail_risk_exp'] * 
                               data['volume'] / data['volume'].rolling(window=20).mean())
    
    # Microstructure-Informed Reversal Component
    # Price formation efficiency
    data['price_discovery_eff'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1)
    data['microstructure_noise'] = abs((data['high'] + data['low']) / 2 - data['close']) / data['close']
    
    # Conditional reversal
    data['conditional_reversal'] = np.sign(returns.shift(1)) * returns
    data['reversal_intensity'] = abs(data['conditional_reversal']) * data['volume']
    
    # Microstructure signal
    data['microstructure_signal'] = (data['conditional_reversal'].fillna(0) * 
                                    data['price_discovery_eff'] * data['microstructure_noise'])
    
    # Combine all components with equal weighting
    factors = ['adaptive_momentum', 'fractal_signal', 'tail_risk_signal', 'microstructure_signal']
    
    # Normalize each component
    for factor in factors:
        if data[factor].std() > 0:
            data[f'{factor}_norm'] = (data[factor] - data[factor].rolling(window=20).mean()) / data[factor].rolling(window=20).std()
        else:
            data[f'{factor}_norm'] = 0
    
    # Final composite factor
    data['composite_factor'] = (data['adaptive_momentum_norm'] + data['fractal_signal_norm'] + 
                               data['tail_risk_signal_norm'] + data['microstructure_signal_norm']) / 4
    
    # Clean up intermediate columns
    cols_to_drop = [col for col in data.columns if col not in ['composite_factor', 'close', 'volume']]
    data = data.drop(columns=cols_to_drop)
    
    return data['composite_factor']
