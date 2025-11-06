import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Asymmetry Patterns
    # Upside volatility dominance
    data['upside_vol_dominance'] = ((data['high'] - data['close']) / (data['high'] - data['low'])) - \
                                  ((data['close'] - data['low']) / (data['high'] - data['low']))
    
    # Downside pressure persistence
    data['close_below_open'] = (data['close'] < data['open']).astype(int)
    data['downside_pressure'] = data['close_below_open'].rolling(window=5, min_periods=1).sum() * \
                               ((data['low'] - data['close'].shift(1)) / (data['high'] - data['low']))
    
    # Volatility regime shift
    data['range'] = data['high'] - data['low']
    data['vol_regime_shift'] = (data['range'] / data['range'].shift(4).rolling(window=5, min_periods=1).mean()) - \
                              (data['range'].shift(1) / data['range'].shift(5).rolling(window=5, min_periods=1).mean())
    
    # Price-Volume Divergence Signals
    # Volume-price decoupling
    data['volume_change'] = (data['volume'] / data['volume'].shift(1)) - 1
    data['price_change'] = (data['close'] / data['close'].shift(1)) - 1
    data['volume_price_decoupling'] = data['volume_change'] - (data['price_change'] * np.sign(data['price_change']))
    
    # Divergence momentum
    data['divergence_momentum'] = np.where(
        np.sign(data['volume_change']) != np.sign(data['price_change']),
        data['volume_change'] * data['price_change'],
        0
    )
    
    # Persistent divergence
    data['divergence_sign'] = np.sign(data['volume_change'] * data['price_change'])
    data['consecutive_divergence'] = 0
    for i in range(1, len(data)):
        if data['divergence_sign'].iloc[i] < 0:
            if data['divergence_sign'].iloc[i-1] < 0:
                data['consecutive_divergence'].iloc[i] = data['consecutive_divergence'].iloc[i-1] + 1
            else:
                data['consecutive_divergence'].iloc[i] = 1
        else:
            data['consecutive_divergence'].iloc[i] = 0
    
    # Gap Behavior Analysis
    # Gap absorption efficiency
    data['gap_absorption'] = ((data['close'] - data['open']) / (data['open'] - data['close'].shift(1))) * \
                           (data['volume'] / data['volume'].shift(1))
    
    # Gap reversal probability
    data['gap_open'] = (data['open'] / data['close'].shift(1)) - 1
    data['gap_close'] = (data['close'] / data['open']) - 1
    data['gap_reversal'] = np.where(
        np.sign(data['gap_open']) != np.sign(data['gap_close']),
        data['gap_open'] * data['gap_close'],
        0
    )
    
    # Gap momentum persistence
    data['gap_direction'] = np.sign(data['gap_open'])
    data['consecutive_gap'] = 0
    for i in range(1, len(data)):
        if data['gap_direction'].iloc[i] != 0:
            if data['gap_direction'].iloc[i] == data['gap_direction'].iloc[i-1]:
                data['consecutive_gap'].iloc[i] = data['consecutive_gap'].iloc[i-1] + 1
            else:
                data['consecutive_gap'].iloc[i] = 1
        else:
            data['consecutive_gap'].iloc[i] = 0
    
    data['gap_momentum'] = data['consecutive_gap'] * data['gap_open']
    
    # Asymmetric Response Regimes
    # Volatility expansion regime
    data['vol_expansion'] = data['vol_regime_shift'] * data['upside_vol_dominance'] * data['volume_price_decoupling']
    
    # Mean reversion regime
    data['mean_reversion'] = data['downside_pressure'] * data['gap_absorption'] * data['divergence_momentum']
    
    # Regime classification
    data['vol_regime'] = np.where(data['vol_regime_shift'] > 0, 1, 0)
    data['div_regime'] = np.where(data['consecutive_divergence'] > 2, 1, 0)
    
    # Composite Alpha Generation
    # Asymmetric momentum
    data['asymmetric_momentum'] = data['upside_vol_dominance'] * data['gap_absorption'] * data['divergence_momentum']
    
    # Volatility-divergence alignment
    data['vol_div_alignment'] = data['vol_regime_shift'] * data['consecutive_divergence'] * data['gap_reversal']
    
    # Regime-specific factors
    data['regime_specific'] = np.where(
        (data['vol_regime'] == 1) & (data['div_regime'] == 1),
        data['vol_expansion'],
        np.where(
            (data['vol_regime'] == 0) & (data['div_regime'] == 1),
            data['mean_reversion'],
            data['asymmetric_momentum']
        )
    )
    
    # Final composite alpha
    alpha = (data['asymmetric_momentum'] + data['vol_div_alignment'] + data['regime_specific']) / 3
    
    return alpha
