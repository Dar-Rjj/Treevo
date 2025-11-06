import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['daily_range_pct'] = (data['high'] - data['low']) / data['low']
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Intraday Momentum Persistence
    data['morning_momentum'] = (data['high'] - data['open']) / data['open']
    data['afternoon_momentum'] = (data['close'] - data['low']) / data['low']
    data['session_ratio'] = data['morning_momentum'] / (data['afternoon_momentum'] + 1e-8)
    data['session_autocorr'] = data['session_ratio'].rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Range Expansion Breakout
    data['range_7day'] = data['daily_range_pct'].rolling(window=7, min_periods=5).mean()
    data['range_expansion'] = data['daily_range_pct'] / (data['range_7day'] + 1e-8)
    data['volume_range_product'] = data['volume'] * data['daily_range_pct']
    data['vrp_ma'] = data['volume_range_product'].rolling(window=5, min_periods=3).mean()
    data['vrp_confirmation'] = data['volume_range_product'] / (data['vrp_ma'] + 1e-8)
    
    # Volatility-Adjusted Volume Momentum
    data['atr_5day'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    data['vol_adj_volume'] = data['volume'] / (data['atr_5day'] + 1e-8)
    data['momentum_3day'] = data['close'].pct_change(periods=3)
    data['momentum_8day'] = data['close'].pct_change(periods=8)
    data['momentum_divergence'] = data['momentum_3day'] - data['momentum_8day']
    data['regime_persistence'] = data['momentum_divergence'].rolling(window=5, min_periods=3).apply(
        lambda x: len(np.where(np.diff(np.sign(x)) != 0)[0]) / max(len(x)-1, 1), raw=False
    )
    
    # Opening Gap Efficiency
    data['gap_pct'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_filling'] = np.where(
        data['gap_pct'] > 0,
        (data['low'] - data['prev_close']) / (data['open'] - data['prev_close'] + 1e-8),
        (data['high'] - data['prev_close']) / (data['open'] - data['prev_close'] - 1e-8)
    )
    data['gap_efficiency'] = 1 - abs(data['gap_filling'])
    data['gap_efficiency_5day'] = data['gap_efficiency'].rolling(window=5, min_periods=3).mean()
    
    # Amount-Volume Divergence
    data['per_share_value'] = data['amount'] / (data['volume'] + 1e-8)
    data['price_ratio'] = data['per_share_value'] / data['close']
    data['amount_volume_corr'] = data['amount'].rolling(window=6, min_periods=4).corr(data['volume'])
    data['divergence_regime'] = data['amount_volume_corr'].diff().rolling(window=3, min_periods=2).std()
    
    # Price-Volume Harmony
    data['momentum_2day'] = data['close'].pct_change(periods=2)
    data['momentum_6day'] = data['close'].pct_change(periods=6)
    data['volume_momentum_2day'] = data['volume'].pct_change(periods=2)
    data['volume_momentum_6day'] = data['volume'].pct_change(periods=6)
    
    data['price_harmony'] = data['momentum_2day'] / (data['momentum_6day'] + 1e-8)
    data['volume_harmony'] = data['volume_momentum_2day'] / (data['volume_momentum_6day'] + 1e-8)
    data['harmony_divergence'] = data['price_harmony'] - data['volume_harmony']
    data['scale_alignment'] = data['harmony_divergence'].rolling(window=5, min_periods=3).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.shift(1))) if len(x) > 1 else 0, raw=False
    )
    
    # Combine factors with weights
    factor = (
        0.15 * data['session_autocorr'] +
        0.18 * data['range_expansion'] +
        0.12 * data['vrp_confirmation'] +
        0.16 * data['vol_adj_volume'] * (1 - data['regime_persistence']) +
        0.14 * data['gap_efficiency_5day'] +
        0.13 * data['divergence_regime'] +
        0.12 * data['scale_alignment']
    )
    
    return factor
