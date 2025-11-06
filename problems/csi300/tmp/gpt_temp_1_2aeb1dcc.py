import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Market Regime Classification
    data['volatility_20d'] = data['true_range'].rolling(window=20).std()
    data['volatility_percentile'] = data['volatility_20d'].rolling(window=60).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )
    
    # Regime classification
    data['regime'] = np.select(
        [
            data['volatility_percentile'] < 0.33,
            data['volatility_percentile'] > 0.67
        ],
        ['low_vol', 'high_vol'],
        default='normal_vol'
    )
    
    # Intraday Momentum Divergence
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_divergence'] = data['return_3d'] - data['return_5d']
    
    # Volume Efficiency Analysis
    data['volume_20d_median'] = data['volume'].rolling(window=20).median()
    data['volume_surge'] = data['volume'] / data['volume_20d_median']
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volume_efficiency'] = data['daily_range'] / (data['volume_surge'] + 1e-8)
    
    # Regime-specific divergence patterns
    data['divergence_low_vol'] = data['momentum_divergence'] * np.where(data['regime'] == 'low_vol', 1.2, 1.0)
    data['divergence_high_vol'] = data['momentum_divergence'] * np.where(data['regime'] == 'high_vol', 0.8, 1.0)
    data['regime_adjusted_divergence'] = np.select(
        [
            data['regime'] == 'low_vol',
            data['regime'] == 'high_vol'
        ],
        [
            data['divergence_low_vol'],
            data['divergence_high_vol']
        ],
        default=data['momentum_divergence']
    )
    
    # Composite Signal Generation
    data['volume_confirmation'] = np.where(
        data['volume_surge'] > 1.2,
        np.sign(data['regime_adjusted_divergence']) * data['volume_efficiency'],
        0
    )
    
    data['composite_signal'] = (
        data['regime_adjusted_divergence'] * 0.6 + 
        data['volume_confirmation'] * 0.4
    )
    
    # Final factor value with regime-specific smoothing
    low_vol_filter = data['regime'] == 'low_vol'
    high_vol_filter = data['regime'] == 'high_vol'
    normal_vol_filter = data['regime'] == 'normal_vol'
    
    data['factor'] = np.nan
    data.loc[low_vol_filter, 'factor'] = data.loc[low_vol_filter, 'composite_signal'].rolling(window=3).mean()
    data.loc[high_vol_filter, 'factor'] = data.loc[high_vol_filter, 'composite_signal'].rolling(window=5).mean()
    data.loc[normal_vol_filter, 'factor'] = data.loc[normal_vol_filter, 'composite_signal']
    
    return data['factor']
