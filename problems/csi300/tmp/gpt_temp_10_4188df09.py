import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility-Efficiency Core
    # True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['vol_ratio'] = data['true_range'] / data['close']
    
    # Path Efficiency
    data['close_5d_ago'] = data['close'].shift(5)
    data['abs_price_change'] = abs(data['close'] - data['close'].shift(1))
    data['path_length'] = data['abs_price_change'].rolling(window=5).sum()
    data['path_efficiency'] = abs(data['close'] - data['close_5d_ago']) / data['path_length']
    data['path_efficiency'] = data['path_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Vol-Efficiency Momentum
    data['price_momentum'] = (data['close'] - data['close_5d_ago']) / data['close_5d_ago']
    data['vol_efficiency_momentum'] = data['price_momentum'] / data['vol_ratio'] * data['path_efficiency']
    
    # Regime Detection
    # Price Entropy
    data['returns'] = data['close'].pct_change()
    data['returns_var'] = data['returns'].rolling(window=20).var()
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['price_entropy'] = data['returns_var'] / data['daily_range']
    data['price_entropy'] = data['price_entropy'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Entropy
    data['volume_median_20d'] = data['volume'].rolling(window=20).median()
    data['volume_threshold'] = 1.5 * data['volume_median_20d']
    
    # Count high volume days in last 5 days
    high_volume_count = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            count = (window_data['volume'] > window_data['volume_threshold']).sum()
            high_volume_count.iloc[i] = count / 5
        else:
            high_volume_count.iloc[i] = np.nan
    data['volume_entropy'] = high_volume_count
    
    # Regime Classification
    price_entropy_median = data['price_entropy'].rolling(window=20).median()
    data['high_vol_regime'] = (data['price_entropy'] > price_entropy_median) & (data['volume_entropy'] > 0.3)
    
    data['volume_5d_sum'] = data['volume'].rolling(window=5).sum()
    data['volume_ratio'] = data['volume'] / data['volume_5d_sum']
    data['low_vol_regime'] = (data['daily_range'] < 0.02) & (data['volume_ratio'] < 0.25)
    
    data['transition_regime'] = ~data['high_vol_regime'] & ~data['low_vol_regime']
    
    # Convergence Framework
    # Price Convergence
    data['ma20'] = data['close'].rolling(window=20).mean()
    data['price_convergence_1'] = 1 - abs(data['close'] - data['ma20']) / data['close']
    
    data['high_low_range'] = data['high'] - data['low']
    data['ma5_range'] = data['high_low_range'].rolling(window=5).mean()
    data['range_diff'] = data['ma5_range'] - data['high_low_range']
    data['price_convergence_2'] = 1 + data['range_diff'] / data['ma5_range']
    data['price_convergence'] = data['price_convergence_1'] * data['price_convergence_2']
    
    # Trade Alignment
    data['trade_size'] = data['amount'] / data['volume']
    data['ma5_trade_size'] = data['trade_size'].rolling(window=5).mean()
    data['price_signal'] = np.sign((data['close'] - data['close_5d_ago']) / data['close_5d_ago'])
    data['trade_size_signal'] = np.sign((data['trade_size'] - data['ma5_trade_size']) / data['ma5_trade_size'])
    data['trade_alignment'] = data['price_signal'] * data['trade_size_signal']
    
    # Alignment Persistence
    alignment_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            window_alignments = data['trade_alignment'].iloc[i-4:i+1]
            if len(window_alignments) == 5:
                count = (window_alignments == window_alignments.iloc[-1]).sum()
                alignment_persistence.iloc[i] = count
            else:
                alignment_persistence.iloc[i] = np.nan
        else:
            alignment_persistence.iloc[i] = np.nan
    data['alignment_persistence'] = alignment_persistence
    
    # Alpha Construction
    # Base Factor
    data['base_factor'] = data['vol_efficiency_momentum'] * data['price_convergence']
    
    # Regime Multiplier
    data['regime_multiplier'] = 1.0
    data.loc[data['low_vol_regime'], 'regime_multiplier'] = 1.4
    data.loc[data['high_vol_regime'], 'regime_multiplier'] = 0.7
    
    # Final Alpha
    data['final_alpha'] = data['base_factor'] * data['regime_multiplier'] * (1 + data['alignment_persistence'] / 5)
    
    return data['final_alpha']
