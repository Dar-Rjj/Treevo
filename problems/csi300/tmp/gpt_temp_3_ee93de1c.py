import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Volatility-Regime Momentum Persistence
    # Multi-timeframe momentum calculation
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_divergence'] = data['momentum_3d'] - data['momentum_8d']
    
    # True Range calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Volatility-adjusted persistence
    data['vol_5d'] = data['tr'].rolling(window=5).mean() / data['close'].shift(1)
    data['vol_scaled_momentum'] = data['momentum_divergence'] / (data['vol_5d'] + 0.001)
    
    # Momentum persistence
    for i in range(len(data)):
        if i >= 2:
            current_sign = np.sign(data['momentum_divergence'].iloc[i])
            persistence_count = 0
            for j in range(3):
                if i - j >= 0 and np.sign(data['momentum_divergence'].iloc[i - j]) == current_sign:
                    persistence_count += 1
            data.loc[data.index[i], 'momentum_persistence'] = persistence_count
    
    # Range Efficiency Volume Confirmation
    # Multi-period range efficiency
    data['range_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['efficiency_3d_avg'] = data['range_efficiency'].rolling(window=3).mean()
    data['efficiency_momentum'] = data['range_efficiency'] - data['efficiency_3d_avg']
    
    # Volume confirmation patterns
    data['high_efficiency_volume'] = np.where(data['range_efficiency'] > 0.7, data['volume'], 0)
    data['volume_efficiency_corr'] = np.sign(data['efficiency_momentum']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Efficiency persistence
    for i in range(len(data)):
        if i >= 2:
            efficiency_count = 0
            for j in range(3):
                if i - j >= 0 and data['range_efficiency'].iloc[i - j] > 0.6:
                    efficiency_count += 1
            data.loc[data.index[i], 'efficiency_persistence'] = efficiency_count
    
    # Extreme Return Volume Clustering Regime
    # Volatility-adjusted extreme detection
    data['vol_10d'] = data['close'].rolling(window=10).std() / data['close'].rolling(window=10).mean()
    data['extreme_threshold'] = 1.5 * data['vol_10d']
    data['extreme_return'] = abs(data['close'] / data['close'].shift(1) - 1) > data['extreme_threshold']
    
    # Volume clustering analysis
    data['volume_rank'] = data['volume'].rolling(window=10).apply(lambda x: (x.iloc[-1] > x).mean(), raw=False)
    
    # Volume clustering persistence
    for i in range(len(data)):
        if i >= 2:
            clustering_count = 0
            for j in range(3):
                if (i - j >= 1 and data['extreme_return'].iloc[i - j] and 
                    data['volume'].iloc[i - j] > data['volume'].iloc[i - j - 1]):
                    clustering_count += 1
            data.loc[data.index[i], 'volume_clustering_persistence'] = clustering_count
    
    data['clustering_momentum'] = data['volume_clustering_persistence'] * (data['close'] / data['close'].shift(1) - 1)
    
    # Amount Flow Direction Persistence
    # Multi-timeframe directional flow
    data['directional_flow'] = np.sign(data['close'] - data['close'].shift(1)) * data['amount']
    data['net_flow_3d'] = data['directional_flow'].rolling(window=3).sum()
    data['net_flow_prev_3d'] = data['directional_flow'].shift(3).rolling(window=3).sum()
    data['flow_momentum'] = data['net_flow_3d'] - data['net_flow_prev_3d']
    
    # Flow persistence strength
    for i in range(len(data)):
        if i >= 2:
            current_sign = np.sign(data['directional_flow'].iloc[i])
            flow_count = 0
            for j in range(3):
                if i - j >= 0 and np.sign(data['directional_flow'].iloc[i - j]) == current_sign:
                    flow_count += 1
            data.loc[data.index[i], 'consecutive_directional_days'] = flow_count
    
    data['range_3d'] = (data['high'] - data['low']).rolling(window=3).sum()
    data['flow_range_ratio'] = data['net_flow_3d'] / (data['range_3d'] + 1e-8)
    data['flow_acceleration'] = data['flow_momentum'] / (abs(data['net_flow_3d']) + 0.001)
    
    # Opening Gap Volatility Regime
    # Gap magnitude and persistence
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    
    # Gap persistence
    for i in range(len(data)):
        if i >= 2:
            current_sign = np.sign(data['overnight_gap'].iloc[i])
            gap_count = 0
            for j in range(3):
                if i - j >= 0 and np.sign(data['overnight_gap'].iloc[i - j]) == current_sign:
                    gap_count += 1
            data.loc[data.index[i], 'gap_persistence'] = gap_count
    
    data['vol_adj_gap'] = abs(data['overnight_gap']) / (data['tr'].shift(1).rolling(window=4).mean() / data['close'].shift(2) + 1e-8)
    
    # Intraday gap dynamics
    data['gap_fill_efficiency'] = (data['close'] - data['open']) / (data['overnight_gap'] + 1e-8)
    data['gap_volume_intensity'] = data['volume'] / data['volume'].shift(1).rolling(window=4).mean()
    data['gap_reversal_signal'] = data['gap_fill_efficiency'] * data['gap_persistence']
    
    # Price-Volume Correlation Regime Persistence
    # Dynamic correlation regimes
    data['close_ret'] = data['close'].pct_change()
    data['volume_ret'] = data['volume'].pct_change()
    data['corr_5d'] = data['close_ret'].rolling(window=5).corr(data['volume_ret'])
    data['correlation_regime'] = np.sign(data['corr_5d'])
    
    # Regime persistence
    for i in range(len(data)):
        if i >= 4:
            current_regime = data['correlation_regime'].iloc[i]
            regime_count = 0
            for j in range(5):
                if i - j >= 0 and data['correlation_regime'].iloc[i - j] == current_regime:
                    regime_count += 1
            data.loc[data.index[i], 'regime_persistence'] = regime_count
    
    # Regime-specific alpha signals
    data['high_corr_momentum'] = data['momentum_3d'] * abs(data['corr_5d'])
    data['low_corr_reversal'] = (data['close'] / data['close'].shift(1) - 1) * (1 - abs(data['corr_5d']))
    data['regime_transition'] = data['regime_persistence'] * data['correlation_regime']
    
    # Combine all factors with appropriate weights
    result = (
        0.2 * data['vol_scaled_momentum'] * data['momentum_persistence'] +
        0.15 * data['volume_efficiency_corr'] * data['efficiency_persistence'] +
        0.15 * data['clustering_momentum'] +
        0.15 * data['flow_acceleration'] * data['consecutive_directional_days'] +
        0.15 * data['gap_reversal_signal'] +
        0.1 * data['high_corr_momentum'] +
        0.05 * data['low_corr_reversal'] +
        0.05 * data['regime_transition']
    )
    
    return result.fillna(0)
