import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Range Efficiency with Volume-Price Divergence alpha factor
    """
    data = df.copy()
    
    # Calculate Multi-Timeframe Price Efficiency
    # True Range Efficiency
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(data['high'], data['prev_close']) - np.minimum(data['low'], data['prev_close'])
    data['price_efficiency'] = (data['close'] - data['prev_close']) / data['true_range'].replace(0, np.nan)
    
    # Efficiency persistence
    data['efficiency_persistence_5d'] = data['price_efficiency'].rolling(window=5, min_periods=3).mean()
    
    # Range Utilization Patterns
    data['intraday_range'] = data['high'] - data['low']
    data['range_utilization'] = (data['close'] - data['low']) / data['intraday_range'].replace(0, np.nan)
    data['range_util_3d'] = data['range_utilization'].rolling(window=3, min_periods=2).mean()
    data['range_util_8d'] = data['range_utilization'].rolling(window=8, min_periods=5).mean()
    data['range_compression'] = (data['intraday_range'].rolling(window=5, min_periods=3).std() / 
                               data['intraday_range'].rolling(window=20, min_periods=10).std())
    
    # Volume-Price Divergence Dynamics
    # Volume Anomalies
    data['volume_ma_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_std_20'] = data['volume'].rolling(window=20, min_periods=10).std()
    data['volume_zscore'] = (data['volume'] - data['volume_ma_20']) / data['volume_std_20'].replace(0, np.nan)
    
    # Volume acceleration
    data['volume_ma_3'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_acceleration'] = (data['volume_ma_3'] - data['volume_ma_3'].shift(3)) / data['volume_ma_3'].shift(3).replace(0, np.nan)
    
    # Price-Volume Relationship
    data['price_momentum_5d'] = data['close'].pct_change(5)
    data['volume_weighted_momentum'] = data['price_momentum_5d'] * data['volume_zscore']
    
    # Volume-price divergence
    data['price_volume_divergence'] = (data['price_momentum_5d'] - data['volume_zscore']) / (
        np.abs(data['price_momentum_5d']) + np.abs(data['volume_zscore'])).replace(0, np.nan)
    
    # Market Microstructure Pressure
    # Order Imbalance Proxy
    data['buying_pressure'] = (data['close'] - data['open']) / data['intraday_range'].replace(0, np.nan)
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_persistence'] = data['opening_gap'] * data['buying_pressure']
    
    # Execution Quality
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['price_improvement'] = (data['close'] - data['midpoint']) / data['midpoint'].replace(0, np.nan)
    data['slippage_indicator'] = data['price_improvement'] * data['volume_zscore']
    
    # Combine Efficiency with Divergence Signals
    # Efficiency-Divergence Matrix components
    data['efficiency_signal'] = np.where(
        data['efficiency_persistence_5d'] > data['efficiency_persistence_5d'].rolling(window=20, min_periods=10).mean(),
        data['efficiency_persistence_5d'], 
        -data['efficiency_persistence_5d']
    )
    
    data['divergence_signal'] = np.where(
        data['price_volume_divergence'] > 0,
        -data['price_volume_divergence'],  # Negative for positive divergence (price up, volume down)
        data['price_volume_divergence']    # Positive for negative divergence (price down, volume up)
    )
    
    # Dynamic weighting
    efficiency_weight = np.abs(data['efficiency_persistence_5d'])
    volume_weight = np.abs(data['volume_zscore'])
    pressure_weight = np.abs(data['buying_pressure'])
    
    # Combined signal
    data['combined_signal'] = (
        efficiency_weight * data['efficiency_signal'] +
        volume_weight * data['divergence_signal'] +
        pressure_weight * data['slippage_indicator']
    ) / (efficiency_weight + volume_weight + pressure_weight).replace(0, np.nan)
    
    # Multi-Timeframe Consistency
    # Cross-horizon alignment
    data['efficiency_trend_3d'] = data['price_efficiency'].rolling(window=3, min_periods=2).mean()
    data['efficiency_trend_8d'] = data['price_efficiency'].rolling(window=8, min_periods=5).mean()
    data['efficiency_alignment'] = np.sign(data['efficiency_trend_3d']) * np.sign(data['efficiency_trend_8d'])
    
    data['volume_div_3d'] = data['price_volume_divergence'].rolling(window=3, min_periods=2).mean()
    data['volume_div_8d'] = data['price_volume_divergence'].rolling(window=8, min_periods=5).mean()
    data['divergence_alignment'] = np.sign(data['volume_div_3d']) * np.sign(data['volume_div_8d'])
    
    # Confidence scoring
    alignment_score = (data['efficiency_alignment'] + data['divergence_alignment']) / 2
    consistency_penalty = np.where(alignment_score < 0, -0.5, 0)
    
    # Final Alpha Factor
    data['alpha_factor'] = (
        data['combined_signal'] * 
        (1 + alignment_score) * 
        (1 + consistency_penalty) *
        data['range_compression']
    )
    
    # Pattern recognition enhancement
    data['efficiency_rank'] = data['efficiency_persistence_5d'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    data['divergence_rank'] = data['price_volume_divergence'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Final composite with pattern weights
    pattern_weight = np.where(
        (data['efficiency_rank'] > 0.7) & (data['divergence_rank'] < 0.3),
        1.5,  # Strong trend with volume confirmation
        np.where(
            (data['efficiency_rank'] < 0.3) & (data['divergence_rank'] > 0.7),
            -1.5,  # Inefficient move with volume rejection
            np.where(
                (data['efficiency_rank'] > 0.6) & (data['divergence_rank'] > 0.6),
                0.5,  # Mixed signals
                1.0   # Neutral
            )
        )
    )
    
    final_alpha = data['alpha_factor'] * pattern_weight
    
    return final_alpha
