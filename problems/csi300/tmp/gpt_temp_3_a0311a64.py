import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple market microstructure insights
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Asymmetric Gap Reaction Factor
    data['overnight_ret'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_ret'] = (data['close'] - data['open']) / data['open']
    
    # Gap fill vs continuation behavior
    data['gap_fill_ratio'] = np.where(
        data['overnight_ret'] > 0,
        -data['intraday_ret'] / (data['overnight_ret'] + 1e-8),
        data['intraday_ret'] / (data['overnight_ret'] - 1e-8)
    )
    
    # 3-day persistence of gap reaction patterns
    data['gap_reaction_persistence'] = data['gap_fill_ratio'].rolling(window=3, min_periods=1).mean()
    
    # 2. Volume-Weighted Range Efficiency
    data['actual_move'] = abs(data['close'] - data['open'])
    data['daily_range'] = data['high'] - data['low']
    data['range_efficiency'] = data['actual_move'] / (data['daily_range'] + 1e-8)
    
    # Volume intensity scoring
    data['volume_rank'] = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    data['volume_weighted_efficiency'] = data['range_efficiency'] * data['volume_rank']
    
    # 3. Multi-Scale Momentum Divergence
    data['mom_1d'] = data['close'] / data['close'].shift(1) - 1
    data['mom_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_acceleration'] = data['mom_3d'] - data['mom_1d']
    
    # Momentum divergence detection
    data['momentum_divergence'] = data['momentum_acceleration'].rolling(window=5, min_periods=1).std()
    
    # 4. Liquidity Absorption Factor
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_change'] = data['avg_trade_size'].pct_change(periods=5)
    
    # Absorption patterns during price moves
    data['price_move'] = data['close'].pct_change()
    data['absorption_score'] = -data['trade_size_change'] * data['price_move']
    
    # 5. Volatility-Compressed Breakout Detection
    # Calculate True Range
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5d'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    
    data['volatility_ratio'] = data['daily_range'] / (data['atr_5d'] + 1e-8)
    
    # Breakout candidate score
    data['volume_expansion'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['breakout_score'] = np.where(
        data['volatility_ratio'] < 0.7,
        data['volume_expansion'] * data['momentum_acceleration'],
        0
    )
    
    # 6. Price-Volume Trend Consistency
    data['price_trend'] = np.sign(data['close'].pct_change(periods=3))
    data['volume_trend'] = np.sign(data['volume'].pct_change(periods=3))
    
    # Trend alignment
    data['trend_alignment'] = data['price_trend'] * data['volume_trend']
    
    # Consecutive aligned days
    aligned_mask = data['trend_alignment'] > 0
    data['consecutive_aligned'] = aligned_mask.astype(int) * (aligned_mask.groupby((~aligned_mask).cumsum()).cumcount() + 1)
    
    # 7. Close-to-Close vs Intraday Momentum
    data['close_to_close_ret'] = data['close'].pct_change()
    data['return_component_ratio'] = abs(data['intraday_ret']) / (abs(data['close_to_close_ret']) + 1e-8)
    
    # 8. Momentum Exhaustion Detection
    data['mom_3d_roc'] = data['mom_3d'].diff()
    data['volume_divergence'] = np.where(
        data['mom_3d'] > data['mom_3d'].rolling(window=5, min_periods=1).mean(),
        data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean(),
        1
    )
    data['exhaustion_score'] = -data['mom_3d_roc'] * (2 - data['volume_divergence'])
    
    # 9. Range Expansion Continuation
    data['avg_range_10d'] = data['daily_range'].rolling(window=10, min_periods=1).mean()
    data['range_expansion_ratio'] = data['daily_range'] / (data['avg_range_10d'] + 1e-8)
    
    # Follow-through behavior
    data['follow_through'] = data['close_to_close_ret'].shift(-1).rolling(window=3, min_periods=1).mean()
    data['range_expansion_score'] = data['range_expansion_ratio'] * data['follow_through'] * data['volume_expansion']
    
    # Combine all factors with appropriate weights
    factors = [
        data['gap_reaction_persistence'],
        data['volume_weighted_efficiency'],
        data['momentum_divergence'],
        data['absorption_score'],
        data['breakout_score'],
        data['consecutive_aligned'],
        data['return_component_ratio'],
        data['exhaustion_score'],
        data['range_expansion_score']
    ]
    
    # Normalize and combine factors
    combined_factor = pd.Series(0, index=data.index)
    for factor in factors:
        normalized = (factor - factor.rolling(window=20, min_periods=1).mean()) / (factor.rolling(window=20, min_periods=1).std() + 1e-8)
        combined_factor += normalized.fillna(0)
    
    return combined_factor
