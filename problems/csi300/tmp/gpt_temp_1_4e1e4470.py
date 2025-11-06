import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Multi-Timeframe Gap Analysis
    df['gap_magnitude'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_fill_efficiency'] = (df['close'] - df['open']) / (df['close'].shift(1) - df['open'])
    df['gap_fill_efficiency'] = df['gap_fill_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Breakout Pattern Detection
    df['up_breakout'] = ((df['high'] > df['high'].shift(1)) & (df['gap_magnitude'] > 0)).astype(int)
    df['down_breakout'] = ((df['low'] < df['low'].shift(1)) & (df['gap_magnitude'] < 0)).astype(int)
    
    # Breakout Persistence
    df['breakout_count'] = (df['up_breakout'] + df['down_breakout']).rolling(window=5, min_periods=1).sum()
    
    # Multi-Timeframe Momentum
    df['short_term_gap_momentum'] = df['gap_magnitude'].rolling(window=5, min_periods=1).sum()
    df['breakout_success_rate'] = (df['up_breakout'] + df['down_breakout']).rolling(window=10, min_periods=1).mean()
    
    # Gap-Breakout Alignment Strength
    df['direction_consistency'] = np.sign(df['gap_magnitude']) * (df['up_breakout'] - df['down_breakout'])
    df['momentum_enhancement'] = df['gap_magnitude'].abs() * df['breakout_count']
    df['multi_timeframe_divergence'] = (df['short_term_gap_momentum'] - df['breakout_success_rate']).abs()
    
    # Trade Size Distribution Analysis
    df['avg_trade_size'] = df['amount'] / df['volume']
    df['large_small_ratio'] = df['avg_trade_size'].rolling(window=5, min_periods=1).mean() / df['avg_trade_size']
    df['trade_size_skew'] = df['avg_trade_size'].rolling(window=10, min_periods=1).apply(
        lambda x: pd.Series(x).skew() if len(x) > 1 else 0
    )
    
    # Volume-Range Efficiency
    df['daily_range_efficiency'] = (df['close'] - df['open']).abs() / (df['high'] - df['low'])
    df['volume_clustering'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    df['range_volume_alignment'] = ((df['high'] - df['low']) / df['high'].shift(1)) * df['volume_clustering']
    
    # Microstructure Divergence Detection
    df['trade_size_imbalance'] = df['large_small_ratio'] * np.sign(df['gap_magnitude'])
    df['volume_pattern'] = df['volume_clustering'] * df['breakout_count']
    df['volume_momentum_5d'] = df['volume'].pct_change().rolling(window=5, min_periods=1).mean()
    df['volume_momentum_10d'] = df['volume'].pct_change().rolling(window=10, min_periods=1).mean()
    df['multi_timeframe_volume_divergence'] = (df['volume_momentum_5d'] - df['volume_momentum_10d']).abs()
    
    # Gap-Breakout Component
    df['gap_breakout_momentum'] = df['direction_consistency'] * df['momentum_enhancement']
    df['multi_timeframe_adjustment'] = 1 - df['multi_timeframe_divergence']
    df['enhanced_gap_breakout'] = df['gap_breakout_momentum'] * df['multi_timeframe_adjustment']
    
    # Microstructure Component
    df['trade_volume_alignment'] = df['trade_size_imbalance'] * df['volume_pattern']
    df['range_efficiency_factor'] = 1 - df['daily_range_efficiency']
    df['microstructure_divergence'] = df['trade_volume_alignment'] * df['range_efficiency_factor']
    
    # Final Alpha Integration
    df['volume_confirmation'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    df['composite_factor'] = df['enhanced_gap_breakout'] * df['microstructure_divergence']
    df['alpha_signal'] = df['composite_factor'] * df['volume_confirmation']
    
    # Clean up and return
    alpha_signal = df['alpha_signal'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_signal
