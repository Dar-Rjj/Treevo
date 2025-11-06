import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Factor with Multi-Timeframe Anchoring
    Combines psychological price anchoring, volume divergence patterns, 
    intraday microstructure, and market memory effects
    """
    
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price Anchoring Behavior Analysis
    # Recent High-Low Anchoring Points
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    data['mid_5d'] = (data['high_5d'] + data['low_5d']) / 2
    
    # Distance from current close to anchor levels
    data['dist_to_high'] = (data['close'] - data['high_5d']) / data['high_5d']
    data['dist_to_low'] = (data['close'] - data['low_5d']) / data['low_5d']
    data['dist_to_mid'] = (data['close'] - data['mid_5d']) / data['mid_5d']
    
    # Round Number Price Anchoring
    data['round_level'] = np.round(data['close'] / 10) * 10
    data['dist_to_round'] = (data['close'] - data['round_level']) / data['round_level']
    data['round_proximity'] = 1 - abs(data['dist_to_round'])
    
    # Anchoring Strength - Price Rejection
    data['high_rejection'] = ((data['high'] - data['high_5d']) / data['high_5d']).rolling(window=3).std()
    data['low_rejection'] = ((data['low'] - data['low_5d']) / data['low_5d']).rolling(window=3).std()
    
    # 2. Volume Divergence Patterns
    # Multi-Timeframe Volume Consistency
    data['volume_3d_ma'] = data['volume'].rolling(window=3, min_periods=2).mean()
    data['volume_10d_ma'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_momentum'] = (data['volume_3d_ma'] - data['volume_10d_ma']) / data['volume_10d_ma']
    
    # Volume acceleration
    data['volume_accel'] = data['volume_momentum'].diff()
    
    # Price-Volume Divergence Signals
    data['price_change_3d'] = data['close'].pct_change(periods=3)
    data['volume_change_3d'] = data['volume'].pct_change(periods=3)
    
    # Negative divergence: price up but volume down
    data['neg_divergence'] = np.where(
        (data['price_change_3d'] > 0) & (data['volume_change_3d'] < 0),
        data['price_change_3d'] * abs(data['volume_change_3d']),
        0
    )
    
    # Positive divergence: price down but volume down (capitulation)
    data['pos_divergence'] = np.where(
        (data['price_change_3d'] < 0) & (data['volume_change_3d'] < 0),
        abs(data['price_change_3d']) * abs(data['volume_change_3d']),
        0
    )
    
    # 3. Intraday Microstructure Analysis
    # Opening Gap Behavior
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_abs'] = abs(data['opening_gap'])
    
    # Gap filling speed (first hour approximation using high/low)
    data['gap_fill_speed'] = np.where(
        data['opening_gap'] > 0,
        (data['high'] - data['open']) / (data['high'] - data['prev_close']),
        (data['open'] - data['low']) / (data['prev_close'] - data['low'])
    )
    
    # End-of-Day Price Positioning
    data['last_hour_range'] = (data['high'] - data['low']) / data['close']
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # 4. Market Memory and Persistence Effects
    # Price Return Memory
    data['return_1d'] = data['close'].pct_change()
    data['return_3d'] = data['close'].pct_change(periods=3)
    
    # Auto-correlation of returns
    data['return_autocorr_1'] = data['return_1d'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    
    # Volatility clustering
    data['volatility_5d'] = data['return_1d'].rolling(window=5, min_periods=3).std()
    data['volatility_persistence'] = data['volatility_5d'] / data['volatility_5d'].shift(1)
    
    # Volume Memory
    data['volume_autocorr'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    
    # 5. Divergence Factor Synthesis
    # Core Divergence Signal Construction
    data['divergence_score'] = (
        data['pos_divergence'] - data['neg_divergence']
    ) * (1 + data['volume_momentum'])
    
    # Anchoring Breakout Confirmation
    data['anchor_breakout'] = np.where(
        abs(data['dist_to_mid']) > 0.02,
        data['dist_to_mid'] * (1 + data['volume_momentum']),
        0
    )
    
    # Multi-Timeframe Signal Integration
    # Short-term signals (3-day)
    data['short_term_signal'] = (
        data['divergence_score'].rolling(window=3).mean() + 
        data['anchor_breakout'].rolling(window=3).mean()
    )
    
    # Medium-term signals (10-day)
    data['medium_term_signal'] = (
        data['divergence_score'].rolling(window=10).mean() + 
        data['anchor_breakout'].rolling(window=10).mean()
    )
    
    # Signal alignment
    data['signal_alignment'] = np.sign(data['short_term_signal']) * np.sign(data['medium_term_signal'])
    
    # Microstructure Timing Enhancement
    data['micro_timing'] = (
        data['gap_fill_speed'] * data['close_position'] * 
        (1 + data['volume_momentum'])
    )
    
    # Final Composite Alpha Output
    alpha = (
        # Multi-timeframe price-volume divergence
        0.4 * data['divergence_score'] +
        # Anchoring breakout momentum with volume confirmation
        0.3 * data['anchor_breakout'] * (1 + data['volume_momentum']) +
        # Market memory-adjusted signals
        0.2 * data['signal_alignment'] * (data['return_autocorr_1'] + 1) +
        # Microstructure enhancement
        0.1 * data['micro_timing']
    )
    
    # Clean up intermediate columns
    result = alpha.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return result
