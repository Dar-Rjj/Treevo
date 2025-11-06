import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Fractal Volatility Analysis
    # Fractal Range Calculation
    data['ultra_short_fractal'] = (
        (data['high'].rolling(window=3).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False)) /
        (data['close'].diff().abs().rolling(window=2).sum())
    )
    
    data['short_term_fractal'] = (
        (data['high'].rolling(window=5).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False)) /
        (data['close'].diff().abs().rolling(window=5).sum())
    )
    
    data['medium_term_fractal'] = (
        (data['high'].rolling(window=10).apply(lambda x: np.sum(np.abs(x - data.loc[x.index, 'low'])), raw=False)) /
        (data['close'].diff().abs().rolling(window=10).sum())
    )
    
    # Volatility Asymmetry Analysis
    data['upside_vol_eff'] = (data['high'] - data['open']) / (data['high'] - data['low'])
    data['downside_vol_eff'] = (data['open'] - data['low']) / (data['high'] - data['low'])
    data['vol_skew_ratio'] = data['upside_vol_eff'] - data['downside_vol_eff']
    
    # Fractal-Volatility Integration
    data['fractal_weighted_skew'] = data['vol_skew_ratio'] * data['ultra_short_fractal']
    data['vol_fractal_momentum'] = data['vol_skew_ratio'] * (data['ultra_short_fractal'] - data['short_term_fractal'])
    data['multi_timeframe_vol_alignment'] = np.sign(data['vol_skew_ratio']) * np.sign(data['ultra_short_fractal'] - data['short_term_fractal'])
    
    # Microstructure Regime Classification
    # Opening Auction Dynamics
    data['opening_gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['auction_imbalance'] = (data['open'] - data['low']) - (data['high'] - data['open'])
    data['opening_efficiency'] = np.abs(data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    
    # Intraday Microstructure Anchors
    data['prev_close_anchor'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['session_anchors'] = np.minimum(
        np.abs(data['high'] - data['high'].shift(1)),
        np.abs(data['low'] - data['low'].shift(1))
    ) / (data['high'] - data['low'])
    data['anchor_convergence'] = data['prev_close_anchor'] - data['session_anchors']
    
    # Volume-Based Regime Assessment
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['volume_concentration'] = data['volume'] / (
        data['volume'].shift(4) + data['volume'].shift(3) + data['volume'].shift(2) + data['volume'].shift(1)
    )
    data['volume_fractal_alignment'] = data['volume_efficiency'] * (data['ultra_short_fractal'] - data['short_term_fractal'])
    
    # Volatility Compression & Breakout Dynamics
    # Volatility Compression Patterns
    data['range_compression_intensity'] = (data['high'] - data['low']) / (data['high'].shift(4) - data['low'].shift(4))
    
    # Compression duration calculation
    compression_mask = data['range_compression_intensity'] < 0.8
    data['compression_duration'] = compression_mask.groupby(compression_mask.ne(compression_mask.shift()).cumsum()).cumcount() + 1
    data['compression_exhaustion'] = data['compression_duration'] * data['range_compression_intensity']
    
    # Fractal Breakout Detection
    data['upward_breakout'] = ((data['close'] > data['close'].shift(1)) & 
                              ((data['ultra_short_fractal'] - data['short_term_fractal']) > 0)).astype(int)
    data['downward_breakout'] = ((data['close'] < data['close'].shift(1)) & 
                                ((data['ultra_short_fractal'] - data['short_term_fractal']) < 0)).astype(int)
    data['breakout_strength'] = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Compression-Breakout Integration
    data['compression_breakout_signal'] = data['breakout_strength'] * (1 - data['range_compression_intensity'])
    data['fractal_compression_momentum'] = (data['ultra_short_fractal'] - data['short_term_fractal']) * data['compression_exhaustion']
    data['volume_breakout_confirmation'] = data['breakout_strength'] * data['volume_concentration']
    
    # Volume-Order Flow & Efficiency Integration
    # Volume Concentration Analysis
    data['recent_vs_historical_volume'] = data['volume'] / data['volume'].shift(5)
    
    # Volume during significant moves
    significant_move = np.abs(data['close'] - data['close'].shift(1)) > (0.02 * data['close'].shift(1))
    data['volume_during_moves'] = np.where(
        significant_move,
        data['volume'] / data['volume'].shift(1),
        np.nan
    )
    
    # Volume persistence
    data['volume_persistence'] = (
        np.sign(data['volume'] - data['volume'].shift(1)) * 
        np.sign(data['volume'].shift(1) - data['volume'].shift(2))
    )
    
    # Order Flow Imbalance
    data['directional_amount'] = data['amount'] * np.sign(data['close'] - data['close'].shift(1))
    
    # Cumulative imbalance over 5 days
    data['cumulative_imbalance'] = (
        data['directional_amount'].rolling(window=5).sum()
    )
    
    data['normalized_imbalance'] = (
        data['cumulative_imbalance'] / 
        data['amount'].abs().rolling(window=5).sum()
    )
    
    # Volume-Microstructure Integration
    data['volume_anchor_confirmation'] = data['volume_concentration'] * data['anchor_convergence']
    data['order_flow_efficiency'] = data['normalized_imbalance'] * data['opening_efficiency']
    data['volume_persistence_score'] = data['volume_persistence'] * np.sign(data['vol_skew_ratio'])
    data['fractal_flow_alignment'] = data['order_flow_efficiency'] * data['multi_timeframe_vol_alignment']
    
    # Regime-Adaptive Signal Processing
    # Determine volatility regime
    vol_regime = data['range_compression_intensity'].rolling(window=10).std()
    high_vol_regime = vol_regime > vol_regime.quantile(0.7)
    low_vol_regime = vol_regime < vol_regime.quantile(0.3)
    
    # High Volatility Regime signals
    data['high_regime_signal'] = (
        (data['vol_fractal_momentum'] / (1 + data['range_compression_intensity'])) +
        (data['vol_fractal_momentum'] * data['anchor_convergence']) +
        (data['vol_fractal_momentum'] * data['breakout_strength'])
    )
    
    # Low Volatility Regime signals
    data['low_regime_signal'] = (
        (data['fractal_compression_momentum'] * data['volume_fractal_alignment']) +
        (data['fractal_compression_momentum'] * data['opening_efficiency']) +
        (data['fractal_compression_momentum'] * data['volume_concentration'])
    )
    
    # Transition Regime signals
    data['balanced_signal'] = (data['high_regime_signal'] + data['low_regime_signal']) / 2
    data['directional_bias'] = np.sign(data['vol_fractal_momentum']) * np.sign(data['fractal_compression_momentum'])
    data['microstructure_alignment'] = data['multi_timeframe_vol_alignment'] * data['anchor_convergence']
    
    # Composite Factor Generation
    # Core Volatility-Fractal Signal
    data['base_signal'] = np.where(
        high_vol_regime, data['high_regime_signal'],
        np.where(low_vol_regime, data['low_regime_signal'], data['balanced_signal'])
    )
    
    data['compression_component'] = data['fractal_compression_momentum'] * data['volume_fractal_alignment']
    data['microstructure_score'] = data['multi_timeframe_vol_alignment'] * data['anchor_convergence']
    
    # Breakout validation with direction
    breakout_direction = np.where(data['upward_breakout'] == 1, 1, np.where(data['downward_breakout'] == 1, -1, 0))
    data['breakout_validation'] = data['base_signal'] * data['breakout_strength'] * breakout_direction
    
    # Volume-Microstructure Enhancement
    data['volume_strength'] = data['volume_anchor_confirmation'] * data['order_flow_efficiency']
    data['persistence_boost'] = data['volume_persistence_score'] * data['opening_efficiency']
    data['flow_volatility_alignment'] = data['order_flow_efficiency'] * data['multi_timeframe_vol_alignment']
    data['fractal_flow_momentum'] = data['fractal_flow_alignment'] * data['volume_persistence']
    
    # Contextual Adjustment
    data['compression_context'] = data['base_signal'] * (1 + data['compression_breakout_signal'] * 0.15)
    data['microstructure_boost'] = data['base_signal'] * (1 + data['microstructure_score'] * 0.12)
    data['breakout_emphasis'] = data['base_signal'] * (1 + data['breakout_validation'] * 0.18)
    data['volume_confirmation'] = data['base_signal'] * (1 + data['volume_strength'] * 0.10)
    
    # Final Output Synthesis
    data['composite_factor'] = (
        data['compression_context'] * 0.25 +
        data['microstructure_boost'] * 0.25 +
        data['breakout_emphasis'] * 0.30 +
        data['volume_confirmation'] * 0.20
    )
    
    # Clean up and return
    result = data['composite_factor'].replace([np.inf, -np.inf], np.nan)
    return result
