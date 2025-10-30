import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility Regime Momentum with Microstructure Signals
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility Regime Classification
    # Multi-Timeframe Volatility Assessment
    data['ret'] = data['close'].pct_change()
    data['5d_rv'] = data['ret'].rolling(window=5).std()
    data['20d_rv'] = data['ret'].rolling(window=20).std()
    
    # Range-based volatility (High-Low Normalized)
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['10d_range_vol'] = data['daily_range'].rolling(window=10).mean()
    
    # Volatility Persistence (Autocorrelation)
    data['vol_persistence'] = data['5d_rv'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Volatility Regime Identification
    data['vol_regime'] = 0  # Transition regime by default
    high_vol_condition = data['5d_rv'] > (data['20d_rv'] * 1.5)
    low_vol_condition = data['5d_rv'] < (data['20d_rv'] * 0.7)
    data.loc[high_vol_condition, 'vol_regime'] = 1  # High volatility
    data.loc[low_vol_condition, 'vol_regime'] = -1  # Low volatility
    
    # 2. Microstructure-Based Momentum Enhancement
    # Bid-Ask Spread Proxy
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['effective_spread'] = 2 * abs(data['close'] - data['mid_price']) / data['mid_price']
    data['5d_spread_persistence'] = data['effective_spread'].rolling(window=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Volume-Weighted Price Efficiency
    data['vwap'] = (data['amount'] / data['volume']).replace([np.inf, -np.inf], np.nan)
    data['vwap_close_divergence'] = (data['close'] - data['vwap']) / data['close']
    
    # Volume-Weighted Return Efficiency
    data['volume_weighted_ret'] = data['ret'] * data['volume']
    data['5d_vw_efficiency'] = data['volume_weighted_ret'].rolling(window=5).mean()
    
    # Order Flow Imbalance Signals
    data['price_volume_corr'] = data['close'].rolling(window=5).corr(data['volume'])
    
    # 3. Multi-Frequency Momentum Alignment
    # Intraday Momentum Patterns (using open, high, low, close)
    data['morning_momentum'] = (data['high'].shift(1) - data['open'].shift(1)) / data['open'].shift(1)
    data['afternoon_momentum'] = (data['close'].shift(1) - data['high'].shift(1)) / data['high'].shift(1)
    
    # Overnight Gap Integration
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_vs_range'] = abs(data['overnight_gap']) / data['daily_range'].shift(1)
    
    # Multi-Scale Momentum
    data['momentum_1d'] = data['close'].pct_change(periods=1)
    data['momentum_3d'] = data['close'].pct_change(periods=3)
    data['momentum_5d'] = data['close'].pct_change(periods=5)
    data['momentum_10d'] = data['close'].pct_change(periods=10)
    
    # Momentum Acceleration
    data['momentum_accel'] = data['momentum_3d'] - data['momentum_10d']
    
    # 4. Liquidity-Adjusted Breakout Signals
    # Dynamic Support/Resistance Levels
    data['5d_high'] = data['high'].rolling(window=5).max()
    data['5d_low'] = data['low'].rolling(window=5).min()
    data['breakout_signal'] = ((data['close'] > data['5d_high'].shift(1)) * 1 + 
                              (data['close'] < data['5d_low'].shift(1)) * -1)
    
    # Volume Concentration Analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_5d_avg']
    
    # 5. Adaptive Signal Integration
    # Regime-specific momentum calculations
    data['high_vol_momentum'] = np.where(
        data['vol_regime'] == 1,
        -data['momentum_1d'] * data['price_volume_corr'],  # Reversal in high vol
        data['momentum_5d']  # Trend in other regimes
    )
    
    data['low_vol_momentum'] = np.where(
        data['vol_regime'] == -1,
        data['momentum_5d'] * (1 + data['momentum_accel']),  # Accelerated trend
        data['momentum_3d']  # Medium-term in other regimes
    )
    
    # Microstructure-enhanced signals
    data['micro_momentum'] = (
        data['vwap_close_divergence'] * data['5d_vw_efficiency'] * 
        (1 - data['effective_spread'])
    )
    
    # Breakout momentum with liquidity adjustment
    data['liquidity_breakout'] = (
        data['breakout_signal'] * data['volume_ratio'] * 
        (1 - data['effective_spread'])
    )
    
    # 6. Final Alpha Generation
    # Volatility-normalized components
    volatility_normalizer = data['5d_rv'].replace(0, np.nan)
    
    # Regime-adaptive weighting
    high_vol_weight = np.where(data['vol_regime'] == 1, 0.6, 0.2)
    low_vol_weight = np.where(data['vol_regime'] == -1, 0.6, 0.2)
    transition_weight = np.where(data['vol_regime'] == 0, 0.6, 0.2)
    
    # Component signals
    high_vol_component = (data['high_vol_momentum'] + data['micro_momentum'] + 
                         data['liquidity_breakout']) / 3
    low_vol_component = (data['low_vol_momentum'] + data['momentum_accel'] + 
                        data['5d_vw_efficiency']) / 3
    transition_component = (data['momentum_3d'] + data['micro_momentum'] + 
                           data['price_volume_corr']) / 3
    
    # Final alpha with regime-adaptive weighting
    alpha = (
        high_vol_weight * high_vol_component +
        low_vol_weight * low_vol_component +
        transition_weight * transition_component
    )
    
    # Volatility normalization
    alpha_normalized = alpha / volatility_normalizer
    
    # Clean up and return
    alpha_series = alpha_normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
