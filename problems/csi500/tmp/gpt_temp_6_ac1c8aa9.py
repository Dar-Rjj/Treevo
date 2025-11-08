import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum with Liquidity Confirmation
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Multi-Timeframe Volatility Analysis
    # Short-term volatility (5-day)
    short_term_vol = true_range.rolling(window=5, min_periods=3).mean()
    
    # Medium-term volatility (10-day)
    medium_term_vol = true_range.rolling(window=10, min_periods=5).mean()
    
    # Volatility ratio and regime identification
    vol_ratio = short_term_vol / medium_term_vol
    volatility_regime = np.where(vol_ratio > 1.2, 'high', 
                                np.where(vol_ratio < 0.8, 'low', 'normal'))
    
    # Volatility acceleration
    vol_acceleration = (short_term_vol - short_term_vol.shift(3)) / short_term_vol.shift(3)
    vol_trend = (short_term_vol - short_term_vol.shift(8)) / short_term_vol.shift(8)
    
    # Adaptive Momentum Calculation
    # Short-term momentum (5-day)
    short_momentum = (df['close'] - df['close'].shift(4)) / df['close'].shift(4)
    short_momentum_vol_adj = short_momentum / (short_term_vol + 1e-8)
    
    # Medium-term momentum (10-day)
    medium_momentum = (df['close'] - df['close'].shift(9)) / df['close'].shift(9)
    momentum_decay = short_momentum / (medium_momentum + 1e-8)
    
    # Volatility-weighted momentum adjustment
    high_vol_momentum = np.where(volatility_regime == 'high', 
                                -short_momentum_vol_adj * short_term_vol,
                                short_momentum_vol_adj)
    
    low_vol_momentum = np.where(volatility_regime == 'low',
                               short_momentum * (1 + vol_acceleration),
                               short_momentum)
    
    regime_adaptive_momentum = np.where(volatility_regime == 'high', high_vol_momentum,
                                      np.where(volatility_regime == 'low', low_vol_momentum,
                                              short_momentum_vol_adj))
    
    # Liquidity Dynamics Integration
    # Order flow analysis
    buying_pressure = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    order_flow_imbalance = 2 * buying_pressure - 1
    
    # Order flow persistence (3-day autocorrelation)
    order_flow_3d = order_flow_imbalance.rolling(window=3, min_periods=2)
    order_flow_persistence = order_flow_3d.apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False)
    
    # Volume momentum
    volume_5d_change = df['volume'] / df['volume'].rolling(window=5, min_periods=3).mean()
    volume_15d_baseline = df['volume'] / df['volume'].rolling(window=15, min_periods=8).mean()
    volume_momentum = volume_5d_change / volume_15d_baseline
    
    # Liquidity gradient (price impact per unit volume)
    avg_trade_size = df['amount'] / (df['volume'] + 1e-8)
    price_change = df['close'].pct_change()
    liquidity_gradient = price_change.rolling(window=5, min_periods=3).apply(
        lambda x: np.corrcoef(x, avg_trade_size.reindex(x.index))[0,1] if len(x) > 1 and not np.isnan(x).any() else 0, 
        raw=False
    )
    
    # Liquidity momentum
    liquidity_momentum = avg_trade_size.pct_change(periods=5)
    
    # Integrated Signal Generation
    # High volatility signal logic
    high_vol_signal = (-regime_adaptive_momentum * order_flow_imbalance * 
                      volume_momentum * short_term_vol)
    
    # Low volatility signal logic
    low_vol_signal = (regime_adaptive_momentum * order_flow_persistence * 
                     volume_momentum * (1 + vol_acceleration))
    
    # Regime-adaptive signal construction
    regime_signal = np.where(volatility_regime == 'high', high_vol_signal,
                           np.where(volatility_regime == 'low', low_vol_signal,
                                   regime_adaptive_momentum * order_flow_imbalance))
    
    # Multi-dimensional confirmation
    # Price-volume-liquidity alignment
    momentum_direction = np.sign(regime_adaptive_momentum)
    volume_direction = np.sign(volume_momentum - 1)
    liquidity_direction = np.sign(liquidity_gradient)
    
    alignment_score = (momentum_direction * volume_direction * liquidity_direction + 3) / 6
    
    # Timeframe convergence
    timeframe_convergence = np.sign(short_momentum) * np.sign(medium_momentum)
    
    # Final alpha factor with confidence weighting
    final_signal = (regime_signal * alignment_score * 
                   (1 + 0.5 * timeframe_convergence) * 
                   (1 + 0.3 * order_flow_persistence))
    
    # Clean up and return
    alpha_factor = pd.Series(final_signal, index=df.index)
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    
    return alpha_factor
