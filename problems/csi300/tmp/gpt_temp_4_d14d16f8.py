import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    if len(data) < 8:
        return factor
    
    # Price-Volume Convergence
    # 3-day vs 8-day convergence
    close_3d_ratio = data['close'] / data['close'].shift(3)
    close_8d_ratio = data['close'] / data['close'].shift(8)
    volume_3d_ratio = data['volume'] / data['volume'].shift(3)
    volume_8d_ratio = data['volume'] / data['volume'].shift(8)
    
    price_volume_convergence = (close_3d_ratio / close_8d_ratio) / (volume_3d_ratio / volume_8d_ratio)
    
    # Momentum alignment
    price_change = data['close'] - data['close'].shift(1)
    volume_change = data['volume'] - data['volume'].shift(1)
    momentum_alignment = np.sign(price_change) * np.sign(volume_change) * np.where(
        volume_change != 0, 
        np.abs(price_change) / np.abs(volume_change), 
        0
    )
    
    # Order Flow Microstructure
    # Effective spread
    mid_price = (data['high'] + data['low']) / 2
    effective_spread = (data['high'] - data['low']) / np.where(mid_price != 0, mid_price, 1)
    
    # Price impact
    price_impact = (data['close'] - data['open']) / np.where(data['amount'] != 0, data['amount'], 1)
    
    # Large trade concentration
    large_trade_concentration = np.where(data['volume'] != 0, data['amount'] / data['volume'], 0)
    
    # Temporal Patterns
    # Intraday momentum
    intraday_momentum = np.where(
        (data['high'] - data['low']) != 0,
        (data['close'] - data['open']) / (data['high'] - data['low']),
        0
    )
    
    # End-of-day acceleration
    eod_acceleration = np.where(
        (data['high'] - data['low']) != 0,
        (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']),
        0
    )
    
    # Directional consistency (count positive returns over 3 days)
    positive_returns = (data['close'] > data['close'].shift(1)).astype(int)
    directional_consistency = positive_returns.rolling(window=3, min_periods=1).sum()
    
    # Liquidity Assessment
    # Liquidity-to-volatility
    liquidity_to_volatility = np.where(
        (data['high'] - data['low']) != 0,
        data['volume'] / (data['high'] - data['low']),
        0
    )
    
    # Price efficiency
    price_efficiency = np.where(
        (data['high'] - data['low']) != 0,
        np.abs(data['close'] - data['open']) / (data['high'] - data['low']),
        0
    )
    
    # Combine all components with appropriate weights
    # Normalize each component by its rolling standard deviation to ensure comparable scales
    window = 20
    
    # Price-Volume Convergence components
    pv_conv_norm = price_volume_convergence / price_volume_convergence.rolling(window=window, min_periods=1).std()
    mom_align_norm = momentum_alignment / momentum_alignment.rolling(window=window, min_periods=1).std()
    
    # Order Flow components
    eff_spread_norm = effective_spread / effective_spread.rolling(window=window, min_periods=1).std()
    price_impact_norm = price_impact / price_impact.rolling(window=window, min_periods=1).std()
    large_trade_norm = large_trade_concentration / large_trade_concentration.rolling(window=window, min_periods=1).std()
    
    # Temporal Patterns components
    intraday_mom_norm = intraday_momentum / intraday_momentum.rolling(window=window, min_periods=1).std()
    eod_accel_norm = eod_acceleration / eod_acceleration.rolling(window=window, min_periods=1).std()
    dir_cons_norm = directional_consistency / directional_consistency.rolling(window=window, min_periods=1).std()
    
    # Liquidity Assessment components
    liq_vol_norm = liquidity_to_volatility / liquidity_to_volatility.rolling(window=window, min_periods=1).std()
    price_eff_norm = price_efficiency / price_efficiency.rolling(window=window, min_periods=1).std()
    
    # Final factor combination with weights
    factor = (
        0.15 * pv_conv_norm +  # Price-Volume Convergence
        0.15 * mom_align_norm +  # Momentum Alignment
        0.10 * eff_spread_norm +  # Effective Spread
        0.10 * price_impact_norm +  # Price Impact
        0.10 * large_trade_norm +  # Large Trade Concentration
        0.10 * intraday_mom_norm +  # Intraday Momentum
        0.10 * eod_accel_norm +  # End-of-day Acceleration
        0.10 * dir_cons_norm +  # Directional Consistency
        0.10 * liq_vol_norm +  # Liquidity-to-Volatility
        0.10 * price_eff_norm   # Price Efficiency
    )
    
    # Handle NaN values
    factor = factor.fillna(0)
    
    return factor
