import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factors using multiple novel approaches including:
    - Price-Velocity Volume Divergence
    - Pressure Accumulation Breakout
    - Efficiency-Weighted Momentum
    - Volume-Profile Trend Confirmation
    - Range-Compression Expansion Signal
    - Asymmetric Return Impact
    - Liquidity-Flow Momentum
    - Price-Volume Concurrence
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price-Velocity Volume Divergence
    # Calculate price velocity (first derivative of price movement)
    price_velocity = data['close'].diff(3) / 3  # 3-day price velocity
    
    # Compute volume momentum (rate of change in trading activity)
    volume_momentum = data['volume'].pct_change(5).rolling(window=5, min_periods=3).mean()
    
    # Detect divergence patterns
    price_velocity_sign = np.sign(price_velocity)
    volume_momentum_sign = np.sign(volume_momentum)
    divergence_strength = (price_velocity_sign != volume_momentum_sign).astype(int) * np.abs(price_velocity)
    
    # Weight by recent volatility
    recent_volatility = data['close'].pct_change().rolling(window=10, min_periods=5).std()
    pvvd_factor = divergence_strength / (recent_volatility + 1e-8)
    
    # 2. Pressure Accumulation Breakout
    # Calculate buying and selling pressure
    buying_pressure = (data['high'] - data['open']).clip(lower=0)
    selling_pressure = (data['open'] - data['low']).clip(lower=0)
    
    # Cumulative pressure with decay
    cum_buy_pressure = buying_pressure.rolling(window=5, min_periods=3).sum()
    cum_sell_pressure = selling_pressure.rolling(window=5, min_periods=3).sum()
    
    # Pressure imbalance
    pressure_imbalance = (cum_buy_pressure - cum_sell_pressure) / (cum_buy_pressure + cum_sell_pressure + 1e-8)
    
    # Breakout detection using historical extremes
    pressure_zscore = (pressure_imbalance - pressure_imbalance.rolling(window=20, min_periods=10).mean()) / \
                     (pressure_imbalance.rolling(window=20, min_periods=10).std() + 1e-8)
    
    pab_factor = pressure_zscore * data['close'].pct_change(3)
    
    # 3. Efficiency-Weighted Momentum
    # Price efficiency: how efficiently price moves within daily range
    daily_range = data['high'] - data['low']
    price_efficiency = (data['close'] - data['open']).abs() / (daily_range + 1e-8)
    
    # Momentum quality (consistency and strength)
    momentum_5d = data['close'].pct_change(5)
    momentum_consistency = momentum_5d.rolling(window=5, min_periods=3).std()
    momentum_quality = np.abs(momentum_5d) / (momentum_consistency + 1e-8)
    
    # Efficiency-weighted momentum
    ewm_factor = momentum_5d * price_efficiency * momentum_quality
    
    # 4. Volume-Profile Trend Confirmation
    # Volume concentration analysis
    avg_trade_size = data['amount'] / (data['volume'] + 1e-8)
    volume_concentration = avg_trade_size.rolling(window=10, min_periods=5).std()
    
    # Trend alignment with volume
    price_trend = data['close'].rolling(window=5, min_periods=3).mean().pct_change(3)
    volume_trend = data['volume'].rolling(window=5, min_periods=3).mean().pct_change(3)
    
    trend_alignment = np.sign(price_trend) == np.sign(volume_trend)
    vptc_factor = trend_alignment.astype(float) * np.abs(price_trend) * (1 - volume_concentration / volume_concentration.rolling(window=20).max())
    
    # 5. Range-Compression Expansion Signal
    # Range compression measurement
    range_5d_avg = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    range_20d_avg = (data['high'] - data['low']).rolling(window=20, min_periods=10).mean()
    compression_ratio = range_5d_avg / (range_20d_avg + 1e-8)
    
    # Compression duration
    low_range_days = (compression_ratio < 0.7).astype(int)
    compression_duration = low_range_days.rolling(window=10, min_periods=5).sum()
    
    # Expansion detection
    range_expansion = (data['high'] - data['low']) / range_20d_avg - 1
    rces_factor = range_expansion * compression_duration * np.sign(data['close'].pct_change())
    
    # 6. Asymmetric Return Impact
    # Return asymmetry
    returns = data['close'].pct_change()
    positive_returns = returns.clip(lower=0)
    negative_returns = (-returns).clip(lower=0)
    
    return_asymmetry = (positive_returns.rolling(window=10, min_periods=5).std() - 
                       negative_returns.rolling(window=10, min_periods=5).std())
    
    # Volume sensitivity to returns
    volume_response_pos = data['volume'].pct_change().rolling(window=5, min_periods=3).corr(positive_returns)
    volume_response_neg = data['volume'].pct_change().rolling(window=5, min_periods=3).corr(negative_returns)
    
    volume_sensitivity_diff = volume_response_pos - volume_response_neg
    
    ari_factor = return_asymmetry * volume_sensitivity_diff * returns
    
    # 7. Liquidity-Flow Momentum
    # Effective liquidity (money flow efficiency)
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    liquidity_efficiency = money_flow.rolling(window=5, min_periods=3).std() / (money_flow.rolling(window=5, min_periods=3).mean() + 1e-8)
    
    # Momentum persistence
    momentum_persistence = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x[i] - x[i-1]) == np.sign(x[0] - x[-1])]) / (len(x) - 1) if len(x) > 1 else 0
    )
    
    lfm_factor = data['close'].pct_change(5) * momentum_persistence / (liquidity_efficiency + 1e-8)
    
    # 8. Price-Volume Concurrence
    # Synchronization measurement
    price_returns = data['close'].pct_change()
    volume_returns = data['volume'].pct_change()
    
    concurrence = price_returns.rolling(window=10, min_periods=5).corr(volume_returns)
    
    # Regime identification based on concurrence
    high_concurrence = concurrence > concurrence.rolling(window=20, min_periods=10).quantile(0.7)
    low_concurrence = concurrence < concurrence.rolling(window=20, min_periods=10).quantile(0.3)
    
    pvc_factor = np.where(high_concurrence, price_returns * 1.5, 
                         np.where(low_concurrence, price_returns * 0.5, price_returns))
    
    # Risk adjustment using recent volatility
    volatility = data['close'].pct_change().rolling(window=10, min_periods=5).std()
    pvc_factor = pvc_factor / (volatility + 1e-8)
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'pvvd': pvvd_factor,
        'pab': pab_factor,
        'ewm': ewm_factor,
        'vptc': vptc_factor,
        'rces': rces_factor,
        'ari': ari_factor,
        'lfm': lfm_factor,
        'pvc': pvc_factor
    })
    
    # Remove any infinite values and fill NaN
    factors = factors.replace([np.inf, -np.inf], np.nan)
    factors = factors.fillna(method='ffill').fillna(0)
    
    # Z-score normalization for each factor
    for col in factors.columns:
        mean = factors[col].rolling(window=20, min_periods=10).mean()
        std = factors[col].rolling(window=20, min_periods=10).std()
        factors[col] = (factors[col] - mean) / (std + 1e-8)
    
    # Equal-weighted combination
    combined_factor = factors.mean(axis=1)
    
    return combined_factor
