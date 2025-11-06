import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators:
    1. Dynamic Volatility-Adjusted Price Momentum
    2. Volume-Price Divergence Factor
    3. Intraday Strength Persistence
    4. Amount-Based Order Flow Imbalance
    5. Volatility Regime Adaptive Factor
    6. Price-Volume Co-movement Efficiency
    7. Multi-timeframe Momentum Convergence
    8. Extreme Event Recovery Factor
    9. Volume-Weighted Price Acceleration
    10. Opening Gap Persistence Factor
    """
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # 1. Dynamic Volatility-Adjusted Price Momentum
    momentum_window = 20
    vol_window = 20
    
    # Calculate rolling momentum
    momentum = data['close'].pct_change(periods=momentum_window)
    
    # Calculate dynamic volatility using daily range
    daily_range = (data['high'] - data['low']) / data['close']
    volatility = daily_range.rolling(window=vol_window, min_periods=1).std()
    
    # Adjust momentum by volatility (handle zero volatility)
    vol_adjusted_momentum = momentum / (volatility + 1e-8)
    
    # 2. Volume-Price Divergence Factor
    trend_window = 10
    
    # Create time index for correlation
    time_index = np.arange(trend_window)
    
    def rolling_correlation(series, window):
        correlations = []
        for i in range(len(series)):
            if i >= window - 1:
                window_data = series.iloc[i-window+1:i+1].reset_index(drop=True)
                if len(window_data) == window:
                    corr = np.corrcoef(time_index, window_data.values)[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
            else:
                correlations.append(0)
        return pd.Series(correlations, index=series.index)
    
    price_trend = rolling_correlation(data['close'], trend_window)
    volume_trend = rolling_correlation(data['volume'], trend_window)
    
    # Compute divergence
    volume_price_divergence = (price_trend - volume_trend) * abs(price_trend)
    
    # 3. Intraday Strength Persistence
    intraday_strength = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Calculate rolling autocorrelation for persistence
    persistence_window = 5
    autocorr_values = []
    for i in range(len(intraday_strength)):
        if i >= persistence_window:
            window_data = intraday_strength.iloc[i-persistence_window:i+1]
            if len(window_data) > 1:
                autocorr = window_data.autocorr(lag=1)
                autocorr_values.append(autocorr if not np.isnan(autocorr) else 0)
            else:
                autocorr_values.append(0)
        else:
            autocorr_values.append(0)
    
    persistence = pd.Series(autocorr_values, index=data.index)
    intraday_signal = intraday_strength * persistence
    
    # 4. Amount-Based Order Flow Imbalance
    # Classify trades using price movement
    price_change = data['close'].pct_change()
    buyer_initiated = data['amount'] * (price_change > 0).astype(int)
    seller_initiated = data['amount'] * (price_change < 0).astype(int)
    
    # Calculate rolling imbalance
    imbalance_window = 10
    buyer_sum = buyer_initiated.rolling(window=imbalance_window, min_periods=1).sum()
    seller_sum = seller_initiated.rolling(window=imbalance_window, min_periods=1).sum()
    
    order_imbalance = (buyer_sum - seller_sum) / (buyer_sum + seller_sum + 1e-8)
    
    # Apply exponential smoothing
    order_imbalance_smooth = order_imbalance.ewm(span=5).mean()
    
    # Scale by volume significance
    volume_ratio = data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    order_flow_signal = order_imbalance_smooth * volume_ratio
    
    # 5. Volatility Regime Adaptive Factor
    volatility_ma = daily_range.rolling(window=50, min_periods=1).mean()
    volatility_std = daily_range.rolling(window=50, min_periods=1).std()
    
    # Classify volatility regime
    high_vol_regime = (daily_range > (volatility_ma + volatility_std)).astype(int)
    low_vol_regime = (daily_range < (volatility_ma - volatility_std)).astype(int)
    normal_regime = 1 - high_vol_regime - low_vol_regime
    
    # Mean reversion factor for high volatility
    mean_reversion = -data['close'].pct_change(periods=5)
    
    # Momentum factor for low volatility
    low_vol_momentum = data['close'].pct_change(periods=10)
    
    # Combine regime-specific factors
    regime_factor = (high_vol_regime * mean_reversion + 
                    low_vol_regime * low_vol_momentum + 
                    normal_regime * (mean_reversion + low_vol_momentum) / 2)
    
    # 6. Price-Volume Co-movement Efficiency
    price_efficiency_window = 10
    price_changes = data['close'].pct_change().abs()
    
    # Calculate price movement efficiency (variance ratio)
    def variance_ratio(series, window):
        ratios = []
        for i in range(len(series)):
            if i >= window - 1:
                window_data = series.iloc[i-window+1:i+1]
                if len(window_data) >= 2:
                    var_1day = window_data.var()
                    var_multiday = (window_data.sum() / len(window_data)) ** 2
                    ratio = var_1day / (var_multiday + 1e-8)
                    ratios.append(ratio if not np.isnan(ratio) else 1)
                else:
                    ratios.append(1)
            else:
                ratios.append(1)
        return pd.Series(ratios, index=series.index)
    
    price_efficiency = variance_ratio(price_changes, price_efficiency_window)
    
    # Volume efficiency (predictability)
    volume_predictability = data['volume'].rolling(window=10, min_periods=1).std() / \
                           (data['volume'].rolling(window=10, min_periods=1).mean() + 1e-8)
    volume_efficiency = 1 / (volume_predictability + 1e-8)
    
    efficiency_signal = price_efficiency * volume_efficiency
    
    # 7. Multi-timeframe Momentum Convergence
    short_momentum = data['close'].pct_change(periods=5)
    medium_momentum = data['close'].pct_change(periods=15)
    long_momentum = data['close'].pct_change(periods=60)
    
    # Calculate convergence using rolling correlation
    def rolling_momentum_correlation(m1, m2, window):
        correlations = []
        for i in range(len(m1)):
            if i >= window - 1:
                m1_window = m1.iloc[i-window+1:i+1]
                m2_window = m2.iloc[i-window+1:i+1]
                if len(m1_window) == window and len(m2_window) == window:
                    corr = np.corrcoef(m1_window, m2_window)[0, 1]
                    correlations.append(corr if not np.isnan(corr) else 0)
                else:
                    correlations.append(0)
            else:
                correlations.append(0)
        return pd.Series(correlations, index=m1.index)
    
    short_medium_corr = rolling_momentum_correlation(short_momentum, medium_momentum, 10)
    medium_long_corr = rolling_momentum_correlation(medium_momentum, long_momentum, 10)
    
    momentum_convergence = (short_medium_corr + medium_long_corr) / 2
    
    # 8. Extreme Event Recovery Factor
    extreme_threshold = data['close'].pct_change().abs().rolling(window=50, min_periods=1).quantile(0.95)
    extreme_events = (data['close'].pct_change().abs() > extreme_threshold).astype(int)
    
    # Track recovery (price movement in opposite direction after extreme event)
    recovery_signal = pd.Series(0.0, index=data.index)
    for i in range(2, len(data)):
        if extreme_events.iloc[i-1] == 1:
            # Check if price is recovering (moving opposite to extreme move)
            prev_return = data['close'].iloc[i-1] - data['close'].iloc[i-2]
            current_return = data['close'].iloc[i] - data['close'].iloc[i-1]
            if prev_return * current_return < 0:  # Opposite direction
                recovery_signal.iloc[i] = abs(current_return) / (abs(prev_return) + 1e-8)
    
    # 9. Volume-Weighted Price Acceleration
    # Calculate price acceleration (second derivative of returns)
    returns = data['close'].pct_change()
    acceleration = returns.diff()
    
    # Volume confidence
    volume_ma = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_confidence = data['volume'] / (volume_ma + 1e-8)
    
    volume_weighted_acceleration = acceleration * volume_confidence
    
    # 10. Opening Gap Persistence Factor
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Calculate gap persistence (how long gap remains unfilled)
    gap_persistence = pd.Series(0.0, index=data.index)
    for i in range(1, len(data)):
        if abs(opening_gap.iloc[i]) > 0.01:  # Significant gap
            gap_direction = np.sign(opening_gap.iloc[i])
            # Check if gap is being filled
            intraday_move = data['close'].iloc[i] - data['open'].iloc[i]
            if gap_direction * intraday_move < 0:  # Moving to fill gap
                gap_persistence.iloc[i] = -abs(intraday_move) / (abs(opening_gap.iloc[i]) + 1e-8)
            else:  # Gap persisting
                gap_persistence.iloc[i] = abs(intraday_move) / (abs(opening_gap.iloc[i]) + 1e-8)
    
    opening_gap_signal = opening_gap * gap_persistence
    
    # Combine all factors with equal weighting
    factors = [
        vol_adjusted_momentum,
        volume_price_divergence,
        intraday_signal,
        order_flow_signal,
        regime_factor,
        efficiency_signal,
        momentum_convergence,
        recovery_signal,
        volume_weighted_acceleration,
        opening_gap_signal
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0.0, index=data.index)
    for factor in factors:
        # Z-score normalization
        factor_normalized = (factor - factor.rolling(window=50, min_periods=1).mean()) / \
                           (factor.rolling(window=50, min_periods=1).std() + 1e-8)
        combined_factor += factor_normalized
    
    # Final alpha factor
    alpha = combined_factor / len(factors)
    
    return alpha
