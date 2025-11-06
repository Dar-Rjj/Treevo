import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Multi-Timeframe Alpha Factor combining momentum, volume dynamics,
    volatility-normalized returns, and regime-aware scaling.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Timeframe Momentum Alignment
    # Short-term momentum (1-3 days)
    data['price_momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    
    # Volume-weighted momentum (3-day)
    vol_weight_3d = []
    for i in range(len(data)):
        if i >= 2:
            weights = data['volume'].iloc[i-2:i+1].values
            price_ratios = data['close'].iloc[i] / data['close'].iloc[i-2:i+1].values
            vol_weight_3d.append(np.sum(price_ratios * weights) / np.sum(weights))
        else:
            vol_weight_3d.append(np.nan)
    data['vol_weighted_momentum_3d'] = vol_weight_3d
    
    # Range-normalized momentum
    data['range_norm_momentum'] = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Medium-term momentum (5-10 days)
    # Decay-weighted momentum (10-day with exponential decay)
    decay_momentum = []
    for i in range(len(data)):
        if i >= 9:
            returns = []
            weights = []
            for j in range(10):
                if i-j > 0:
                    ret = data['close'].iloc[i] / data['close'].iloc[i-j-1] - 1
                    weight = np.exp(-0.2 * j)
                    returns.append(ret)
                    weights.append(weight)
            decay_momentum.append(np.sum(np.array(returns) * np.array(weights)) / np.sum(weights))
        else:
            decay_momentum.append(np.nan)
    data['decay_momentum_10d'] = decay_momentum
    
    # Volatility-scaled momentum (5-day return / 10-day vol)
    data['vol_scaled_momentum_5d'] = (data['close'] / data['close'].shift(5) - 1) / data['close'].rolling(window=10).std()
    
    # Volume-confirmed momentum
    vol_mean_10d = data['volume'].rolling(window=10).mean()
    data['vol_confirmed_momentum'] = (data['close'] / data['close'].shift(5) - 1) * (data['volume'] / vol_mean_10d)
    
    # Long-term momentum (20-60 days) with regime detection
    # Price trajectory for regime detection
    data['price_trend_20d'] = data['close'].rolling(window=20).apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)
    data['price_trend_60d'] = data['close'].rolling(window=60).apply(lambda x: (x[-1] - x[0]) / x[0], raw=True)
    
    # Trend-consistent momentum filtering
    data['trend_consistency'] = np.sign(data['price_trend_20d']) * np.sign(data['price_trend_60d'])
    
    # Volatility regime adjustment
    vol_20d = data['close'].rolling(window=20).std()
    vol_60d = data['close'].rolling(window=60).std()
    data['vol_regime'] = vol_20d / vol_60d
    
    # Volume Acceleration Dynamics
    # Volume trend analysis
    data['vol_slope_3d'] = (data['volume'] - data['volume'].shift(3)) / 3
    data['vol_accel_10d'] = (data['volume'] - 2 * data['volume'].shift(5) + data['volume'].shift(10)) / 25
    
    # Volume regime classification
    vol_mean_20d = data['volume'].rolling(window=20).mean()
    data['vol_regime_class'] = data['volume'] / vol_mean_20d
    
    # Volume-Price Convergence
    price_roc_5d = data['close'].pct_change(5)
    vol_roc_5d = data['volume'].pct_change(5)
    data['vol_price_divergence'] = price_roc_5d - vol_roc_5d
    
    # Volume Quality Assessment
    # Large trade concentration (using amount/volume as proxy)
    data['trade_size'] = data['amount'] / data['volume']
    data['trade_size_persistence'] = data['trade_size'].rolling(window=5).std()
    
    # Volatility-Normalized Returns Framework
    # Multi-scale volatility estimation
    range_vol = (data['high'] - data['low']) / data['close']
    close_vol = data['close'].pct_change().rolling(window=10).std()
    data['multi_scale_vol'] = 0.6 * range_vol + 0.4 * close_vol
    
    # Risk-adjusted return calculation
    data['vol_normalized_return'] = data['price_momentum_1d'] / data['multi_scale_vol']
    
    # Volatility regime detection
    vol_quantile_20d = data['multi_scale_vol'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['high_vol_regime'] = (vol_quantile_20d > 0.7).astype(int)
    data['low_vol_regime'] = (vol_quantile_20d < 0.3).astype(int)
    
    # Regime-Aware Scaling Mechanism
    # Market regime identification
    trend_strength = data['price_trend_20d'].abs()
    data['trend_regime'] = (trend_strength > trend_strength.rolling(window=20).quantile(0.7)).astype(int)
    
    # Adaptive weighting scheme
    # Timeframe weights based on regime
    short_weight = np.where(data['trend_regime'] == 1, 0.3, 0.5)
    medium_weight = np.where(data['trend_regime'] == 1, 0.4, 0.3)
    long_weight = np.where(data['trend_regime'] == 1, 0.3, 0.2)
    
    # Volume confirmation strength adjustment
    vol_conf_strength = np.clip(data['vol_regime_class'], 0.5, 2.0)
    
    # Volatility scaling factor adaptation
    vol_scale = np.where(data['high_vol_regime'] == 1, 0.7, 
                         np.where(data['low_vol_regime'] == 1, 1.3, 1.0))
    
    # Multi-Timeframe Factor Integration
    # Signal alignment across timeframes
    short_signals = (data['price_momentum_1d'] + data['vol_weighted_momentum_3d'] + data['range_norm_momentum']) / 3
    medium_signals = (data['decay_momentum_10d'] + data['vol_scaled_momentum_5d'] + data['vol_confirmed_momentum']) / 3
    long_signals = data['price_trend_20d'] * data['trend_consistency'] / np.maximum(data['vol_regime'], 0.1)
    
    # Consistency scoring
    signal_alignment = (np.sign(short_signals) == np.sign(medium_signals)) & (np.sign(medium_signals) == np.sign(long_signals))
    alignment_strength = signal_alignment.astype(float)
    
    # Robust combination methodology
    # Volatility-weighted signal aggregation
    short_component = short_signals * short_weight * vol_conf_strength * vol_scale
    medium_component = medium_signals * medium_weight * vol_conf_strength * vol_scale
    long_component = long_signals * long_weight * vol_conf_strength * vol_scale
    
    # Regime-dependent signal mixing with volume confirmation
    volume_boost = np.where(data['vol_accel_10d'] > 0, 1.2, 0.8)
    divergence_penalty = np.where(data['vol_price_divergence'].abs() > 0.1, 0.8, 1.0)
    
    # Final composite score
    composite_score = (
        short_component * 0.4 + 
        medium_component * 0.35 + 
        long_component * 0.25
    ) * alignment_strength * volume_boost * divergence_penalty
    
    # Outlier handling and winsorization
    final_factor = pd.Series(composite_score, index=data.index)
    q_low = final_factor.quantile(0.05)
    q_high = final_factor.quantile(0.95)
    final_factor = np.clip(final_factor, q_low, q_high)
    
    # Normalize final factor
    final_factor = (final_factor - final_factor.mean()) / final_factor.std()
    
    return final_factor
