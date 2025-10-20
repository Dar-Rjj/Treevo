import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Period Momentum Divergence with Adaptive Weighting
    # Calculate momentum across different timeframes
    momentum_3d = df['close'].pct_change(3)
    momentum_8d = df['close'].pct_change(8)
    momentum_21d = df['close'].pct_change(21)
    momentum_55d = df['close'].pct_change(55)
    
    # Compute momentum divergence patterns
    momentum_series = pd.DataFrame({
        'm3': momentum_3d,
        'm8': momentum_8d,
        'm21': momentum_21d,
        'm55': momentum_55d
    })
    
    # Divergence intensity: max absolute difference between adjacent timeframes
    divergence_intensity = pd.DataFrame({
        'd1': (momentum_series['m3'] - momentum_series['m8']).abs(),
        'd2': (momentum_series['m8'] - momentum_series['m21']).abs(),
        'd3': (momentum_series['m21'] - momentum_series['m55']).abs()
    }).max(axis=1)
    
    # Momentum convergence: variance across all timeframe returns
    momentum_convergence = momentum_series.rolling(window=10, min_periods=5).std().mean(axis=1)
    
    # Adaptive weighting based on recent volatility
    recent_volatility = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    volatility_weight = 1 / (1 + recent_volatility)
    
    # Volume-Amount Asymmetry and Imbalance Detection
    # Volume distribution analysis
    volume_concentration = df['volume'].rolling(window=5, min_periods=3).std() / \
                          df['volume'].rolling(window=20, min_periods=10).std()
    
    volume_skewness = df['volume'].rolling(window=20, min_periods=10).skew()
    
    # Volume clustering: consecutive high/low volume days
    volume_ma20 = df['volume'].rolling(window=20, min_periods=10).mean()
    high_volume_days = (df['volume'] > volume_ma20 * 1.2).rolling(window=3).sum()
    low_volume_days = (df['volume'] < volume_ma20 * 0.8).rolling(window=3).sum()
    volume_clustering = (high_volume_days - low_volume_days) / 3
    
    # Amount-based imbalance detection
    amount_concentration = df['amount'].rolling(window=5, min_periods=3).std() / \
                          df['amount'].rolling(window=20, min_periods=10).std()
    
    amount_volume_corr = df['amount'].pct_change().rolling(window=10, min_periods=5).corr(
        df['volume'].pct_change())
    
    amount_ma20 = df['amount'].rolling(window=20, min_periods=10).mean()
    large_order_imbalance = (df['amount'] > amount_ma20 * 2).rolling(window=5).sum()
    
    # Asymmetry analysis between volume and amount
    volume_momentum = df['volume'].pct_change(5)
    amount_momentum = df['amount'].pct_change(5)
    volume_amount_divergence = (volume_momentum - amount_momentum).abs()
    
    # Directional asymmetry
    volume_up_amount_down = ((volume_momentum > 0) & (amount_momentum < 0)).astype(int)
    volume_down_amount_up = ((volume_momentum < 0) & (amount_momentum > 0)).astype(int)
    directional_asymmetry = (volume_up_amount_down - volume_down_amount_up).rolling(window=5).mean()
    
    # Persistence of asymmetry patterns
    asymmetry_persistence = (volume_amount_divergence > volume_amount_divergence.rolling(window=10).mean()
                           ).rolling(window=5).sum()
    
    # Price-Volume-Amount Synchronization Analysis
    # Multi-dimensional momentum synchronization
    price_volume_corr = df['close'].pct_change().rolling(window=10, min_periods=5).corr(
        df['volume'].pct_change())
    price_amount_corr = df['close'].pct_change().rolling(window=10, min_periods=5).corr(
        df['amount'].pct_change())
    volume_amount_corr = df['volume'].pct_change().rolling(window=10, min_periods=5).corr(
        df['amount'].pct_change())
    
    # Synchronization divergence detection
    corr_breakdown = ((price_volume_corr - price_volume_corr.rolling(window=20).mean()).abs() +
                     (price_amount_corr - price_amount_corr.rolling(window=20).mean()).abs() +
                     (volume_amount_corr - volume_amount_corr.rolling(window=20).mean()).abs()) / 3
    
    synchronization_momentum = (price_volume_corr.diff(3) + price_amount_corr.diff(3) + 
                               volume_amount_corr.diff(3)) / 3
    
    # Multi-dimensional misalignment score
    misalignment_score = ((price_volume_corr - price_amount_corr).abs() +
                         (price_volume_corr - volume_amount_corr).abs() +
                         (price_amount_corr - volume_amount_corr).abs()) / 3
    
    # Asymmetric Breakout and Reversal Patterns
    price_breakout = df['close'] > df['high'].rolling(window=10).max().shift(1)
    volume_expansion = df['volume'] > df['volume'].rolling(window=20).mean() * 1.5
    amount_expansion = df['amount'] > df['amount'].rolling(window=20).mean() * 1.5
    
    volume_led_breakouts = (price_breakout & volume_expansion & 
                           (df['volume'].pct_change(3) > df['amount'].pct_change(3))).astype(int)
    amount_led_breakouts = (price_breakout & amount_expansion & 
                           (df['amount'].pct_change(3) > df['volume'].pct_change(3))).astype(int)
    
    # Divergence-based reversal patterns
    momentum_divergence_reversal = ((momentum_3d * momentum_21d < 0) & 
                                   (volume_amount_divergence > volume_amount_divergence.rolling(window=10).mean())).astype(int)
    
    price_amount_divergence_reversal = ((df['close'].pct_change(3) * df['amount'].pct_change(3) < 0) & 
                                       (price_amount_corr < 0)).astype(int)
    
    # Adaptive Signal Processing Based on Asymmetry Regimes
    asymmetry_intensity = (volume_amount_divergence.rolling(window=10).std() + 
                          directional_asymmetry.abs().rolling(window=10).std()) / 2
    
    high_asymmetry = asymmetry_intensity > asymmetry_intensity.rolling(window=20).quantile(0.7)
    low_asymmetry = asymmetry_intensity < asymmetry_intensity.rolling(window=20).quantile(0.3)
    
    # Composite Alpha Synthesis
    # Core momentum divergence component
    core_momentum = (divergence_intensity * volatility_weight - momentum_convergence)
    
    # Volume-amount asymmetry adjustment
    asymmetry_score = (volume_amount_divergence * directional_asymmetry.sign() * 
                      asymmetry_persistence / 5)
    
    # Synchronization filtering
    synchronization_filter = (price_volume_corr * price_amount_corr * volume_amount_corr).abs()
    synchronization_breakdown_penalty = corr_breakdown * (synchronization_momentum < 0)
    
    # Regime-based weighting
    high_asymmetry_weight = np.where(high_asymmetry, 1.5, 1.0)
    low_asymmetry_weight = np.where(low_asymmetry, 0.7, 1.0)
    regime_weight = high_asymmetry_weight * low_asymmetry_weight
    
    # Breakout/reversal pattern recognition
    breakout_signal = (volume_led_breakouts - amount_led_breakouts).rolling(window=5).mean()
    reversal_signal = (momentum_divergence_reversal + price_amount_divergence_reversal).rolling(window=5).mean()
    
    # Final alpha generation
    alpha = (core_momentum * regime_weight + 
             asymmetry_score * 0.3 + 
             breakout_signal * 0.2 + 
             reversal_signal * 0.1) * synchronization_filter - synchronization_breakdown_penalty * 0.5
    
    return alpha
