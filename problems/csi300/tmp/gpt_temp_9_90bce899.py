import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Price Fractal Divergence Factor
    # Calculate volume entropy
    volume_5d_std = df['volume'].rolling(window=5).std()
    volume_20d_std = df['volume'].rolling(window=20).std()
    volume_variance_ratio = volume_5d_std / volume_20d_std
    
    # 10-day volume autocorrelation
    volume_autocorr = df['volume'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    volume_entropy = volume_variance_ratio + volume_autocorr
    
    # Price fractal dimension - multi-timeframe range complexity
    def range_complexity(series, window):
        high_low_range = series.rolling(window=window).apply(
            lambda x: (x.max() - x.min()) / x.mean() if x.mean() != 0 else 0, raw=False
        )
        return high_low_range
    
    range_5d = range_complexity(df['close'], 5)
    range_10d = range_complexity(df['close'], 10)
    range_20d = range_complexity(df['close'], 20)
    price_fractal = (range_5d + range_10d + range_20d) / 3
    
    # Price path efficiency
    def price_efficiency(close_series, window):
        net_move = close_series.diff(window).abs()
        total_move = close_series.diff().abs().rolling(window=window).sum()
        efficiency = net_move / total_move.replace(0, np.nan)
        return efficiency.fillna(0)
    
    efficiency_5d = price_efficiency(df['close'], 5)
    efficiency_10d = price_efficiency(df['close'], 10)
    price_efficiency_combined = (efficiency_5d + efficiency_10d) / 2
    
    # 5-day momentum and volume surge
    momentum_5d = df['close'].pct_change(5)
    volume_surge = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Volume-Price Fractal Divergence Factor
    vp_divergence = (volume_entropy * price_fractal * price_efficiency_combined * 
                    momentum_5d * volume_surge)
    
    # Liquidity Absorption Momentum
    # Effective spread (using high-low range as proxy)
    effective_spread = (df['high'] - df['low']) / df['close']
    
    # Volume concentration (volume std relative to mean)
    volume_concentration = df['volume'].rolling(window=10).std() / df['volume'].rolling(window=10).mean()
    
    # Price impact (price change per unit volume)
    price_impact = df['close'].pct_change().abs() / df['volume'].replace(0, np.nan)
    price_impact = price_impact.fillna(0)
    
    # Price resilience (recovery after moves)
    def price_resilience(close_series, window=5):
        negative_moves = close_series.diff().where(close_series.diff() < 0, 0).abs()
        recovery = close_series.diff(window).where(close_series.diff(window) > 0, 0)
        resilience = recovery / negative_moves.rolling(window=window).sum().replace(0, np.nan)
        return resilience.fillna(0)
    
    resilience_5d = price_resilience(df['close'], 5)
    
    # 3-day breakouts
    breakout_3d = (df['close'] > df['high'].shift(3)).astype(int) - (df['close'] < df['low'].shift(3)).astype(int)
    
    # Trade size weighting (using amount/volume as proxy)
    avg_trade_size = df['amount'] / df['volume'].replace(0, np.nan)
    avg_trade_size = avg_trade_size.fillna(0)
    trade_size_weight = avg_trade_size / avg_trade_size.rolling(window=20).mean()
    
    # Liquidity Absorption Momentum
    liquidity_momentum = ((effective_spread + volume_concentration) * 
                         (price_impact + resilience_5d) * 
                         breakout_3d * trade_size_weight)
    
    # Volatility Regime Transition Detector
    # Volatility persistence (autocorrelation of volatility)
    volatility = df['close'].pct_change().rolling(window=5).std()
    vol_persistence = volatility.rolling(window=10).apply(
        lambda x: x.autocorr(lag=1), raw=False
    )
    
    # Transition probability (volatility regime changes)
    vol_regime = volatility > volatility.rolling(window=20).mean()
    regime_changes = vol_regime.astype(int).diff().abs()
    transition_prob = regime_changes.rolling(window=10).mean()
    
    # Shift magnitude (volatility change magnitude)
    vol_change_magnitude = volatility.pct_change().abs()
    
    # Momentum acceleration
    momentum_accel = momentum_5d.diff(3)
    
    # Volume pattern changes
    volume_pattern = df['volume'].pct_change().rolling(window=5).std()
    
    # Multi-timeframe confirmation
    vol_short = volatility.rolling(window=5).mean()
    vol_medium = volatility.rolling(window=10).mean()
    vol_long = volatility.rolling(window=20).mean()
    multi_timeframe_conf = (vol_short / vol_medium + vol_medium / vol_long) / 2
    
    # Volatility Regime Transition Detector
    vol_transition = ((vol_persistence + transition_prob + vol_change_magnitude) * 
                     (momentum_accel + volume_pattern) * multi_timeframe_conf)
    
    # Price Memory Decay Factor
    # Past return decay (exponential decay of past returns)
    def exponential_decay(series, halflife=5):
        weights = np.exp(-np.log(2) / halflife * np.arange(len(series))[::-1])
        return series.rolling(window=len(weights)).apply(
            lambda x: np.average(x, weights=weights[-len(x):]), raw=False
        )
    
    returns = df['close'].pct_change()
    decayed_returns = exponential_decay(returns, 10)
    
    # Support/resistance persistence
    def support_resistance_persistence(close_series, window=20):
        rolling_high = close_series.rolling(window=window).max()
        rolling_low = close_series.rolling(window=window).min()
        near_high = (close_series / rolling_high - 1).abs() < 0.02
        near_low = (close_series / rolling_low - 1).abs() < 0.02
        return (near_high | near_low).astype(int)
    
    sr_persistence = support_resistance_persistence(df['close'])
    
    # Pattern recognition (deviation from expected)
    expected_price = df['close'].rolling(window=10).mean()
    pattern_deviation = (df['close'] - expected_price) / expected_price
    
    # Volume confirmation
    volume_confirmation = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Price Memory Decay Factor
    price_memory = (decayed_returns * sr_persistence * pattern_deviation * 
                   volume_confirmation)
    
    # Momentum Quality Spectrum
    # Multi-timeframe consistency
    mom_3d = df['close'].pct_change(3)
    mom_5d = df['close'].pct_change(5)
    mom_10d = df['close'].pct_change(10)
    momentum_consistency = (mom_3d * mom_5d * mom_10d).apply(np.sign)
    
    # Acceleration patterns
    acceleration_3d = mom_3d.diff(2)
    acceleration_5d = mom_5d.diff(3)
    acceleration_pattern = (acceleration_3d + acceleration_5d) / 2
    
    # Risk-adjusted return (Sharpe-like ratio)
    returns_10d = df['close'].pct_change().rolling(window=10)
    risk_adjusted = returns_10d.mean() / returns_10d.std().replace(0, np.nan)
    risk_adjusted = risk_adjusted.fillna(0)
    
    # Volume efficiency (price change per unit volume)
    volume_efficiency = df['close'].pct_change().abs() / df['volume'].replace(0, np.nan)
    volume_efficiency = volume_efficiency.fillna(0)
    
    # Regime adjustment (using volatility regime)
    regime_adjustment = np.where(vol_regime, 1.5, 1.0)
    
    # Price-volume confirmation
    price_volume_conf = (df['close'].pct_change() * df['volume'].pct_change()).rolling(window=5).mean()
    
    # Momentum Quality Spectrum
    momentum_quality = ((momentum_consistency + acceleration_pattern) * 
                       (risk_adjusted + volume_efficiency) * 
                       regime_adjustment * price_volume_conf)
    
    # Combine all factors with equal weighting
    combined_factor = (vp_divergence + liquidity_momentum + vol_transition + 
                      price_memory + momentum_quality) / 5
    
    # Normalize the final factor
    factor_series = (combined_factor - combined_factor.rolling(window=20).mean()) / combined_factor.rolling(window=20).std()
    
    return factor_series.fillna(0)
