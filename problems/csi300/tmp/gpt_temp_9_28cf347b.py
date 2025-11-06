import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Decay Adjusted by Volatility Regime
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Short-term momentum (5-day)
    mom_short = data['close'].pct_change(periods=5)
    
    # Medium-term momentum (21-day)
    mom_medium = data['close'].pct_change(periods=21)
    
    # Apply exponential decay to momentum
    decay_weights = np.exp(-np.arange(21) / 7)  # 7-day decay half-life
    decay_weights = decay_weights / decay_weights.sum()
    
    # Create momentum decay series
    mom_decay = pd.Series(index=data.index, dtype=float)
    for i in range(21, len(data)):
        window_mom = mom_medium.iloc[i-20:i+1]  # 21-day window
        mom_decay.iloc[i] = (window_mom * decay_weights).sum()
    
    # Volatility regime assessment
    vol_21d = returns.rolling(window=21).std()
    vol_median = vol_21d.rolling(window=63).median()  # 3-month median
    
    # Volatility regime classification
    high_vol_regime = vol_21d > vol_median
    vol_sensitivity = pd.Series(1.0, index=data.index)
    vol_sensitivity[high_vol_regime] = 0.7  # Reduce impact in high vol
    vol_sensitivity[~high_vol_regime] = 1.3  # Amplify in low vol
    
    # Adjusted momentum
    adj_momentum = mom_decay * vol_sensitivity
    
    # Volume-Price Divergence Convergence
    # Volume trend
    vol_ma_10 = data['volume'].rolling(window=10).mean()
    vol_deviation = (data['volume'] - vol_ma_10) / vol_ma_10
    
    # Price trend
    price_ma_10 = data['close'].rolling(window=10).mean()
    price_deviation = (data['close'] - price_ma_10) / price_ma_10
    
    # Divergence detection
    volume_increasing = vol_deviation > 0
    price_decreasing = price_deviation < 0
    volume_decreasing = vol_deviation < 0
    price_increasing = price_deviation > 0
    
    bearish_divergence = volume_increasing & price_decreasing
    bullish_divergence = volume_decreasing & price_increasing
    
    # Convergence strength (simplified)
    divergence_strength = pd.Series(0.0, index=data.index)
    divergence_strength[bearish_divergence] = -abs(vol_deviation * price_deviation)
    divergence_strength[bullish_divergence] = abs(vol_deviation * price_deviation)
    
    # Intraday Range Efficiency
    # True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Effective movement
    effective_move = abs(data['close'] - data['open'])
    
    # Efficiency ratio
    efficiency_ratio = effective_move / true_range
    efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Liquidity-Adjusted Reversal
    # Identify extreme moves (top/bottom 10%)
    ret_5d = data['close'].pct_change(5)
    extreme_positive = ret_5d > ret_5d.rolling(window=63).quantile(0.9)
    extreme_negative = ret_5d < ret_5d.rolling(window=63).quantile(0.1)
    
    # Liquidity assessment
    high_low_range = (data['high'] - data['low']) / data['close']
    volume_ma = data['volume'].rolling(window=10).mean()
    volume_ratio = data['volume'] / volume_ma
    
    # Liquidity classification
    tight_liquidity = (high_low_range < high_low_range.rolling(window=21).median()) & (volume_ratio > 1)
    loose_liquidity = (high_low_range > high_low_range.rolling(window=21).median()) | (volume_ratio < 0.8)
    
    # Reversal prediction
    reversal_strength = pd.Series(0.0, index=data.index)
    reversal_strength[extreme_positive & loose_liquidity] = -1.0  # Strong negative reversal
    reversal_strength[extreme_negative & loose_liquidity] = 1.0   # Strong positive reversal
    reversal_strength[extreme_positive & tight_liquidity] = -0.3  # Weak negative reversal
    reversal_strength[extreme_negative & tight_liquidity] = 0.3   # Weak positive reversal
    
    # Market Microstructure Imbalance
    # Opening auction pressure (simplified)
    open_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    early_range = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / data['close']
    opening_pressure = open_gap / (early_range + 1e-8)
    
    # Closing momentum
    close_ma_last_hour = data['close'].rolling(window=3).mean()  # Simplified last hour
    close_momentum = (data['close'] - close_ma_last_hour) / close_ma_last_hour
    
    # Combine microstructure signals
    microstructure_signal = opening_pressure.rolling(window=5).mean() + close_momentum
    
    # Final factor combination (equal weighting for demonstration)
    factor = (
        adj_momentum.fillna(0) * 0.25 +
        divergence_strength.fillna(0) * 0.25 +
        efficiency_ratio.fillna(0) * 0.25 +
        reversal_strength.fillna(0) * 0.15 +
        microstructure_signal.fillna(0) * 0.10
    )
    
    return factor
