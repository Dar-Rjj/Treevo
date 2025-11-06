import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining momentum acceleration, volatility-adjusted trend persistence,
    price-volume divergence, and regime-based return asymmetry.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    factor = pd.Series(index=data.index, dtype=float)
    
    # 1. Momentum Acceleration Factor
    # Calculate short-term momentum (5-day)
    mom_5d = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Calculate medium-term momentum (20-day)
    mom_20d = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    # Calculate momentum acceleration
    mom_accel = mom_5d - mom_20d
    
    # Calculate volume ratio
    vol_5d_avg = data['volume'].rolling(window=5).mean()
    vol_20d_avg = data['volume'].rolling(window=20).mean()
    vol_ratio = vol_5d_avg / vol_20d_avg
    
    # Momentum acceleration factor
    momentum_factor = mom_accel * vol_ratio
    
    # 2. Volatility-Adjusted Trend Persistence
    # Calculate true range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate absolute return
    abs_return = abs(data['close'] - data['close'].shift(1))
    
    # Calculate efficiency
    efficiency = abs_return / true_range
    
    # Calculate trend persistence (consecutive same direction days)
    returns = data['close'].pct_change()
    direction = np.sign(returns)
    
    def calculate_trend_persistence(direction_series, window=10):
        persistence = pd.Series(index=direction_series.index, dtype=float)
        for i in range(window, len(direction_series)):
            window_directions = direction_series.iloc[i-window:i]
            if len(window_directions) > 0:
                current_dir = direction_series.iloc[i-1]
                consecutive = 0
                for j in range(len(window_directions)-1, -1, -1):
                    if window_directions.iloc[j] == current_dir and window_directions.iloc[j] != 0:
                        consecutive += 1
                    else:
                        break
                persistence.iloc[i] = consecutive
        return persistence
    
    trend_persistence = calculate_trend_persistence(direction)
    
    # Calculate relative volatility
    atr_5d = true_range.rolling(window=5).mean()
    atr_20d = true_range.rolling(window=20).mean()
    rel_vol = atr_5d / atr_20d
    
    # Volatility-adjusted trend persistence factor
    trend_factor = (trend_persistence * efficiency) / rel_vol
    
    # 3. Price-Volume Divergence Momentum
    # Price momentum component (10-day)
    price_mom = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Momentum consistency (positive returns in last 5 days)
    def calculate_momentum_consistency(returns_series, window=5):
        consistency = pd.Series(index=returns_series.index, dtype=float)
        for i in range(window, len(returns_series)):
            window_returns = returns_series.iloc[i-window:i]
            positive_count = (window_returns > 0).sum()
            consistency.iloc[i] = positive_count / window
        return consistency
    
    mom_consistency = calculate_momentum_consistency(returns)
    
    # Volume divergence component
    vol_10d_avg = data['volume'].rolling(window=10).mean()
    vol_mom = (data['volume'] / vol_5d_avg) - (data['volume'] / vol_10d_avg)
    
    # Price-volume divergence factor
    divergence_factor = price_mom * mom_consistency * (1 - vol_mom)
    
    # 4. Regime-Based Return Asymmetry
    # Calculate volume z-score
    vol_mean_20d = data['volume'].rolling(window=20).mean()
    vol_std_20d = data['volume'].rolling(window=20).std()
    vol_zscore = (data['volume'] - vol_mean_20d) / vol_std_20d
    
    # Calculate 5-day forward returns
    forward_returns = data['close'].pct_change(5).shift(-5)
    
    def calculate_regime_asymmetry(vol_zscore, forward_returns, window=60):
        asymmetry = pd.Series(index=vol_zscore.index, dtype=float)
        for i in range(window, len(vol_zscore)):
            window_zscore = vol_zscore.iloc[i-window:i]
            window_returns = forward_returns.iloc[i-window:i]
            
            high_vol_mask = window_zscore > 1
            low_vol_mask = window_zscore < -1
            
            high_vol_returns = window_returns[high_vol_mask]
            low_vol_returns = window_returns[low_vol_mask]
            
            if len(high_vol_returns) > 0 and len(low_vol_returns) > 0:
                asymmetry.iloc[i] = high_vol_returns.mean() - low_vol_returns.mean()
            else:
                asymmetry.iloc[i] = 0
        return asymmetry
    
    regime_factor = calculate_regime_asymmetry(vol_zscore, forward_returns)
    
    # Combine all factors with equal weights
    factor = (momentum_factor.fillna(0) + 
              trend_factor.fillna(0) + 
              divergence_factor.fillna(0) + 
              regime_factor.fillna(0)) / 4
    
    return factor
