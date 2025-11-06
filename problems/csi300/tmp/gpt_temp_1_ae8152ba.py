import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Liquidity Memory Asymmetry Alpha
    Combines order flow dynamics, price impact asymmetry, and liquidity memory structure
    to predict future returns based on market microstructure patterns.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols_required = ['open', 'high', 'low', 'close', 'amount', 'volume']
    for col in cols_required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate daily returns and price changes
    returns = df['close'].pct_change()
    price_range = (df['high'] - df['low']) / df['close']
    
    # 1. Measure Bid-Ask Flow Imbalance
    # Compute order flow autocorrelation at multiple lags
    volume_changes = df['volume'].pct_change()
    autocorr_lag1 = volume_changes.rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 5 else np.nan, raw=False
    )
    autocorr_lag2 = volume_changes.rolling(window=15, min_periods=8).apply(
        lambda x: x.autocorr(lag=2) if len(x) >= 8 else np.nan, raw=False
    )
    
    # Calculate bid-ask depth ratio stability (proxy using volume/amount ratio volatility)
    depth_ratio = df['volume'] / (df['amount'] + 1e-10)
    depth_stability = 1.0 / (depth_ratio.rolling(window=10).std() + 1e-10)
    
    flow_imbalance = autocorr_lag1 - autocorr_lag2 + depth_stability
    
    # 2. Quantify Price Impact Asymmetry
    # Compare buy vs sell trade price impact (proxy using high-close vs low-close asymmetry)
    buy_impact = (df['high'] - df['close']) / df['close']
    sell_impact = (df['close'] - df['low']) / df['close']
    impact_asymmetry = (buy_impact.rolling(window=5).mean() - 
                       sell_impact.rolling(window=5).mean())
    
    # Analyze impact decay rate differences
    impact_persistence_buy = buy_impact.rolling(window=5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) >= 5 and not np.isnan(x).any() else np.nan, 
        raw=False
    )
    impact_persistence_sell = sell_impact.rolling(window=5).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) >= 5 and not np.isnan(x).any() else np.nan, 
        raw=False
    )
    decay_asymmetry = impact_persistence_buy - impact_persistence_sell
    
    price_impact_asymmetry = impact_asymmetry * decay_asymmetry
    
    # 3. Assess Liquidity Memory Structure
    # Calculate depth restoration speed after large trades
    large_trade_threshold = df['volume'].rolling(window=20).quantile(0.8)
    is_large_trade = df['volume'] > large_trade_threshold
    
    # Restoration speed: how quickly volume normalizes after large trades
    restoration_speed = pd.Series(index=df.index, dtype=float)
    for i in range(2, len(df)):
        if is_large_trade.iloc[i-1]:
            current_vol = df['volume'].iloc[i]
            prev_vol = df['volume'].iloc[i-1]
            normal_vol = df['volume'].iloc[max(0, i-10):i].median()
            if prev_vol > normal_vol and normal_vol > 0:
                speed = (prev_vol - current_vol) / (prev_vol - normal_vol)
                restoration_speed.iloc[i] = np.clip(speed, -1, 1)
    
    restoration_speed = restoration_speed.rolling(window=10, min_periods=3).mean()
    
    # Measure permanent vs temporary impact ratio
    temp_impact = (df['high'] - df['low']) / df['close']  # Temporary impact
    perm_impact = abs(returns)  # Permanent impact
    impact_ratio = perm_impact / (temp_impact + 1e-10)
    impact_ratio_smooth = impact_ratio.rolling(window=10, min_periods=5).mean()
    
    liquidity_memory = restoration_speed * impact_ratio_smooth
    
    # 4. Generate Alpha Signal
    # Combine flow imbalance with impact asymmetry
    core_signal = flow_imbalance * price_impact_asymmetry
    
    # Incorporate memory breakdown timing patterns
    memory_timing = liquidity_memory.rolling(window=5).std()
    
    # Final alpha combining all components
    alpha = (core_signal - core_signal.rolling(window=20).mean()) * memory_timing
    
    # Normalize and handle edge cases
    alpha_rank = alpha.rank(pct=True)
    alpha_zscore = (alpha - alpha.rolling(window=50).mean()) / (alpha.rolling(window=50).std() + 1e-10)
    
    # Final signal with robustness adjustments
    final_signal = 0.6 * alpha_rank + 0.4 * alpha_zscore
    
    # Fill initial NaN values
    result = final_signal.fillna(0)
    
    return result
