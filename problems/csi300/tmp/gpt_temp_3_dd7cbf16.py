import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Regime Detection Component
    # Volatility Regime Classification
    vol_short = returns.rolling(window=5).std()
    vol_medium = returns.rolling(window=10).std()
    vol_long = returns.rolling(window=20).std()
    
    short_term_vol_state = vol_short / vol_long
    medium_term_vol_state = vol_medium / vol_long
    
    # Volatility regime classification (1 for high vol, -1 for low vol, 0 for neutral)
    vol_regime = np.where(short_term_vol_state > 1.2, 1, 
                         np.where(short_term_vol_state < 0.8, -1, 0))
    vol_regime = pd.Series(vol_regime, index=df.index)
    
    # Regime persistence
    regime_persistence = vol_regime.rolling(window=10).apply(
        lambda x: len(set(x)) == 1 if not x.isna().all() else np.nan, raw=False
    )
    
    # Trend Regime Classification
    price_acceleration = (df['close'] - df['close'].shift(5)) - (df['close'].shift(5) - df['close'].shift(10))
    volume_change = df['volume'].pct_change(periods=5)
    volume_trend_alignment = np.sign(price_acceleration) * volume_change
    
    regime_strength = np.abs(volume_trend_alignment) / df['volume'].rolling(window=5).mean()
    
    # Combined Regime Score
    combined_regime_score = vol_regime * regime_strength
    
    # Adaptive Momentum Component
    # Regime-Weighted Returns
    high_vol_returns = returns.rolling(window=5).apply(
        lambda x: x[vol_regime.loc[x.index] > 0].sum() if (vol_regime.loc[x.index] > 0).any() else 0, 
        raw=False
    )
    
    low_vol_returns = returns.rolling(window=10).apply(
        lambda x: x[vol_regime.loc[x.index] < 0].sum() if (vol_regime.loc[x.index] < 0).any() else 0, 
        raw=False
    )
    
    regime_adaptive_return = (high_vol_returns * regime_persistence) + (low_vol_returns * (1 - regime_persistence))
    
    # Volume-Confirmed Momentum
    def calc_volume_pressure(window_returns, window_volume, condition):
        mask = condition(window_returns)
        return window_volume[mask].sum() if mask.any() else 0
    
    upside_volume_pressure = returns.rolling(window=5).apply(
        lambda x: calc_volume_pressure(x, df['volume'].loc[x.index], lambda r: r > 0), 
        raw=False
    )
    
    downside_volume_pressure = returns.rolling(window=5).apply(
        lambda x: calc_volume_pressure(x, df['volume'].loc[x.index], lambda r: r < 0), 
        raw=False
    )
    
    total_5day_volume = df['volume'].rolling(window=5).sum()
    volume_momentum_bias = (upside_volume_pressure - downside_volume_pressure) / total_5day_volume
    
    # Adaptive Momentum Score
    adaptive_momentum_score = regime_adaptive_return * volume_momentum_bias
    
    # Price Efficiency Component
    # Path-Dependent Efficiency
    actual_price_change = df['close'] - df['close'].shift(5)
    min_path_distance = returns.abs().rolling(window=5).sum()
    efficiency_ratio = actual_price_change / min_path_distance
    
    # Gap Efficiency Analysis
    prev_close = df['close'].shift(1)
    prev_range = df['high'].shift(1) - df['low'].shift(1)
    overnight_gap_momentum = (df['open'] - prev_close) / prev_range
    
    daily_range = df['high'] - df['low']
    intraday_gap_momentum = (df['close'] - df['open']) / daily_range
    
    gap_efficiency = overnight_gap_momentum * intraday_gap_momentum
    
    # Combined Efficiency
    combined_efficiency = efficiency_ratio * gap_efficiency
    
    # Signal Integration
    multi_regime_factor = adaptive_momentum_score * combined_efficiency * combined_regime_score
    
    # Dynamic Scaling
    avg_abs_return = returns.abs().rolling(window=20).mean()
    final_factor = multi_regime_factor / avg_abs_return
    
    return final_factor
