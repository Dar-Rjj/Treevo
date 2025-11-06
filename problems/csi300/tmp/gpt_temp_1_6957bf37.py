import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Liquidity Flow Asymmetry with Price Impact Memory factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Parameters
    lookback_short = 5
    lookback_medium = 20
    lookback_long = 60
    volume_threshold = 0.8
    
    # Calculate basic price and volume metrics
    df['returns'] = df['close'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=lookback_medium).mean()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['close_open_range'] = (df['close'] - df['open']) / df['open']
    
    # Estimate bid-ask imbalance proxy using intraday patterns
    df['intraday_direction'] = np.where(df['close'] > df['open'], 1, 
                                       np.where(df['close'] < df['open'], -1, 0))
    
    # Calculate directional order flow autocorrelation
    def compute_flow_autocorr(series, lag):
        if len(series) < lag + 10:
            return np.nan
        return series.autocorr(lag=lag)
    
    # Price impact asymmetry measures
    df['large_buy_impact'] = np.where(
        (df['close'] > df['open']) & (df['volume'] > df['volume_ma']),
        df['high_low_range'] * df['close_open_range'].abs(), 0
    )
    
    df['large_sell_impact'] = np.where(
        (df['close'] < df['open']) & (df['volume'] > df['volume_ma']),
        df['high_low_range'] * df['close_open_range'].abs(), 0
    )
    
    # Rolling price impact differential
    df['buy_impact_ma'] = df['large_buy_impact'].rolling(window=lookback_short).mean()
    df['sell_impact_ma'] = df['large_sell_impact'].rolling(window=lookback_short).mean()
    df['impact_asymmetry'] = (df['buy_impact_ma'] - df['sell_impact_ma']) / (
        df['buy_impact_ma'] + df['sell_impact_ma'] + 1e-8)
    
    # Liquidity restoration speed (price range normalization after large moves)
    df['large_move'] = df['returns'].abs() > df['returns'].rolling(window=lookback_medium).std()
    df['post_move_range'] = df['high_low_range'].shift(1)
    df['restoration_speed'] = np.where(
        df['large_move'],
        df['high_low_range'] / (df['post_move_range'] + 1e-8),
        np.nan
    )
    
    # Price impact memory decay (autocorrelation of impact asymmetry)
    df['impact_asymmetry_returns'] = df['impact_asymmetry'] * df['returns']
    impact_memory = []
    for i in range(len(df)):
        if i < lookback_medium:
            impact_memory.append(np.nan)
            continue
        window_data = df['impact_asymmetry_returns'].iloc[i-lookback_medium+1:i+1]
        if window_data.notna().sum() >= lookback_short:
            decay_rate = 1 - abs(window_data.autocorr(lag=1))
            impact_memory.append(decay_rate)
        else:
            impact_memory.append(np.nan)
    df['impact_memory_decay'] = impact_memory
    
    # Liquidity regime detection using volume and range patterns
    df['volatility_regime'] = df['high_low_range'].rolling(window=lookback_medium).rank(pct=True)
    df['volume_regime'] = df['volume'].rolling(window=lookback_medium).rank(pct=True)
    
    # Regime stability measure
    df['regime_stability'] = (
        df['volatility_regime'].rolling(window=lookback_short).std() + 
        df['volume_regime'].rolling(window=lookback_short).std()
    )
    
    # Asymmetric memory collapse detection
    df['memory_breakdown'] = np.where(
        (df['impact_memory_decay'] > df['impact_memory_decay'].rolling(window=lookback_long).quantile(0.8)) &
        (df['regime_stability'] > df['regime_stability'].rolling(window=lookback_long).quantile(0.8)),
        1, 0
    )
    
    # Liquidity exhaustion patterns
    df['liquidity_exhaustion'] = np.where(
        (df['volume'] > df['volume'].rolling(window=lookback_medium).quantile(volume_threshold)) &
        (df['impact_asymmetry'].abs() > df['impact_asymmetry'].abs().rolling(window=lookback_medium).quantile(0.7)),
        df['intraday_direction'] * df['impact_asymmetry'],
        0
    )
    
    # Final alpha factor construction
    for i in range(lookback_long, len(df)):
        if i < lookback_long:
            result.iloc[i] = 0
            continue
            
        # Current window data
        window_data = df.iloc[i-lookback_medium+1:i+1]
        
        # Liquidity memory score components
        flow_persistence = window_data['intraday_direction'].autocorr(lag=1) if len(window_data) > 1 else 0
        impact_asymmetry_stable = window_data['impact_asymmetry'].std()
        memory_stability = 1 - window_data['impact_memory_decay'].mean() if not pd.isna(window_data['impact_memory_decay'].mean()) else 0
        
        # Combine components with weights
        memory_score = (
            0.4 * (flow_persistence if not pd.isna(flow_persistence) else 0) +
            0.3 * (1 / (impact_asymmetry_stable + 1e-8)) +
            0.3 * memory_stability
        )
        
        # Asymmetric flow timing signal
        recent_memory_breakdown = window_data['memory_breakdown'].iloc[-5:].sum() if len(window_data) >= 5 else 0
        recent_exhaustion = window_data['liquidity_exhaustion'].iloc[-3:].mean() if len(window_data) >= 3 else 0
        
        timing_signal = (
            -0.6 * recent_memory_breakdown +
            0.4 * recent_exhaustion
        )
        
        # Final alpha value
        result.iloc[i] = memory_score * timing_signal
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    # Normalize the final factor
    if len(result) > lookback_long:
        rolling_mean = result.rolling(window=lookback_long, min_periods=1).mean()
        rolling_std = result.rolling(window=lookback_long, min_periods=1).std()
        result = (result - rolling_mean) / (rolling_std + 1e-8)
    
    return result
