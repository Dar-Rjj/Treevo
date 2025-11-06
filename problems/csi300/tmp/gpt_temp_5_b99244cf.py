import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Momentum Memory Divergence Factor
    Combines multi-timeframe momentum with volatility regime memory and divergence analysis
    """
    # Calculate basic price data
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # True Range calculation
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Multi-Timeframe Momentum Quality Assessment
    # Short-term momentum (5-day)
    mom_5d = close / close.shift(5) - 1
    
    # Medium-term momentum (20-day)
    mom_20d = close / close.shift(20) - 1
    
    # Momentum Consistency: Sign agreement between 5-day and 20-day momentum
    mom_consistency = ((mom_5d > 0) & (mom_20d > 0)) | ((mom_5d < 0) & (mom_20d < 0))
    mom_consistency_score = mom_consistency.astype(float)
    
    # Momentum Persistence: Count of same-sign momentum days in past 5 sessions
    mom_daily = close.pct_change()
    mom_sign = np.sign(mom_daily)
    mom_persistence = mom_sign.rolling(window=5).apply(
        lambda x: len(set(x)) == 1 if not x.isnull().any() else np.nan, raw=False
    )
    
    # Volatility Regime Memory Detection
    # Current Volatility State: 5-day average of TrueRange/Close
    vol_state = (true_range / close).rolling(window=5).mean()
    
    # Historical Volatility Pattern Matching (20-day lookback)
    rolling_5d_vol = (true_range / close).rolling(window=5).mean()
    
    def find_similar_regimes(current_vol, lookback=20):
        """Find similar volatility periods in recent history"""
        if pd.isna(current_vol):
            return np.nan
        
        historical_vol = rolling_5d_vol.shift(1).rolling(window=lookback).apply(
            lambda x: np.nan if x.isnull().any() else np.mean(np.abs(x - current_vol)), 
            raw=False
        )
        
        # Inverse of average deviation (higher = more similar)
        similarity = 1 / (1 + historical_vol)
        return similarity
    
    regime_similarity = rolling_5d_vol.apply(lambda x: find_similar_regimes(x))
    
    # Price behavior persistence across matched regimes
    def regime_persistence_score(current_idx, vol_series, price_series, lookback=20):
        if current_idx < lookback:
            return np.nan
        
        current_vol = vol_series.iloc[current_idx]
        if pd.isna(current_vol):
            return np.nan
        
        # Find similar volatility periods in lookback
        similar_periods = []
        for i in range(1, lookback + 1):
            hist_vol = vol_series.iloc[current_idx - i]
            if not pd.isna(hist_vol) and abs(hist_vol - current_vol) / current_vol < 0.2:
                similar_periods.append(i)
        
        if len(similar_periods) == 0:
            return 0.5  # Neutral score
        
        # Calculate price persistence across similar regimes
        current_return = price_series.iloc[current_idx] / price_series.iloc[current_idx - 1] - 1
        persistence_scores = []
        
        for period in similar_periods:
            hist_return = price_series.iloc[current_idx - period] / price_series.iloc[current_idx - period - 1] - 1
            if np.sign(current_return) == np.sign(hist_return):
                persistence_scores.append(1.0)
            else:
                persistence_scores.append(0.0)
        
        return np.mean(persistence_scores) if persistence_scores else 0.5
    
    # Calculate regime persistence
    regime_persistence = pd.Series(index=close.index, dtype=float)
    for idx in range(len(close)):
        if idx >= 20:
            regime_persistence.iloc[idx] = regime_persistence_score(idx, rolling_5d_vol, close)
    
    # Regime Transition Quality
    # Volatility change magnitude
    vol_20d_avg = (true_range / close).rolling(window=20).mean()
    vol_change_magnitude = rolling_5d_vol / vol_20d_avg
    
    # Volume confirmation during regime shifts
    vol_5d_avg = volume.rolling(window=5).mean()
    volume_confirmation = volume / vol_5d_avg
    
    # Momentum-Volatility Divergence Analysis
    # Calculate 5-day momentum volatility (std of daily returns)
    mom_vol_5d = mom_daily.rolling(window=5).std()
    
    # Regime-adaptive divergence scoring
    def divergence_scoring(mom_short, mom_medium, mom_consistency, mom_persistence, 
                          current_vol, avg_vol, mom_vol):
        """Calculate divergence score based on volatility regime"""
        if pd.isna(current_vol) or pd.isna(avg_vol):
            return np.nan
        
        # High volatility regime (current vol > historical average)
        if current_vol > avg_vol * 1.1:
            # Expect higher momentum volatility in high vol regimes
            expected_mom_vol = avg_vol * 1.5
            if mom_vol > expected_mom_vol:
                divergence = (mom_short / mom_vol) * mom_consistency
            else:
                # Positive divergence: strong momentum with low volatility
                divergence = mom_short * 2 * mom_consistency
        else:
            # Low volatility regime - expect stable momentum persistence
            divergence = mom_medium * mom_persistence
        
        return divergence
    
    # Calculate divergence scores
    divergence_scores = pd.Series(index=close.index, dtype=float)
    for idx in range(len(close)):
        if idx >= 20:
            score = divergence_scoring(
                mom_short=mom_5d.iloc[idx],
                mom_medium=mom_20d.iloc[idx],
                mom_consistency=mom_consistency_score.iloc[idx],
                mom_persistence=mom_persistence.iloc[idx],
                current_vol=rolling_5d_vol.iloc[idx],
                avg_vol=vol_20d_avg.iloc[idx],
                mom_vol=mom_vol_5d.iloc[idx]
            )
            divergence_scores.iloc[idx] = score
    
    # Memory-Weighted Factor Synthesis
    # Historical regime similarity weighting
    memory_weight = regime_similarity * regime_persistence
    
    # Volume entropy adjustment (measure of volume concentration)
    volume_entropy = volume.rolling(window=5).apply(
        lambda x: -np.sum((x / x.sum()) * np.log(x / x.sum())) if x.sum() > 0 else 0,
        raw=False
    )
    volume_entropy_norm = (volume_entropy - volume_entropy.rolling(20).min()) / \
                         (volume_entropy.rolling(20).max() - volume_entropy.rolling(20).min() + 1e-8)
    
    # Volume-confirmed divergence integration
    volume_confirmed_divergence = divergence_scores * volume_confirmation
    
    # Final factor: Regime-adaptive, memory-weighted momentum divergence score
    final_factor = (volume_confirmed_divergence * memory_weight * 
                   (1 - volume_entropy_norm)).fillna(0)
    
    return final_factor
