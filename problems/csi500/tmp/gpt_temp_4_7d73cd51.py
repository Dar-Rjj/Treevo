import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Divergence Alpha Factor
    
    This factor combines price and volume momentum divergences with regime detection
    based on amount data to create adaptive trading signals.
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required lookback periods
    min_periods = 20
    
    # Calculate price momentum components
    price_momentum_5 = df['close'] / df['close'].shift(5) - 1
    price_momentum_10 = df['close'] / df['close'].shift(10) - 1
    price_momentum_20 = df['close'] / df['close'].shift(20) - 1
    
    # Calculate volume momentum components
    volume_momentum_5 = df['volume'] / df['volume'].shift(5) - 1
    volume_momentum_10 = df['volume'] / df['volume'].shift(10) - 1
    volume_momentum_20 = df['volume'] / df['volume'].shift(20) - 1
    
    # Apply exponential smoothing (alpha=0.3)
    alpha = 0.3
    
    # Smoothed price momentum
    smooth_price_5 = price_momentum_5.ewm(alpha=alpha, adjust=False).mean()
    smooth_price_10 = price_momentum_10.ewm(alpha=alpha, adjust=False).mean()
    smooth_price_20 = price_momentum_20.ewm(alpha=alpha, adjust=False).mean()
    
    # Smoothed volume momentum
    smooth_volume_5 = volume_momentum_5.ewm(alpha=alpha, adjust=False).mean()
    smooth_volume_10 = volume_momentum_10.ewm(alpha=alpha, adjust=False).mean()
    smooth_volume_20 = volume_momentum_20.ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate momentum divergence for each timeframe
    divergence_5 = smooth_price_5 - smooth_volume_5
    divergence_10 = smooth_price_10 - smooth_volume_10
    divergence_20 = smooth_price_20 - smooth_volume_20
    
    # Average divergence across timeframes
    raw_divergence = (divergence_5 + divergence_10 + divergence_20) / 3
    
    # Regime detection using amount data
    amount_20d_avg = df['amount'].rolling(window=20, min_periods=min_periods).mean()
    amount_momentum = df['amount'] / df['amount'].shift(20) - 1
    
    # Volatility assessment
    daily_range = (df['high'] - df['low']) / df['close']
    range_20d_avg = daily_range.rolling(window=20, min_periods=min_periods).mean()
    
    # Classify regimes based on amount momentum thresholds
    high_participation = amount_momentum > 0.1
    low_participation = amount_momentum < -0.05
    volatile_regime = daily_range > range_20d_avg * 1.2
    
    # Cross-sectional processing (within-day ranking)
    def cross_sectional_processing(date_data):
        if len(date_data) < 10:  # Minimum universe size
            return pd.Series(index=date_data.index, data=0.0)
        
        # Rank by divergence magnitude
        divergence_rank = date_data['raw_divergence'].abs().rank(pct=True)
        
        # Identify significant divergence (top 30%)
        significant_divergence = divergence_rank > 0.7
        
        # Base signal from divergence direction
        base_signal = np.sign(date_data['raw_divergence'])
        
        # Regime-adaptive weighting
        final_signal = base_signal.copy()
        
        # High participation regimes: emphasize recent divergence
        high_participation_mask = date_data['high_participation']
        if high_participation_mask.any():
            recent_weight = 1 + date_data.loc[high_participation_mask, 'amount_momentum'].abs()
            final_signal.loc[high_participation_mask] *= recent_weight
        
        # Low participation regimes: emphasize persistent divergence
        low_participation_mask = date_data['low_participation']
        if low_participation_mask.any():
            # Use longer-term divergence for persistence
            persistent_weight = 1 + date_data.loc[low_participation_mask, 'divergence_20'].abs()
            final_signal.loc[low_participation_mask] *= persistent_weight
        
        # Volatile regimes: increase volume momentum weight
        volatile_mask = date_data['volatile_regime']
        if volatile_mask.any():
            # In volatile markets, give more weight to volume confirmation
            volume_weight = 1 + date_data.loc[volatile_mask, 'smooth_volume_20'].abs()
            final_signal.loc[volatile_mask] *= volume_weight
        
        # Apply significant divergence filter
        final_signal[~significant_divergence] *= 0.3  # Reduce signal for insignificant divergence
        
        return final_signal
    
    # Prepare data for cross-sectional processing
    processing_data = pd.DataFrame({
        'raw_divergence': raw_divergence,
        'divergence_5': divergence_5,
        'divergence_10': divergence_10,
        'divergence_20': divergence_20,
        'amount_momentum': amount_momentum,
        'smooth_volume_20': smooth_volume_20,
        'high_participation': high_participation,
        'low_participation': low_participation,
        'volatile_regime': volatile_regime
    })
    
    # Apply cross-sectional processing by date
    for date in df.index:
        date_data = processing_data.loc[date:date]
        if len(date_data) == 1:  # Single stock case
            # For single stock, use simplified approach
            base_signal = np.sign(date_data['raw_divergence'].iloc[0])
            
            # Apply regime adjustments
            if date_data['high_participation'].iloc[0]:
                base_signal *= (1 + abs(date_data['amount_momentum'].iloc[0]))
            elif date_data['low_participation'].iloc[0]:
                base_signal *= (1 + abs(date_data['divergence_20'].iloc[0]))
            if date_data['volatile_regime'].iloc[0]:
                base_signal *= (1 + abs(date_data['smooth_volume_20'].iloc[0]))
                
            result.loc[date] = base_signal
        else:
            # For multiple stocks, use cross-sectional ranking
            signals = cross_sectional_processing(date_data)
            result.loc[date] = signals.iloc[0] if len(signals) > 0 else 0.0
    
    # Handle NaN values
    result = result.fillna(0)
    
    return result
