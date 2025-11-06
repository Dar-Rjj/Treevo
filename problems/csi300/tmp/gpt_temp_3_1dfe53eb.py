import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    # Calculate true range
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Calculate rolling metrics for regime classification
    data['avg_true_range_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['return_volatility_20d'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    
    # Classify volatility regime
    data['high_vol_regime'] = data['avg_true_range_10d'] > data['return_volatility_20d']
    
    # Regime-Specific Momentum Calculation
    # High volatility regime momentum
    data['momentum_2d'] = data['close'] / data['close'].shift(2) - 1
    data['momentum_5d_rolling'] = data['close'].pct_change(periods=5).rolling(window=5, min_periods=3).apply(lambda x: np.var(x) if len(x) >= 3 else np.nan)
    data['high_vol_momentum'] = data['momentum_2d'] / (1 + data['momentum_5d_rolling'])
    
    # Low volatility regime momentum
    data['momentum_15d'] = data['close'] / data['close'].shift(15) - 1
    data['momentum_acceleration'] = data['momentum_15d'] - data['momentum_15d'].shift(3)
    data['momentum_consistency'] = data['momentum_15d'].rolling(window=10, min_periods=5).apply(lambda x: np.mean(np.abs(x)) if len(x) >= 5 else np.nan)
    data['low_vol_momentum'] = data['momentum_15d'] * (1 + data['momentum_acceleration']) / (1 + data['momentum_consistency'])
    
    # Combine regime-specific momentum
    data['regime_momentum'] = np.where(
        data['high_vol_regime'],
        data['high_vol_momentum'],
        data['low_vol_momentum']
    )
    
    # Microstructure Confirmation
    # Calculate effective tick size and variability
    data['effective_tick'] = data['amount'] / data['volume']
    data['tick_size_variability'] = data['effective_tick'].rolling(window=5, min_periods=3).std()
    
    # Analyze trade size patterns
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['large_trade_threshold'] = data['avg_trade_size'].rolling(window=10, min_periods=5).quantile(0.8)
    data['large_trade_concentration'] = np.where(
        data['avg_trade_size'] > data['large_trade_threshold'],
        data['amount'] / data['amount'].rolling(window=5, min_periods=3).mean(),
        0
    )
    
    # Trade size momentum
    data['trade_size_momentum'] = data['avg_trade_size'].pct_change(periods=3)
    
    # Microstructure score
    data['microstructure_score'] = (
        -data['tick_size_variability'] +  # Lower variability is better
        data['large_trade_concentration'] +  # Higher concentration suggests institutional flow
        data['trade_size_momentum']  # Positive momentum in trade size
    )
    
    # Adaptive Signal Generation
    # Liquidity quality filter
    data['turnover_rate'] = data['volume'] / data['amount']
    data['avg_turnover_10d'] = data['turnover_rate'].rolling(window=10, min_periods=5).mean()
    data['liquidity_quality'] = data['turnover_rate'] / data['avg_turnover_10d']
    
    # Combine momentum with microstructure confirmation
    # High volatility: emphasize stability and trade size confirmation
    high_vol_signal = (
        data['regime_momentum'] * 
        (1 + data['microstructure_score']) * 
        np.where(data['liquidity_quality'] > 0.8, 1.2, 0.8)
    )
    
    # Low volatility: emphasize acceleration and institutional flow
    low_vol_signal = (
        data['regime_momentum'] * 
        (1 + data['large_trade_concentration']) * 
        np.where(data['liquidity_quality'] > 1.0, 1.1, 0.9)
    )
    
    # Final alpha factor
    alpha_factor = np.where(
        data['high_vol_regime'],
        high_vol_signal,
        low_vol_signal
    )
    
    # Confidence-based scaling
    momentum_strength = np.abs(data['regime_momentum'])
    microstructure_alignment = np.abs(data['microstructure_score'])
    liquidity_condition = np.where(data['liquidity_quality'].between(0.8, 1.2), 1.0, 0.7)
    
    # Final confidence adjustment
    confidence_multiplier = np.where(
        (momentum_strength > momentum_strength.rolling(window=20, min_periods=10).quantile(0.7)) &
        (microstructure_alignment > microstructure_alignment.rolling(window=20, min_periods=10).quantile(0.6)) &
        (liquidity_condition == 1.0),
        1.2,  # High confidence
        np.where(
            (momentum_strength > momentum_strength.rolling(window=20, min_periods=10).quantile(0.4)) &
            (microstructure_alignment > microstructure_alignment.rolling(window=20, min_periods=10).quantile(0.3)),
            1.0,  # Medium confidence
            0.7   # Low confidence
        )
    )
    
    final_alpha = alpha_factor * confidence_multiplier
    
    return pd.Series(final_alpha, index=data.index, name='alpha_factor')
