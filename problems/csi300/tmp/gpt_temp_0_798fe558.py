import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Momentum with Liquidity Filtering alpha factor
    
    This factor combines momentum signals across multiple timeframes, 
    adjusts for volatility regimes, and filters based on liquidity conditions
    to generate regime-aware momentum signals.
    """
    
    # Data validation
    required_cols = ['open', 'high', 'low', 'close', 'amount', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Price Momentum Component with Exponential Decay
    def momentum_with_decay(window, decay_rate=0.9):
        """Calculate momentum with exponential decay weighting"""
        momentum = df['close'].pct_change(window)
        weights = np.array([decay_rate ** i for i in range(window, 0, -1)])
        weights = weights / weights.sum()
        
        # Apply weighted rolling average
        decayed_momentum = momentum.rolling(window).apply(
            lambda x: np.sum(x * weights) if not x.isnull().any() else np.nan
        )
        return decayed_momentum
    
    # Multi-timeframe momentum
    short_momentum = momentum_with_decay(5, 0.95)      # 5-day with fast decay
    medium_momentum = momentum_with_decay(10, 0.9)     # 10-day with medium decay
    long_momentum = momentum_with_decay(20, 0.85)      # 20-day with slow decay
    
    # Volatility Regime Classification
    rolling_vol = returns.rolling(20).std()
    vol_quantiles = rolling_vol.rolling(60).apply(
        lambda x: pd.qcut(x, q=[0, 0.3, 0.7, 1.0], labels=False, duplicates='drop').iloc[-1] 
        if not x.isnull().any() else np.nan, raw=False
    )
    
    # Regime-specific momentum adjustment
    def regime_adjustment(momentum, regime):
        """Adjust momentum based on volatility regime"""
        if regime == 0:  # Low volatility
            return momentum * 1.2  # Amplify in low vol
        elif regime == 1:  # Normal volatility
            return momentum * 1.0  # No adjustment
        else:  # High volatility
            return momentum * 0.7  # Dampen in high vol
    
    # Apply regime adjustments
    short_momentum_adj = short_momentum.combine(vol_quantiles, 
                                               lambda mom, reg: regime_adjustment(mom, reg))
    medium_momentum_adj = medium_momentum.combine(vol_quantiles, 
                                                 lambda mom, reg: regime_adjustment(mom, reg))
    long_momentum_adj = long_momentum.combine(vol_quantiles, 
                                             lambda mom, reg: regime_adjustment(mom, reg))
    
    # Liquidity Assessment
    # Volume-based liquidity
    volume_trend = (df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean() - 1)
    volume_stability = 1 / (df['volume'].rolling(10).std() / df['volume'].rolling(10).mean())
    volume_participation = df['volume'] / df['volume'].rolling(20).mean()
    
    # Amount-based liquidity
    avg_trade_size = df['amount'] / df['volume']
    amount_volatility = 1 / (avg_trade_size.rolling(10).std() / avg_trade_size.rolling(10).mean())
    trading_quality = df['amount'].rolling(5).mean() / df['amount'].rolling(20).mean()
    
    # Combined Liquidity Score
    volume_score = (volume_trend.rank() + volume_stability.rank() + volume_participation.rank()) / 3
    amount_score = (avg_trade_size.rank() + amount_volatility.rank() + trading_quality.rank()) / 3
    liquidity_score = (volume_score + amount_score) / 2
    
    # Normalize liquidity score to 0-1 range
    liquidity_filter = (liquidity_score - liquidity_score.rolling(60).min()) / \
                      (liquidity_score.rolling(60).max() - liquidity_score.rolling(60).min())
    
    # Signal Integration
    # Multi-timeframe momentum combination with regime weights
    momentum_weights = pd.DataFrame({
        'short': short_momentum_adj,
        'medium': medium_momentum_adj,
        'long': long_momentum_adj
    })
    
    # Dynamic weights based on volatility regime
    def calculate_dynamic_weights(regime):
        if regime == 0:  # Low volatility - favor longer-term
            return [0.2, 0.3, 0.5]  # short, medium, long
        elif regime == 1:  # Normal volatility - balanced
            return [0.3, 0.4, 0.3]  # short, medium, long
        else:  # High volatility - favor shorter-term
            return [0.5, 0.3, 0.2]  # short, medium, long
    
    # Apply dynamic weights
    combined_momentum = pd.Series(index=df.index, dtype=float)
    for date in df.index:
        if pd.notna(vol_quantiles.loc[date]):
            weights = calculate_dynamic_weights(vol_quantiles.loc[date])
            weighted_sum = (momentum_weights.loc[date, 'short'] * weights[0] +
                          momentum_weights.loc[date, 'medium'] * weights[1] +
                          momentum_weights.loc[date, 'long'] * weights[2])
            combined_momentum.loc[date] = weighted_sum
    
    # Liquidity Filter Application
    final_alpha = combined_momentum * liquidity_filter
    
    # Final validation and cleaning
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = final_alpha.fillna(method='ffill').fillna(0)
    
    # Ensure reasonable scale
    final_alpha = final_alpha / final_alpha.abs().rolling(60).mean()
    
    return final_alpha
