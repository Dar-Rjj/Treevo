import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Volatility-Adjusted Intraday Momentum Decay
    # Calculate Intraday Momentum
    intraday_momentum = (df['high'] - df['low']) / df['close']
    
    # Compute Rolling Volatility
    high_low_range = df['high'] - df['low']
    volatility = high_low_range.rolling(window=5).std()
    
    # Apply Exponential Decay
    decay_factor = 0.5  # 3-day decay factor
    decay_weights = np.array([decay_factor**i for i in range(3)])[::-1]
    decay_weights = decay_weights / decay_weights.sum()
    
    # Combine Components
    volatility_adjusted = intraday_momentum / (volatility + 1e-8)
    decayed_momentum = volatility_adjusted.rolling(window=3).apply(
        lambda x: np.sum(x * decay_weights), raw=True
    )
    
    # Volume-Price Divergence Acceleration
    # Calculate Price Trend
    def linear_slope(series):
        x = np.arange(len(series))
        return stats.linregress(x, series.values).slope if len(series) >= 2 else 0
    
    price_trend = df['close'].rolling(window=10).apply(linear_slope, raw=False)
    
    # Calculate Volume Trend
    volume_trend = df['volume'].rolling(window=10).apply(linear_slope, raw=False)
    
    # Detect Divergence
    divergence_magnitude = np.abs(price_trend * volume_trend)
    divergence_sign = np.sign(price_trend) != np.sign(volume_trend)
    divergence = divergence_magnitude * divergence_sign.astype(float)
    
    # Measure Acceleration
    divergence_acceleration = divergence.diff(5).rolling(window=3).mean()
    
    # Amplitude-Modulated Return Persistence
    # Identify Large Moves
    high_low_range_20d = (df['high'] - df['low']).rolling(window=20).median()
    large_moves = (df['high'] - df['low']) > high_low_range_20d
    
    # Calculate Return Persistence
    returns = df['close'].pct_change()
    sign_changes = returns.diff().ne(0).cumsum()
    persistence = sign_changes.groupby(sign_changes).cumcount() + 1
    persistence = persistence * np.sign(returns)
    
    # Modulate by Amplitude
    amplitude_weight = (df['high'] - df['low']) / df['close']
    modulated_persistence = persistence * amplitude_weight * np.abs(persistence)
    
    # Combine Signals
    amplitude_signal = modulated_persistence * large_moves.astype(float)
    
    # Liquidity-Efficient Price Reversal
    # Calculate Price Reversal
    price_reversal = -df['close'].pct_change(2)
    
    # Assess Liquidity Efficiency
    price_move = df['close'].diff().abs()
    liquidity_efficiency = df['amount'] / (price_move + 1e-8)
    
    # Weight Reversal by Efficiency
    efficiency_weighted = price_reversal / (liquidity_efficiency + 1e-8)
    activity_level = df['volume'].rolling(window=5).mean()
    liquidity_signal = efficiency_weighted * activity_level
    
    # Asymmetric Volatility Response Factor
    # Measure Upside Volatility
    upside_moves = np.maximum(df['high'] - df['close'].shift(1), 0)
    upside_volatility = upside_moves.rolling(window=5).std()
    
    # Measure Downside Volatility
    downside_moves = np.maximum(df['close'].shift(1) - df['low'], 0)
    downside_volatility = downside_moves.rolling(window=5).std()
    
    # Compute Asymmetry Ratio
    asymmetry_ratio = np.log((upside_volatility + 1e-8) / (downside_volatility + 1e-8))
    
    # Combine with Momentum
    momentum_5d = df['close'].pct_change(5)
    asymmetric_signal = momentum_5d * asymmetry_ratio
    
    # Regime-Adaptive Trend Strength
    # Identify Market Regime
    range_volatility = (df['high'] - df['low']).rolling(window=20).std()
    volume_volatility = df['volume'].rolling(window=20).std()
    regime_score = range_volatility * volume_volatility
    trending_regime = regime_score > regime_score.rolling(window=20).median()
    
    # Calculate Trend Strength
    trend_short = df['close'].rolling(window=5).apply(linear_slope, raw=False)
    trend_medium = df['close'].rolling(window=10).apply(linear_slope, raw=False)
    trend_long = df['close'].rolling(window=20).apply(linear_slope, raw=False)
    trend_strength = (trend_short + trend_medium + trend_long) / 3
    
    # Adapt Weights by Regime
    regime_weight = trending_regime.astype(float) * 1.5 + (~trending_regime).astype(float) * 0.5
    regime_persistence = trending_regime.rolling(window=5).mean()
    
    # Generate Final Signal
    regime_signal = trend_strength * regime_weight * regime_persistence
    
    # Combine all factors with equal weights
    combined_factor = (
        decayed_momentum + 
        divergence_acceleration + 
        amplitude_signal + 
        liquidity_signal + 
        asymmetric_signal + 
        regime_signal
    ) / 6
    
    return combined_factor
