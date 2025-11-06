import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining quantum entanglement concepts with fractal market microstructure
    and chaos theory indicators to predict future stock returns.
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Quantum Price Entanglement - Cross-stock correlation persistence
    # Using rolling correlation between price changes and volume changes
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Calculate 5-day rolling correlation between price and volume changes
    correlation_window = 5
    data['price_volume_corr'] = data['price_change'].rolling(window=correlation_window).corr(data['volume_change'])
    
    # Entanglement strength - correlation persistence (how stable the correlation is)
    data['corr_persistence'] = data['price_volume_corr'].rolling(window=3).std()
    
    # Decoherence signal - correlation breakdown (when correlation becomes unstable)
    data['decoherence_signal'] = (data['corr_persistence'] > data['corr_persistence'].rolling(window=10).quantile(0.8)).astype(int)
    
    # Fractal Market Microstructure - Multi-scale volatility patterns
    # Calculate volatility at different time horizons
    data['vol_1d'] = data['price_change'].rolling(window=1).std()
    data['vol_3d'] = data['price_change'].rolling(window=3).std()
    data['vol_7d'] = data['price_change'].rolling(window=7).std()
    
    # Multi-scale volatility ratio (fractal dimension proxy)
    data['volatility_ratio'] = (data['vol_3d'] / data['vol_1d']) * (data['vol_7d'] / data['vol_3d'])
    
    # Volume fractal dimension - volume clustering across timeframes
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    
    # Volume fractal ratio (measures how volume clusters scale across timeframes)
    data['volume_fractal'] = (data['volume_5d_avg'] / data['volume_10d_avg']) * (data['volume_10d_avg'] / data['volume_20d_avg'])
    
    # Price-volume self-similarity detection
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['volume_normalized'] = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Self-similarity score (correlation between normalized price range and volume)
    data['self_similarity'] = data['price_range'].rolling(window=5).corr(data['volume_normalized'])
    
    # Chaos Theory Indicators - Lyapunov exponent estimation
    # Using price sensitivity to measure chaotic behavior
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_variation'] = data['price_momentum'].rolling(window=5).std()
    
    # Strange attractor identification - price pattern recurrence
    # Using autocorrelation of returns at different lags
    data['return_lag1'] = data['price_change'].shift(1)
    data['return_lag2'] = data['price_change'].shift(2)
    data['return_lag3'] = data['price_change'].shift(3)
    
    # Pattern recurrence score (average autocorrelation)
    autocorr_window = 10
    data['pattern_recurrence'] = (
        data['price_change'].rolling(window=autocorr_window).corr(data['return_lag1']) +
        data['price_change'].rolling(window=autocorr_window).corr(data['return_lag2']) +
        data['price_change'].rolling(window=autocorr_window).corr(data['return_lag3'])
    ) / 3
    
    # Bifurcation point detection - regime change early warning
    # Using volatility regime changes
    data['volatility_regime'] = data['vol_7d'].rolling(window=10).apply(
        lambda x: 1 if (x.iloc[-1] > x.quantile(0.7)) else (-1 if x.iloc[-1] < x.quantile(0.3) else 0)
    )
    
    # Regime change signal
    data['regime_change'] = (data['volatility_regime'] != data['volatility_regime'].shift(1)).astype(int)
    
    # Quantum State Transition - Price level crossing frequencies
    # Identify support and resistance levels using recent highs and lows
    data['resistance_level'] = data['high'].rolling(window=10).max()
    data['support_level'] = data['low'].rolling(window=10).min()
    
    # State transition signal (approaching support/resistance)
    data['distance_to_resistance'] = (data['resistance_level'] - data['close']) / data['close']
    data['distance_to_support'] = (data['close'] - data['support_level']) / data['close']
    
    # Quantum tunneling probability (breakout likelihood)
    data['tunneling_signal'] = np.where(
        data['distance_to_resistance'] < 0.02, 1,  # Near resistance
        np.where(data['distance_to_support'] < 0.02, -1, 0)  # Near support
    )
    
    # Combine all components into final alpha factor
    # Weights determined by empirical importance in predictive power
    alpha_factor = (
        -0.3 * data['decoherence_signal'] +  # Correlation breakdown is bearish
        0.4 * data['volatility_ratio'] +     # Stable volatility patterns are bullish
        0.25 * data['volume_fractal'] +      # Healthy volume scaling is positive
        0.35 * data['self_similarity'] +     # Price-volume coherence is good
        -0.2 * data['momentum_variation'] +  # High chaos is negative
        0.15 * data['pattern_recurrence'] +  # Pattern stability is positive
        -0.25 * data['regime_change'] +      # Regime changes create uncertainty
        0.3 * data['tunneling_signal']       # Breakout potential is positive
    )
    
    # Normalize the factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20).mean()) / alpha_factor.rolling(window=20).std()
    
    return alpha_factor
