import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining multi-timeframe asymmetry analysis, session-based elasticity,
    amount microstructure signals, and momentum phase analysis.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Intraday Asymmetry Factors
    # Morning vs afternoon return ratio weighted by volume distribution
    morning_return = (data['close'] - data['open']) / data['open']
    afternoon_return = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1)
    
    # Volume-weighted return ratio (using rolling window for stability)
    volume_ratio = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    session_return_ratio = (morning_return / (afternoon_return + 1e-8)) * volume_ratio
    
    # Asymmetric volume efficiency
    price_change = data['close'] - data['open']
    upside_volume = data['volume'].where(price_change > 0, 0)
    downside_volume = data['volume'].where(price_change < 0, 0)
    
    upside_volume_ma = upside_volume.rolling(window=5, min_periods=1).mean()
    downside_volume_ma = downside_volume.rolling(window=5, min_periods=1).mean()
    
    volume_efficiency = (upside_volume_ma / (downside_volume_ma + 1e-8)) * \
                       np.sign(price_change) * np.abs(morning_return)
    
    # 2. Session-Based Price Elasticity
    # Morning price elasticity with volume adjustment
    high_low_range = data['high'] - data['low']
    open_low_range = data['open'] - data['low']
    
    morning_elasticity = ((data['high'] - data['open']) / (open_low_range + 1e-8)) * \
                        (data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean())
    
    # Session elasticity divergence
    elasticity_ma = morning_elasticity.rolling(window=5, min_periods=1).mean()
    elasticity_divergence = morning_elasticity - elasticity_ma
    
    # 3. Amount Microstructure Signals
    # Large trade concentration (amount per trade distribution skewness)
    trades_estimate = data['amount'] / (data['close'] * data['volume'] + 1e-8)
    amount_per_trade = data['amount'] / (trades_estimate + 1e-8)
    
    # Rolling skewness of amount per trade
    trade_size_skew = amount_per_trade.rolling(window=10, min_periods=1).apply(
        lambda x: (x - x.mean()).pow(3).mean() / (x.std() + 1e-8)**3 if x.std() > 0 else 0
    )
    
    # Microstructure efficiency
    price_volatility = data['close'].pct_change().rolling(window=5, min_periods=1).std()
    amount_volatility = data['amount'].pct_change().rolling(window=5, min_periods=1).std()
    microstructure_efficiency = amount_volatility / (price_volatility + 1e-8)
    
    # 4. Multi-Timeframe Momentum Analysis
    # Short-term vs medium-term momentum phase alignment
    short_term_momentum = data['close'].pct_change(periods=3)
    medium_term_momentum = data['close'].pct_change(periods=10)
    
    momentum_alignment = np.sign(short_term_momentum) * np.sign(medium_term_momentum) * \
                        np.abs(short_term_momentum - medium_term_momentum)
    
    # Phase transition signals with volume confirmation
    momentum_cross = short_term_momentum - medium_term_momentum
    volume_confirmation = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    phase_transition = momentum_cross * volume_confirmation
    
    # Combine all factors with appropriate weights
    factor = (
        0.25 * session_return_ratio +
        0.20 * volume_efficiency +
        0.15 * morning_elasticity +
        0.10 * elasticity_divergence +
        0.10 * trade_size_skew +
        0.10 * microstructure_efficiency +
        0.05 * momentum_alignment +
        0.05 * phase_transition
    )
    
    # Normalize the factor
    factor_ma = factor.rolling(window=20, min_periods=1).mean()
    factor_std = factor.rolling(window=20, min_periods=1).std()
    normalized_factor = (factor - factor_ma) / (factor_std + 1e-8)
    
    return normalized_factor
