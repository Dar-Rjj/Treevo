import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Liquidity-Driven Microstructure Asymmetry factor capturing bid-ask spread dynamics,
    order flow imbalance, market impact asymmetry, and microstructure regime classification.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Bid-Ask Spread Dynamics
    # Estimate spread using high-low range relative to close (proxy for spread volatility)
    spread_volatility = (data['high'] - data['low']) / data['close']
    
    # Spread persistence using rolling autocorrelation
    spread_persistence = spread_volatility.rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # Quote intensity asymmetry using volume/amount ratio changes
    quote_intensity = (data['volume'] / data['amount']).pct_change(periods=1)
    
    # 2. Order Flow Imbalance
    # Tick-level directionality using close-open relationship with volume weighting
    price_momentum = (data['close'] - data['open']) / data['open']
    volume_weighted_direction = price_momentum * np.log1p(data['volume'])
    
    # Large trade absorption using volume volatility relative to price impact
    volume_volatility = data['volume'].rolling(window=5).std()
    price_impact = (data['high'] - data['low']).abs() / data['close']
    absorption_capacity = volume_volatility / (price_impact + 1e-8)
    
    # Hidden liquidity detection using abnormal volume patterns
    volume_ma = data['volume'].rolling(window=20).mean()
    volume_std = data['volume'].rolling(window=20).std()
    hidden_liquidity = (data['volume'] - volume_ma) / (volume_std + 1e-8)
    
    # 3. Market Impact Asymmetry
    # Buy vs sell pressure decay using asymmetric price responses
    up_moves = (data['close'] > data['open']).astype(int)
    down_moves = (data['close'] < data['open']).astype(int)
    
    # Price elasticity to volume shocks
    volume_shock = data['volume'].pct_change(periods=1)
    price_response = data['close'].pct_change(periods=1)
    elasticity = price_response / (volume_shock + 1e-8)
    
    # Temporary vs permanent impact separation
    intraday_range = (data['high'] - data['low']) / data['close']
    close_to_open_gap = (data['close'] - data['open'].shift(1)) / data['open'].shift(1)
    temporary_impact = intraday_range - np.abs(close_to_open_gap)
    
    # 4. Microstructure Regime Classification
    # High-frequency vs low-frequency dominance using volume clustering
    volume_autocorr = data['volume'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # Market maker inventory cycles using price reversal patterns
    price_reversal = -data['close'].pct_change(periods=1) * data['close'].pct_change(periods=2)
    
    # Latent liquidity state using combined liquidity signals
    liquidity_state = (
        spread_volatility.rolling(window=5).mean() + 
        np.abs(quote_intensity.rolling(window=5).mean()) +
        hidden_liquidity.rolling(window=5).std()
    )
    
    # Combine all components with appropriate weights
    factor = (
        0.15 * spread_persistence.fillna(0) +
        0.12 * quote_intensity.fillna(0) +
        0.18 * volume_weighted_direction.fillna(0) +
        0.14 * absorption_capacity.fillna(0) +
        0.11 * hidden_liquidity.fillna(0) +
        0.10 * elasticity.fillna(0) +
        0.08 * temporary_impact.fillna(0) +
        0.06 * volume_autocorr.fillna(0) +
        0.06 * price_reversal.fillna(0)
    ) / liquidity_state.rolling(window=3).mean().fillna(1)
    
    # Final normalization
    factor = (factor - factor.rolling(window=20).mean()) / (factor.rolling(window=20).std() + 1e-8)
    
    return factor
