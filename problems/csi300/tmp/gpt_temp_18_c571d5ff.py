import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-window momentum-volume synergy with volatility scaling and efficiency filters.
    Combines short-term (3-day) and medium-term (6-day) momentum signals with volume confirmation,
    volatility normalization, and range efficiency adjustments. Uses exponential decay weighting
    for robustness and emphasizes recent market dynamics while filtering noisy price action.
    """
    # Multi-window parameters for capturing different momentum horizons
    short_window = 3
    medium_window = 6
    
    # Exponential decay weights for both windows (emphasize recent observations)
    short_decay_weights = np.exp(-np.arange(short_window) / 1.2)
    short_decay_weights = short_decay_weights / short_decay_weights.sum()
    
    medium_decay_weights = np.exp(-np.arange(medium_window) / 2.0)
    medium_decay_weights = medium_decay_weights / medium_decay_weights.sum()
    
    # Calculate returns for momentum computation
    returns = df['close'].pct_change()
    
    # Compute decayed momentum for both windows
    short_decayed_momentum = pd.Series(index=df.index, dtype=float)
    medium_decayed_momentum = pd.Series(index=df.index, dtype=float)
    
    for i in range(medium_window, len(df)):
        # Short window momentum
        short_window_returns = returns.iloc[i-short_window+1:i+1].values
        short_decayed_momentum.iloc[i] = np.sum(short_window_returns * short_decay_weights)
        
        # Medium window momentum
        medium_window_returns = returns.iloc[i-medium_window+1:i+1].values
        medium_decayed_momentum.iloc[i] = np.sum(medium_window_returns * medium_decay_weights)
    
    # Volume momentum confirmation with multi-timeframe alignment
    short_volume_ma = df['volume'].rolling(window=short_window).mean()
    medium_volume_ma = df['volume'].rolling(window=medium_window).mean()
    
    short_volume_strength = df['volume'] / (short_volume_ma + 1e-7)
    medium_volume_strength = df['volume'] / (medium_volume_ma + 1e-7)
    
    # Volume alignment with momentum direction
    short_volume_alignment = np.sign(short_decayed_momentum) * short_volume_strength
    medium_volume_alignment = np.sign(medium_decayed_momentum) * medium_volume_strength
    
    # Volatility normalization using multi-timeframe volatility
    short_volatility = returns.rolling(window=short_window).std()
    medium_volatility = returns.rolling(window=medium_window).std()
    
    # Range efficiency factor (penalize excessive daily range usage)
    daily_range_ratio = (df['high'] - df['low']) / df['close']
    short_avg_range = daily_range_ratio.rolling(window=short_window).mean()
    medium_avg_range = daily_range_ratio.rolling(window=medium_window).mean()
    
    # Range efficiency: lower values for stocks with excessive volatility relative to trend
    short_range_efficiency = 1 / (1 + short_avg_range)
    medium_range_efficiency = 1 / (1 + medium_avg_range)
    
    # Gap persistence factor (reward stocks that maintain gap direction)
    opening_gap = df['open'] - df['close'].shift(1)
    gap_direction = np.sign(opening_gap)
    close_direction = np.sign(df['close'] - df['open'])
    gap_persistence = 1 - abs(gap_direction - close_direction) / 2  # 1 if same direction, 0 if reversed
    
    # Amount-based liquidity adjustment (higher amount suggests stronger conviction)
    amount_ma = df['amount'].rolling(window=short_window).mean()
    liquidity_strength = df['amount'] / (amount_ma + 1e-7)
    
    # Combine short and medium-term factors with appropriate weights
    short_term_factor = (short_decayed_momentum * short_volume_alignment * 
                        short_range_efficiency * gap_persistence * liquidity_strength) / (short_volatility + 1e-7)
    
    medium_term_factor = (medium_decayed_momentum * medium_volume_alignment * 
                         medium_range_efficiency * gap_persistence) / (medium_volatility + 1e-7)
    
    # Final composite factor: 60% short-term, 40% medium-term weighting
    alpha_factor = 0.6 * short_term_factor + 0.4 * medium_term_factor
    
    return alpha_factor
