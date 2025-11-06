import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Transition Alpha Factor
    Detects transitions between different volatility regimes and captures
    the dynamics of volatility jumps and term structure changes.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility regime detection parameters
    short_window = 5
    medium_window = 20
    long_window = 60
    
    for i in range(long_window, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Volatility Clustering Regime Detection
        # High-low range persistence asymmetry
        hl_range = (current_data['high'] - current_data['low']) / current_data['close']
        hl_volatility = hl_range.rolling(window=medium_window).std()
        
        # Volume-volatility coupling regime shifts
        volume_ma = current_data['volume'].rolling(window=medium_window).mean()
        vol_vol_coupling = (hl_range / volume_ma.rolling(window=medium_window).std()).iloc[-1]
        
        # Overnight vs intraday volatility regime divergence
        overnight_vol = (current_data['open'] / current_data['close'].shift(1) - 1).abs().rolling(window=short_window).std()
        intraday_vol = hl_range.rolling(window=short_window).std()
        vol_regime_divergence = (overnight_vol / intraday_vol).iloc[-1]
        
        # 2. Volatility Jump Dynamics
        # Price gap absorption capacity asymmetry
        price_gaps = (current_data['open'] / current_data['close'].shift(1) - 1).abs()
        gap_absorption = (current_data['high'] - current_data['low']) / price_gaps
        gap_capacity = gap_absorption.rolling(window=short_window).mean().iloc[-1]
        
        # Volatility spillover directionality
        vol_momentum_short = hl_volatility.pct_change(periods=short_window).iloc[-1]
        vol_momentum_medium = hl_volatility.pct_change(periods=medium_window).iloc[-1]
        vol_spillover = vol_momentum_short - vol_momentum_medium
        
        # Jump mean-reversion speed difference
        recent_jumps = price_gaps.rolling(window=short_window).std()
        historical_jumps = price_gaps.rolling(window=medium_window).std()
        jump_reversion = (recent_jumps / historical_jumps).iloc[-1]
        
        # 3. Volatility Term Structure Dynamics
        # Short-term vs long-term volatility momentum
        short_term_vol = hl_volatility.rolling(window=short_window).mean()
        long_term_vol = hl_volatility.rolling(window=long_window).mean()
        vol_term_momentum = (short_term_vol.pct_change() - long_term_vol.pct_change()).iloc[-1]
        
        # Volatility curve steepening/flattening signals
        vol_curve_steepness = (short_term_vol / long_term_vol).iloc[-1]
        
        # Regime-dependent volatility term premium
        vol_term_premium = (hl_volatility - long_term_vol).iloc[-1]
        
        # Combine components with regime-dependent weights
        regime_stability = 1.0 / (1.0 + abs(vol_regime_divergence - 1.0))
        
        # Main alpha calculation
        volatility_transition_alpha = (
            regime_stability * vol_vol_coupling +
            gap_capacity * vol_spillover +
            jump_reversion * vol_term_momentum +
            vol_curve_steepness * vol_term_premium
        )
        
        result.iloc[i] = volatility_transition_alpha
    
    # Handle initial NaN values
    result = result.fillna(method='bfill').fillna(0)
    
    return result
