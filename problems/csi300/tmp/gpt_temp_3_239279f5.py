import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Liquidity Momentum Alpha
    Combines asymmetric liquidity memory, regime detection, fractal efficiency,
    volatility-adjusted breakouts, and cross-sectional momentum with liquidity asymmetry
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols = ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    for i in range(len(df)):
        if i < 20:  # Minimum window for calculations
            result.iloc[i] = 0
            continue
            
        # Extract current window (only past and current data)
        window = df.iloc[max(0, i-59):i+1]  # 60-day window max
        
        # 1. Asymmetric Liquidity Memory
        # Buy vs sell flow autocorrelation difference
        returns = window['close'].pct_change().dropna()
        amount_changes = window['amount'].pct_change().dropna()
        
        # Directional flow classification
        buy_flow_mask = returns > 0
        sell_flow_mask = returns < 0
        
        if len(buy_flow_mask) > 5 and len(sell_flow_mask) > 5:
            buy_flow_autocorr = amount_changes[buy_flow_mask].autocorr(lag=1) if len(amount_changes[buy_flow_mask]) > 2 else 0
            sell_flow_autocorr = amount_changes[sell_flow_mask].autocorr(lag=1) if len(amount_changes[sell_flow_mask]) > 2 else 0
            liquidity_asymmetry = (buy_flow_autocorr - sell_flow_autocorr) if not (pd.isna(buy_flow_autocorr) or pd.isna(sell_flow_autocorr)) else 0
        else:
            liquidity_asymmetry = 0
        
        # 2. Regime Detection with Volume-Price Alignment
        # Price-amount divergence regime identification
        price_trend = window['close'].iloc[-5:].mean() / window['close'].iloc[-20:-15].mean() - 1
        amount_trend = window['amount'].iloc[-5:].mean() / window['amount'].iloc[-20:-15].mean() - 1
        
        # Volume validation of liquidity regimes
        volume_std = window['volume'].iloc[-20:].std()
        volume_current = window['volume'].iloc[-1]
        volume_regime = (volume_current - window['volume'].iloc[-20:].mean()) / volume_std if volume_std > 0 else 0
        
        regime_strength = np.sign(price_trend) * np.sign(amount_trend) * abs(volume_regime)
        
        # 3. Fractal Efficiency with Pressure-Liquidity Alignment
        # Multi-timeframe fractal efficiency analysis
        def fractal_efficiency(price_series, n):
            if len(price_series) < n:
                return 0
            net_move = abs(price_series.iloc[-1] - price_series.iloc[-n])
            total_move = sum(abs(price_series.diff().iloc[-n+1:]))
            return net_move / total_move if total_move > 0 else 0
        
        eff_short = fractal_efficiency(window['close'], 5)
        eff_medium = fractal_efficiency(window['close'], 10)
        eff_long = fractal_efficiency(window['close'], 20)
        
        # Liquidity pressure accumulation
        volume_pressure = (window['volume'].iloc[-5:].sum() - window['volume'].iloc[-10:-5].sum()) / window['volume'].iloc[-10:].mean() if window['volume'].iloc[-10:].mean() > 0 else 0
        
        fractal_score = (eff_short + eff_medium + eff_long) / 3 * volume_pressure
        
        # 4. Volatility-Adjusted Breakout with Liquidity Confirmation
        # Volatility calculation with liquidity adjustment
        vol_short = window['close'].iloc[-5:].pct_change().std()
        vol_medium = window['close'].iloc[-20:].pct_change().std()
        
        # Relative breakout with liquidity validation
        current_close = window['close'].iloc[-1]
        high_20 = window['high'].iloc[-20:].max()
        low_20 = window['low'].iloc[-20:].min()
        
        breakout_position = (current_close - low_20) / (high_20 - low_20) if (high_20 - low_20) > 0 else 0.5
        
        # Liquidity validation
        volume_ratio = window['volume'].iloc[-1] / window['volume'].iloc[-20:].mean() if window['volume'].iloc[-20:].mean() > 0 else 1
        volatility_adjustment = vol_short / vol_medium if vol_medium > 0 else 1
        
        breakout_score = (breakout_position - 0.5) * volume_ratio / max(volatility_adjustment, 0.1)
        
        # 5. Cross-Sectional Momentum with Liquidity Asymmetry
        # Relative momentum with liquidity conditions
        momentum_5 = window['close'].iloc[-1] / window['close'].iloc[-5] - 1
        momentum_10 = window['close'].iloc[-1] / window['close'].iloc[-10] - 1
        
        # Asymmetric liquidity weighting scheme
        amount_momentum = window['amount'].iloc[-5:].mean() / window['amount'].iloc[-10:-5].mean() - 1
        
        liquidity_weight = 1 + abs(amount_momentum) * np.sign(amount_momentum)
        momentum_score = (momentum_5 + momentum_10) / 2 * liquidity_weight
        
        # Combine all components with regime-adaptive weights
        regime_weight = abs(regime_strength)
        
        alpha_value = (
            liquidity_asymmetry * 0.2 +
            regime_strength * 0.25 +
            fractal_score * 0.15 +
            breakout_score * 0.2 +
            momentum_score * 0.2
        ) * (1 + 0.1 * regime_weight)
        
        result.iloc[i] = alpha_value
    
    # Normalize the final result
    if len(result) > 0:
        result = (result - result.mean()) / result.std() if result.std() > 0 else result
    
    return result
