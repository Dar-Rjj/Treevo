import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Regime-Adaptive Alpha Factor
    Combines market regime detection with multi-timeframe momentum signals
    """
    
    # Market Regime Detection
    def detect_market_regime(df):
        # Volatility regime
        vol_20 = df['close'].pct_change().rolling(20).std()
        vol_60 = df['close'].pct_change().rolling(60).std()
        vol_ratio = vol_20 / vol_60
        
        # Range-based volatility
        atr = (df['high'] - df['low']).rolling(14).mean()
        atr_pct = atr / df['close']
        range_compression = (df['high'] - df['low']) / df['close']
        
        # Trend regime
        ma_5 = df['close'].rolling(5).mean()
        ma_20 = df['close'].rolling(20).mean()
        ma_10 = df['close'].rolling(10).mean()
        ma_50 = df['close'].rolling(50).mean()
        
        trend_short = (df['close'] > ma_5).astype(int) + (df['close'] > ma_20).astype(int)
        trend_medium = (df['close'] > ma_10).astype(int) + (df['close'] > ma_50).astype(int)
        
        # Regime classification
        high_vol_regime = (vol_ratio > 1.2).astype(int)
        low_vol_regime = (vol_ratio < 0.8).astype(int)
        strong_trend = ((trend_short + trend_medium) >= 3).astype(int)
        no_trend = ((trend_short + trend_medium) <= 1).astype(int)
        
        return high_vol_regime, low_vol_regime, strong_trend, no_trend
    
    # Multi-Timeframe Factor Convergence
    def multi_timeframe_signals(df):
        # Short-term momentum (1-3 days)
        ret_1 = df['close'].pct_change(1)
        ret_3 = df['close'].pct_change(3)
        momentum_accel = ret_1 - ret_3.shift(2)
        
        # Volume-weighted intraday pressure
        up_days = (df['close'] > df['open']).astype(int)
        down_days = (df['close'] < df['open']).astype(int)
        up_volume_conc = (up_days * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
        down_volume_dist = (down_days * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
        volume_pressure = up_volume_conc - down_volume_dist
        
        # Medium-term momentum (5-10 days)
        ret_5 = df['close'].pct_change(5)
        ret_10 = df['close'].pct_change(10)
        vol_10 = df['close'].pct_change().rolling(10).std()
        vol_adjusted_ret = ret_5 / (vol_10 + 1e-8)
        
        # Volume confirmation
        volume_trend = df['volume'].rolling(10).mean() / df['volume'].rolling(30).mean()
        volume_surprise = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
        
        # Signal alignment
        short_term_signal = np.sign(momentum_accel) * np.abs(momentum_accel)
        medium_term_signal = np.sign(vol_adjusted_ret) * np.abs(vol_adjusted_ret)
        
        signal_alignment = (np.sign(short_term_signal) == np.sign(medium_term_signal)).astype(int)
        convergence_score = signal_alignment * (np.abs(short_term_signal) + np.abs(medium_term_signal))
        
        return momentum_accel, volume_pressure, vol_adjusted_ret, volume_trend, convergence_score
    
    # Dynamic Factor Adjustment
    def dynamic_adjustment(df, high_vol_regime, low_vol_regime, strong_trend, no_trend):
        # Regime-based factor emphasis
        momentum_accel, volume_pressure, vol_adjusted_ret, volume_trend, convergence_score = multi_timeframe_signals(df)
        
        # Mean reversion factors for high volatility
        mean_reversion = -df['close'].pct_change(3)
        volatility_breakout = (df['high'] - df['low']) / df['close'].rolling(5).std()
        
        # Momentum factors for low volatility
        momentum_persistence = df['close'].pct_change(5).rolling(5).mean()
        trend_strength = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Liquidity-aware scaling
        volume_to_vol = df['volume'] / (df['close'].pct_change().rolling(10).std() + 1e-8)
        turnover_eff = df['amount'] / (df['high'] - df['low'] + 1e-8)
        
        # Adaptive smoothing
        vol_level = df['close'].pct_change().rolling(20).std()
        short_window = 5
        long_window = 20
        adaptive_window = np.where(vol_level > vol_level.rolling(50).median(), short_window, long_window)
        
        # Apply adaptive smoothing to main signals
        def adaptive_rolling(series, windows):
            result = pd.Series(index=series.index, dtype=float)
            for i in range(len(series)):
                if i >= max(windows):
                    window = windows[i]
                    result.iloc[i] = series.iloc[i-window+1:i+1].mean()
            return result
        
        smoothed_momentum = adaptive_rolling(momentum_accel, adaptive_window.astype(int))
        
        return (mean_reversion, volatility_breakout, momentum_persistence, trend_strength, 
                volume_to_vol, turnover_eff, smoothed_momentum)
    
    # Composite Alpha Generation
    def generate_composite_alpha(df):
        # Get regime signals
        high_vol_regime, low_vol_regime, strong_trend, no_trend = detect_market_regime(df)
        
        # Get adjusted factors
        (mean_reversion, volatility_breakout, momentum_persistence, trend_strength,
         volume_to_vol, turnover_eff, smoothed_momentum) = dynamic_adjustment(df, high_vol_regime, low_vol_regime, strong_trend, no_trend)
        
        # Get multi-timeframe signals
        momentum_accel, volume_pressure, vol_adjusted_ret, volume_trend, convergence_score = multi_timeframe_signals(df)
        
        # Regime-weighted factor combination
        # High volatility: emphasize mean reversion
        high_vol_component = high_vol_regime * (0.6 * mean_reversion + 0.4 * volatility_breakout)
        
        # Low volatility: emphasize momentum
        low_vol_component = low_vol_regime * (0.7 * momentum_persistence + 0.3 * trend_strength)
        
        # Trend regime weights
        trend_component = strong_trend * (0.8 * smoothed_momentum + 0.2 * vol_adjusted_ret)
        no_trend_component = no_trend * (0.9 * mean_reversion + 0.1 * volume_pressure)
        
        # Multi-timeframe signal integration
        timeframe_multiplier = 1 + 0.5 * convergence_score
        recent_weight = 0.7
        historical_weight = 0.3
        
        recent_signals = (momentum_accel * recent_weight + 
                         vol_adjusted_ret * (1 - recent_weight))
        
        # Composite alpha
        composite_alpha = (
            high_vol_component.fillna(0) +
            low_vol_component.fillna(0) +
            trend_component.fillna(0) +
            no_trend_component.fillna(0)
        ) * timeframe_multiplier.fillna(1) + recent_signals.fillna(0)
        
        # Risk-adjusted final output
        alpha_vol = composite_alpha.rolling(20).std()
        risk_adjusted_alpha = composite_alpha / (alpha_vol + 1e-8)
        
        # Liquidity constraints
        volume_capacity = np.minimum(volume_to_vol / volume_to_vol.rolling(50).mean(), 2)
        turnover_efficiency = np.minimum(turnover_eff / turnover_eff.rolling(50).mean(), 1.5)
        
        final_alpha = risk_adjusted_alpha * volume_capacity * turnover_efficiency
        
        return final_alpha
    
    return generate_composite_alpha(df)
