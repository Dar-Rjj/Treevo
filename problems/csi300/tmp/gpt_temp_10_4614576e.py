import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Dimensional Market Microstructure Momentum with Regime-Dependent Efficiency
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Microstructure Momentum Core Components
    
    # Price Formation Efficiency
    # Opening gap efficiency measure
    data['gap_ratio'] = (data['open'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['gap_efficiency'] = data['gap_ratio'].rolling(window=5).std()
    
    # Overnight vs Intraday Momentum
    data['overnight_ret'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_ret'] = (data['close'] - data['open']) / data['open']
    data['overnight_intraday_ratio'] = data['overnight_ret'] / (data['intraday_ret'] + 1e-8)
    
    # Trade Flow Imbalance Dynamics
    # Volume-Weighted Price Momentum
    data['vwap'] = (data['amount'] / data['volume']).replace([np.inf, -np.inf], np.nan)
    data['close_vwap_divergence'] = (data['close'] - data['vwap']) / data['vwap']
    data['vwap_momentum'] = data['close_vwap_divergence'].rolling(window=3).mean()
    
    # Volume-accelerated price moves
    data['volume_ret'] = data['volume'] * data['intraday_ret']
    data['volume_acceleration'] = data['volume_ret'].rolling(window=5).mean()
    
    # Market Impact Asymmetry
    # Up-volume vs down-volume impact
    data['up_day'] = data['close'] > data['open']
    data['down_day'] = data['close'] < data['open']
    
    up_volume_impact = data[data['up_day']]['volume'].rolling(window=10).mean() / \
                      data[data['up_day']]['intraday_ret'].abs().rolling(window=10).mean()
    down_volume_impact = data[data['down_day']]['volume'].rolling(window=10).mean() / \
                        data[data['down_day']]['intraday_ret'].abs().rolling(window=10).mean()
    
    data['volume_impact_ratio'] = up_volume_impact / (down_volume_impact + 1e-8)
    
    # 2. Regime-Dependent Efficiency Scoring
    
    # Volatility Regime
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['volatility_regime'] = data['daily_range'].rolling(window=20).rank(pct=True)
    
    # Volume Regime
    data['volume_regime'] = data['volume'].rolling(window=20).rank(pct=True)
    
    # Efficiency measures
    data['price_efficiency'] = (data['close'] - data['open']).abs() / (data['high'] - data['low'])
    data['efficiency_score'] = data['price_efficiency'].rolling(window=10).mean()
    
    # 3. Multi-Timeframe Momentum Integration
    
    # Ultra-short term (1-3 days)
    data['short_term_momentum'] = data['close'].pct_change(periods=3).rolling(window=3).mean()
    
    # Medium-term (5-10 days)
    data['medium_term_momentum'] = data['close'].pct_change(periods=5).rolling(window=5).mean()
    
    # Timeframe convergence
    data['timeframe_alignment'] = np.sign(data['short_term_momentum']) * np.sign(data['medium_term_momentum'])
    
    # 4. Adaptive Signal Weighting
    
    # Regime confidence weights
    data['vol_regime_confidence'] = 1 - (data['volatility_regime'] - 0.5).abs() * 2
    data['volume_regime_confidence'] = 1 - (data['volume_regime'] - 0.5).abs() * 2
    
    # Efficiency-based weights
    data['efficiency_weight'] = data['efficiency_score'].rolling(window=10).apply(
        lambda x: 1.0 if x.mean() > x.median() else 0.5
    )
    
    # 5. Final Alpha Factor Construction
    
    # Core momentum components
    momentum_components = [
        data['vwap_momentum'],
        data['volume_acceleration'],
        data['overnight_intraday_ratio'],
        data['gap_efficiency']
    ]
    
    # Normalize components
    normalized_momentum = []
    for component in momentum_components:
        normalized = (component - component.rolling(window=20).mean()) / component.rolling(window=20).std()
        normalized_momentum.append(normalized)
    
    # Combine momentum components
    microstructure_momentum = sum(normalized_momentum) / len(normalized_momentum)
    
    # Regime adjustments
    volatility_adjustment = np.where(
        data['volatility_regime'] > 0.7,  # High volatility
        microstructure_momentum * 0.7,    # Reduce signal in high vol
        np.where(
            data['volatility_regime'] < 0.3,  # Low volatility
            microstructure_momentum * 1.2,    # Amplify in low vol
            microstructure_momentum            # Neutral adjustment
        )
    )
    
    volume_adjustment = np.where(
        data['volume_regime'] > 0.7,      # High volume
        volatility_adjustment * 1.1,      # Slight amplification
        np.where(
            data['volume_regime'] < 0.3,  # Low volume
            volatility_adjustment * 0.8,  # Reduction in low volume
            volatility_adjustment          # Neutral
        )
    )
    
    # Timeframe convergence adjustment
    final_alpha = volume_adjustment * data['timeframe_alignment'] * data['efficiency_weight']
    
    # Apply regime confidence weights
    regime_confidence = (data['vol_regime_confidence'] + data['volume_regime_confidence']) / 2
    final_alpha = final_alpha * regime_confidence
    
    # Clean and return
    alpha_series = pd.Series(final_alpha, index=data.index)
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    
    return alpha_series
