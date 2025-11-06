import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Order Flow Imbalance with Regime-Aware Momentum Structure
    """
    data = df.copy()
    
    # Calculate basic price-based features
    data['returns'] = data['close'].pct_change()
    data['high_low_ratio'] = data['high'] / data['low']
    data['close_open_ratio'] = data['close'] / data['open']
    data['dollar_volume'] = data['volume'] * data['close']
    
    # Micro-Scale Order Flow Components (5-min equivalent using intraday proxies)
    data['micro_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['micro_imbalance'] = (data['close'] - data['open']) * data['volume']
    
    # Rolling micro-scale aggregation (5-day window for daily data)
    data['micro_pressure_ma'] = data['micro_pressure'].rolling(window=5, min_periods=3).mean()
    data['micro_imbalance_ma'] = data['micro_imbalance'].rolling(window=5, min_periods=3).mean()
    
    # Meso-Scale Order Flow Components
    # Large block trade concentration (using dollar volume volatility)
    data['dollar_volume_std'] = data['dollar_volume'].rolling(window=10, min_periods=7).std()
    data['block_concentration'] = data['dollar_volume'] / (data['dollar_volume_std'] + 1e-8)
    
    # Order size distribution skewness (using volume/amount relationship)
    data['avg_trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['size_skewness'] = data['avg_trade_size'].rolling(window=10, min_periods=7).apply(
        lambda x: x.skew() if len(x) > 2 else 0
    )
    
    # Macro-Scale Order Flow Components
    # Regime change detection using volatility clustering
    data['volatility_20d'] = data['returns'].rolling(window=20, min_periods=15).std()
    data['volatility_regime'] = (data['volatility_20d'] > data['volatility_20d'].rolling(window=50, min_periods=35).quantile(0.7)).astype(int)
    
    # Flow persistence (autocorrelation of order flow)
    data['flow_persistence'] = data['micro_imbalance'].rolling(window=15, min_periods=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 5 and x.std() > 0 else 0
    )
    
    # Market Regime Context
    # Volatility regime classification
    data['vol_regime_high'] = (data['volatility_20d'] > data['volatility_20d'].rolling(window=50, min_periods=35).median()).astype(int)
    
    # Trend regime assessment
    data['price_trend_20d'] = data['close'].rolling(window=20, min_periods=15).apply(
        lambda x: (x[-1] - x[0]) / x[0] if len(x) > 1 else 0
    )
    data['trend_regime'] = np.sign(data['price_trend_20d'])
    
    # Regime-Adaptive Momentum
    # High-volatility momentum (shorter lookback)
    data['momentum_high_vol'] = data['close'] / data['close'].rolling(window=5, min_periods=3).mean() - 1
    
    # Low-volatility momentum (longer lookback)
    data['momentum_low_vol'] = data['close'] / data['close'].rolling(window=20, min_periods=15).mean() - 1
    
    # Multi-Regime Framework
    data['bull_regime'] = ((data['trend_regime'] > 0) & (data['vol_regime_high'] == 0)).astype(int)
    data['bear_regime'] = ((data['trend_regime'] < 0) & (data['vol_regime_high'] == 1)).astype(int)
    data['range_regime'] = ((data['trend_regime'].abs() < 0.05) & (data['vol_regime_high'] == 0)).astype(int)
    
    # Regime-appropriate momentum selection
    data['regime_momentum'] = (
        data['bull_regime'] * data['momentum_low_vol'] +
        data['bear_regime'] * data['momentum_high_vol'] +
        data['range_regime'] * (data['momentum_high_vol'] + data['momentum_low_vol']) / 2
    )
    
    # Combine Order Flow Layers with information content weighting
    # Normalize components
    micro_weighted = 0.4 * (data['micro_pressure_ma'] - data['micro_pressure_ma'].rolling(window=20, min_periods=15).mean())
    meso_weighted = 0.3 * (data['block_concentration'] - data['block_concentration'].rolling(window=20, min_periods=15).mean())
    macro_weighted = 0.3 * (data['flow_persistence'] - data['flow_persistence'].rolling(window=20, min_periods=15).mean())
    
    # Calculate flow coherence (correlation between scales)
    data['flow_coherence'] = (
        data['micro_pressure_ma'].rolling(window=10, min_periods=7).corr(data['block_concentration']) +
        data['micro_pressure_ma'].rolling(window=10, min_periods=7).corr(data['flow_persistence'])
    ) / 2
    
    # Composite order flow factor
    data['order_flow_composite'] = (
        micro_weighted + meso_weighted + macro_weighted
    ) * (1 + data['flow_coherence'].fillna(0))
    
    # Integrate Regime Signals with flow-confirmed momentum
    regime_weight = 0.6
    flow_weight = 0.4
    
    # Final composite factor: Order flow × regime momentum
    alpha_factor = (
        regime_weight * data['regime_momentum'] * (1 + data['order_flow_composite']) +
        flow_weight * data['order_flow_composite'] * (1 + data['regime_momentum'])
    )
    
    # Multi-scale confirmation scoring
    confirmation_score = (
        (data['micro_pressure_ma'] > data['micro_pressure_ma'].rolling(window=20, min_periods=15).mean()).astype(int) +
        (data['block_concentration'] > data['block_concentration'].rolling(window=20, min_periods=15).mean()).astype(int) +
        (data['flow_persistence'] > 0).astype(int)
    ) / 3
    
    # Apply confirmation filter
    final_factor = alpha_factor * (1 + 0.2 * confirmation_score)
    
    return final_factor
