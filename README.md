# 量化交易系统 - 快速开始

## 📁 项目结构

```
Quant_Project_Root/
│
├── config.yaml              # 全局配置 (重要!)
├── requirements.txt         # 依赖列表
│
├── data_manager.py          # 数据管理 (下载、缓存、清洗)
├── factor_library.py        # 因子库 (计算、测试)
├── backtest_engine.py       # 回测引擎
├── alpaca_execute_fixed.py  # 实盘执行器
│
├── /data_cache              # 数据缓存目录 (自动创建)
├── /logs                    # 日志目录 (自动创建)
│
└── .env                     # Alpaca API密钥 (需创建)
```

---

## 🚀 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥 (实盘交易需要)

创建 `.env` 文件:

```bash
APCA_API_KEY_ID=your_alpaca_key
APCA_API_SECRET_KEY=your_alpaca_secret
```

---

## 🧪 快速测试 (Ctrl+F5)

每个模块都可以独立测试!

### 测试1: 数据获取

```bash
python data_manager.py
```

**输出**: 下载20个股票的数据，显示质量检查

### 测试2: 因子计算

```bash
python factor_library.py
```

**输出**: 计算动量、反转因子，显示IC分析

### 测试3: 完整回测

```bash
python backtest_engine.py
```

**输出**: 运行2年回测，显示收益曲线和统计

---

## 📊 运行完整回测

```python
from backtest_engine import BacktestEngine

# 初始化
engine = BacktestEngine()

# 运行回测 (最近10年)
results = engine.run()

# 查看结果
print(results['stats'])
```

---

## 💰 实盘交易 (Alpaca)

### 模拟账户测试

```bash
python alpaca_execute_fixed.py --paper --dry-run
```

### 实际执行 (每周五自动)

```bash
python alpaca_execute_fixed.py --paper
```

### 强制立即执行

```bash
python alpaca_execute_fixed.py --paper --force
```

---

## ⚙️ 配置修改

所有配置都在 `config.yaml` 中:

### 修改股票池

```yaml
universe:
  mode: "sp500_nasdaq100"  # 或 "sp500" 或 "nasdaq100"
```

### 调整因子权重

```yaml
factors:
  momentum:
    weight: 0.70  # 动量因子权重
  pullback:
    weight: 0.30  # 反转因子权重
```

### 修改调仓频率

```yaml
portfolio:
  rebalance: "W-FRI"  # W-FRI=每周五, D=每天, M=每月
```

### 添加防御资产

```yaml
regime:
  defensive_allocation:
    BIL: 0.60   # 60% 现金等价物
    GLD: 0.30   # 30% 黄金
    TLT: 0.10   # 10% 长期国债
```

### 启用止损/止盈

```yaml
portfolio:
  stop_loss:
    enabled: true
    threshold: -0.08  # -8% 止损
  
  take_profit:
    enabled: true
    threshold: 0.20   # +20% 止盈
```

---

## 🔬 高级功能

### 1. 滚动窗口测试 (检验因子稳定性)

```python
from factor_library import rolling_window_test
from data_manager import DataManager

dm = DataManager()
universe = dm.get_universe_list()[:50]
close, volume = dm.get_market_data(universe)

# 每4年一个窗口，步进1年
results = rolling_window_test(close, volume, window_years=4, step_years=1)
```

**输出**: 每个窗口的IC统计，检查因子是否持续有效

### 2. 单因子IC分析

```python
from factor_library import FactorEngine, FactorAnalyzer

engine = FactorEngine()
analyzer = FactorAnalyzer()

factors = engine.compute_all_factors(close, volume)
returns = close.pct_change()

# IC分析
ic_df = analyzer.calculate_ic(factors['momentum_12_1'], returns)
ic_stats = analyzer.ic_summary(ic_df)
print(ic_stats)
```

### 3. 因子分组收益测试

```python
# 将股票按因子分5组，看收益差异
quantile_rets = analyzer.factor_quantile_returns(
    factor=factors['momentum_12_1'],
    returns=returns,
    quantiles=5,
    periods=20
)

print(quantile_rets.mean())  # 各组平均收益
```

---

## 🛠️ 添加新因子

### 步骤1: 在 `factor_library.py` 中添加函数

```python
def my_new_factor(close: pd.DataFrame, param: int = 10) -> pd.DataFrame:
    """我的新因子"""
    return close.rolling(param).std()  # 示例: 波动率
```

### 步骤2: 注册到因子引擎

```python
# 在 FactorEngine.FACTOR_REGISTRY 中添加
'my_factor': my_new_factor,
```

### 步骤3: 在 config.yaml 中启用

```yaml
factors:
  my_factor:
    enabled: true
    weight: 0.20
    param: 10
```

### 步骤4: 测试

```bash
python factor_library.py
```

---

## 📈 性能优化建议

1. **使用数据缓存**: `use_cache=True` (默认开启)
2. **减少回测股票数**: 测试时用前50个股票
3. **调整调仓频率**: 降低调仓频率减少计算
4. **并行下载**: yfinance 默认启用多线程

---

## ⚠️ 常见问题

### Q: 数据下载失败?

**A**: 检查网络连接，或使用VPN。也可切换到Alpaca数据源:

```yaml
data:
  source: "alpaca"  # 替换 "yahoo"
```

### Q: 因子IC很低?

**A**: 正常! 单因子IC通常在 0.02-0.05。组合后效果更好。

### Q: 回测收益不稳定?

**A**: 尝试:
1. 增加持仓数量 (`top_n: 50`)
2. 延长最短持有期 (`min_hold_days: 20`)
3. 启用趋势过滤 (`trend_gate: true`)

### Q: 实盘执行出错?

**A**: 检查:
1. API密钥是否正确
2. 是否在交易时间
3. 账户余额是否足够
4. 先用 `--dry-run` 测试

---

## 📝 文件说明

| 文件 | 功能 | 可独立运行 |
|------|------|-----------|
| `config.yaml` | 全局配置 | ❌ |
| `data_manager.py` | 数据获取与缓存 | ✅ |
| `factor_library.py` | 因子计算与分析 | ✅ |
| `backtest_engine.py` | 完整回测系统 | ✅ |
| `alpaca_execute_fixed.py` | 实盘执行 | ✅ |

---

## 🎯 下一步

1. **调整配置**: 修改 `config.yaml` 中的参数
2. **添加因子**: 在 `factor_library.py` 中实现新因子
3. **优化策略**: 通过滚动窗口测试找到最佳参数
4. **模拟交易**: 用 `--paper` 模式运行一段时间
5. **实盘部署**: 设置定时任务每周五自动执行

---

## 💡 提示

- **先测试后实盘**: 务必先用 `--dry-run` 验证逻辑
- **小仓位开始**: 实盘初期使用较小的 `top_n`
- **监控日志**: 检查 `logs/` 目录下的执行日志
- **定期回测**: 每月运行一次完整回测检查策略表现

---

## 📞 支持

遇到问题? 检查日志文件或提issue。

Happy Trading! 🚀