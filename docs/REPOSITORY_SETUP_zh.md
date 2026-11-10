# 代码仓使用与版本发布

公开仓库：https://github.com/LeashHawk/rainnet-nowcast-bench 。
论文对应代码版本：[v1.0](https://github.com/LeashHawk/rainnet-nowcast-bench/tree/v1.0)。

## 复现流程

1. 软件检查：运行 `python scripts/check_repository.py`、`python examples/synthetic_demo.py` 和 README 列出的检查脚本。合成演示使用 identity canary 验证数据布局和评估器。
2. 主协议分析：使用 `results/` 中的 JSON 和逐窗口 NPZ 生成统计与图表，命令见 README 和 `docs/USER_GUIDE.md`。
3. 模型评估：准备 RainNet 数据与训练好的模型权重，通过 `--checkpoints` 指定权重文件，并使用 `CHECKPOINTS.sha256` 核对哈希。
4. 模型训练：使用下面的种子包装器启动训练，将每次运行的配置、日志和权重分别保存。

```bash
python scripts/train_seeded.py --seed 0 -- --arch simvp --loss mse --data_path data/rainnet --batch_size 4 --lr 1e-3 --lr_schedule cosine --num_epochs 50 --early_stop_patience 10 --log_path logs/simvp_seed0.log --checkpoints checkpoints/simvp_seed0.pth
```

分别以种子 0、1、2 和独立输出路径运行，记录各次运行的软件环境。训练入口支持的四种架构及参数见用户指南。主协议使用窗口级验证划分，训练与验证窗口存在帧重叠；整月隔离协议使用带有显式 `val` 列表的索引。

## 文件与来源

- `corrected/`、`benchmark_tools/` 和结果文件保留其科研来源记录。
- `docs/provenance/` 保存整理前的说明文件与清单。
- `MANIFEST.sha256` 记录发布版本的文件哈希；自身和 `.git/` 目录不纳入校验。
- `.gitattributes` 禁止换行转换，使克隆后的文件可直接校验。
- 逐窗口 NPZ 结果纳入版本管理；原始全量数据和模型权重按 `.gitignore` 规则管理。
- 历史 shell 编排脚本保留服务器路径，新环境运行请使用 README 中的 Python 命令。

## 版本发布

论文作者：Anlin Zhang、Shihao Li、Xibei Yu、Xinyue Xing、Jiaqi Liu、Jialin Wei、Yishuo Wang、Junju Du。
代码开发者：Anlin Zhang、Shihao Li。

提交审核过的文件，然后推送代码与版本标签：

```bash
git add <reviewed-files>
git commit -m "Update repository documentation"
git push origin main
git tag -a <version> -m "Describe the release"
git push origin <version>
```

在论文中记录所用版本标签。数据与权重归档应提供对应的版本信息和 SHA256，便于将评估文件与输入材料相互核对。
