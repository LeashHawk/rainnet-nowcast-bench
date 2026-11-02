# 代码仓建立与发布

本目录是独立代码仓根目录。上层目录的论文、期刊模板和临时排版文件不纳入 Git。

## 复现层次

1. 软件检查：`python scripts/check_repository.py`；安装依赖后运行 `python examples/synthetic_demo.py` 及 README 的检查脚本。演示中的 identity canary 使用目标值，仅验证评估器，不代表预测能力。
2. 已发布主实验分析：使用 `results/` 中已有 JSON 和逐窗口 NPZ 生成统计与图表，参阅 README。
3. 重新评估：另行取得 RainNet 与匹配 `CHECKPOINTS.sha256` 的权重。目前本地没有权重文件或已确认的公开下载地址。
4. 重新训练：使用现有训练入口，或下面的新种子包装器。新入口用于后续实验，不能证明历史运行的种子、环境或数值一致性。主协议的窗口验证划分存在帧重叠披露，不能与整月隔离协议混称。

```bash
python scripts/train_seeded.py --seed 0 -- --arch simvp --loss mse --data_path data/rainnet --batch_size 4 --lr 1e-3 --lr_schedule cosine --num_epochs 50 --early_stop_patience 10 --log_path logs/simvp_seed0.log --checkpoints checkpoints/simvp_seed0.pth
```

分别以种子 0、1、2 和独立输出路径运行。固定随机种子不保证跨设备或环境逐位一致；续训源码未恢复完整随机状态，不能声称续训等价于不中断运行。原始训练入口只覆盖其声明的四种架构，不用这个命令冒充全部论文行的完整训练矩阵。

## 文件与来源

- `corrected/`、`benchmark_tools/`、结果文件保持原始字节；不重命名科研源码目录，避免影响历史来源追踪。
- `docs/provenance/` 保留代码仓整理前的清单及修改过的说明文件。
- `MANIFEST.sha256` 为整理后文件清单，自身和 `.git/` 不纳入。
- `.gitattributes` 禁止换行转换，避免克隆后校验失效。
- 原始 `.gitignore` 会排除所有 NPZ；现规则明确纳入已打包的实验产物，原始全量数据和权重仍不纳入。
- 历史 shell 编排脚本含服务器路径，保留作来源记录；新环境请使用 README 中的 Python 命令。

## 发布前仍需补齐

- 作者及 MIT 版权权利人已按作者提供名单填写：Anlin Zhang、Shihao Li、Xibei Yu、Xinyue Xing、Jiaqi Liu。第三方代码许可归属仍需按实际来源核对。
- 上传并核对模型权重，提供真实可访问的归档链接。
- 补充整月隔离实验冻结索引、训练配置、日志、逐窗口结果、诊断及配对分析。
- 历史 `RESULTS_LEDGER.md` 未找到；不得事后新建并宣称是实验前记录。
- 已指定远程仓库：https://github.com/LeashHawk/rainnet-nowcast-bench，目前为 private，作者计划投稿时公开。投稿前将真实链接写入论文；归档 DOI 确认后再补引用信息。
- 在论文锁定环境（Python 3.10 / torch 1.12.1）跑通 CI，再创建正式发布标签。

## 远程发布

远程仓库已由作者创建，本地配置 origin 指向该地址。本次发布保持 private；不会自动修改可见性。Git 提交身份使用仓库账户 LeashHawk 及 GitHub noreply 地址，仅设置在本仓库，不代表新增论文作者。后续更新在本目录执行：

```bash
git add .
git commit -m "Prepare paper benchmark repository"
git push -u origin main
```

建议源码、固定划分和必要结果放 Git；权重、完整数据及大型实验归档使用独立存储并提供 SHA256。正式标签对应的源码与归档必须相互指向，避免论文链接随开发分支变化。
