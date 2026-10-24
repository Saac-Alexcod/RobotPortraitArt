# 文档版本、实验口径与证据索引

> 校订日期：2026-10-24  
> 范围：文档与汇总数据可视化，不修改业务算法、前后端 API 或自动化测试。

## 1. 版本区分（重要）

| 对象 | 版本 / 时间 | 可以支持的结论 | 不可据此推断 |
|---|---|---|---|
| 历史固定栅格 E0–E5 | 开发阶段，见 `EXPERIMENTS.md` | 单个固定 raster 的 trail、stroke 与 pen-up 对比 | 多样本平均表现；艺术质量显著优越 |
| 20 张 Core Benchmark | 2026-10-19；实验提交 `0f0426a` | 40 次执行，Baseline 20/20、APD 18/20，18 对共同成功输入的汇总统计 | `v1.0.1` 重新运行过相同数据；任意人脸的泛化性能 |
| 结构保真度 V2 | 同一历史实验；评价 ROI threshold 0.50 | 18 对输入的结构边缘保持代理指标 | 主观艺术审美或 GAN 训练精度 |
| 公开 Web 软件 | Git tag `v1.0.1`，提交 `10a6b78b753b75796a2dc6343b4a71e7aa17bb0a` | 四张合成 demo、图库、历史恢复、请求耗时显示等功能 | 旧 Benchmark 自动成为新版本实验；物理机械臂可执行 |

## 2. 正式报告的数值原始来源

公开文档 **引用** 以下历史数据文件及输出路径；这些产物位于本地忽略目录 `outputs/`，**当前公开源码包中没有相应逐样本数据或人脸输入**：

```text
outputs/benchmark_faces_v1_3/
outputs/evaluation_benchmark/benchmark_rows.csv
outputs/evaluation_benchmark/benchmark_summary.json
outputs/evaluation_benchmark/failures.json
outputs/evaluation_benchmark/run_metadata.json
outputs/structural_fidelity_v2/structural_fidelity_rows.csv
outputs/structural_fidelity_v2/structural_fidelity_summary.json
outputs/structural_fidelity_v2/structural_fidelity_failures.json
```

表格与图中的数值是 **报告记录的历史实验汇总**，在 2026-10-24 的文档更新中没有重新执行原始 Benchmark；请勿将生成的新图视为第二次独立实验。若需要学术审查级复核，须在合规前提下提供**脱敏的样本清单及逐样本统计**，或在本地重跑脚本验证。

### 汇总图表数据表（仅重绘用，不是新结果）

| 指标 | Baseline | APDrawingGAN | 口径 |
|---|---:|---:|---|
| Core 成功率 | 100% (20/20) | 90% (18/20) | 20 张/方法 |
| 18 对样本 pen-down 均值（px） | 6600.94 | 17616.76 | 共同成功输入 |
| 18 对样本 pen-up 均值（px） | 5053.40 | 6085.57 | 共同成功输入 |
| 区域 Edge-F1 均值 | 0.5318 | 0.4738 | 18 对结构代理评价 |
| 固定图 pen-up（px） | E3 5980.14 | E4 5665.22 / E5 5001.77 | **单图**三阶段，不比较 Baseline/APD |

区域 F1 更细分数值见 `FINAL_REPORT.md` §12.1。图 `report_fixed_raster_penup.png`、`report_success_rate.png`、`report_paired_path_lengths.png`、`report_regional_edge_f1.png` 均根据报告的四个汇总表绘制。图表不包含置信区间，不能替代原始记录。

## 3. 测试通过数为什么不一致

- `149 passed`：固定栅格实验阶段日志。
- `173 passed`：`0f0426a` 记录的实验提交回归快照。
- `177 passed`：2026-10-24 较早交接快照。
- `187 passed`：后续图库与本地历史功能修改时的开发回归记录（**不是本轮文档校订重新执行的结果**）。

上述数字属于不同的提交/时间点，应分别保留。确认现有本机工作树的实际状态应运行：

```powershell
python -m pytest -q
npm --prefix frontend run typecheck
npm --prefix frontend run build
```

## 4. 实验和软件的能力边界

- **软件演示样本 ≠ 冻结 Benchmark**：四张公开 AI 合成人像用于方便上手；正式 Core 含 16 张固定公共数据候选与 4 张本地照片，并非同一批素材。
- **结构代理 ≠ 审美评价**：Regional Edge-F1 衡量相对于输入边缘的结构保留程度，没有专业艺术家参考图或多人盲评，不能将分数变化写成“艺术质量提升/下降”。
- **笔画顺序优化 ≠ 硬件速度保证**：轨迹单位是像素，不含实际机器人关节/速度/加速度、接触压力和物理执行误差。
- **缓存改善提交体验 ≠ GAN 推理变快**：短期 Preflight 缓存减少重复探测；报告没有通过对照实验给出可宣称的平均加速倍数。
- **不可用的原始数据 ≠ 伪造**：公开源码因体积、数据版权和隐私限制不含 CelebA 和本地真人照片；使用“报告记录值”措辞，避免声称第三方能够单击复现所有数值。

## 5. 正式提交与答辩前的作者核对

- [ ] 保存本机 `outputs/evaluation_benchmark` 与 `outputs/structural_fidelity_v2` 的只读备份（不上传私人照片）。
- [ ] 核对 Core 20 的 manifest / SHA、失败案例 `public_13` 和 `public_14`，与报告中的对应陈述一致。
- [ ] 如需宣称某项“平均显著提高”，补逐样本变化、误差区间与适当统计检验；否则表述为本样本内观察趋势。
- [ ] 展示图片时注明真实 OpenCV Baseline、APDrawingGAN 或软件 SVG 模拟，避免混淆。
- [ ] 若需要提交院系指定 Word/PDF 格式，按学校模板排版后再检查公式、图表和引用；本仓库 Markdown 不是学校正式论文排版模板。

## 6. 文献与参考链接

- Yi R et al. APDrawingGAN, CVPR 2019: <https://openaccess.thecvf.com/content_CVPR_2019/html/Yi_APDrawingGAN_Generating_Artistic_Portrait_Drawings_From_Face_Photos_With_Hierarchical_CVPR_2019_paper.html>
- Chong M J, Forsyth D. GANs N' Roses, arXiv:2106.06561: <https://arxiv.org/abs/2106.06561>
- Kim J et al. U-GAT-IT, ICLR 2020: <https://openreview.net/forum?id=BJlZ5ySKPH>
