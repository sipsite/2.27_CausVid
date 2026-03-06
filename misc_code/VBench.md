# 在 VBench 上做 eval、看哪些指标

根据 VBench 官方文档和你当前 CausVid 项目，整理成下面这份「在 VBench 上做 eval、看哪些指标」的说明。

---

## 1. 在 VBench 上做 eval 的流程

### 安装 VBench

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install vbench
# 部分维度需要 detectron2
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
```

或克隆仓库：

```bash
git clone https://github.com/Vchitect/VBench.git
cd VBench && pip install .
```

并把 [VBench_full_info.json](https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json) 放到运行目录（用标准 prompt 评测时需要）。

### 和你当前项目的衔接

你现在已经在用 `autoregressive_inference.py` + `MovieGenVideoBench_*.txt` 生成视频并写到 `--output_folder`。要上 VBench 有两种用法：

**方式 A：评估「自己生成的视频」（自定义视频）**

不要求用官方 prompt、不要求文件名格式，只要把「视频所在目录」或「单个视频路径」给 VBench 即可：

```bash
# 单维度
python evaluate.py \
  --dimension subject_consistency \
  --videos_path /path/to/your/output_folder/ \
  --mode=custom_input

# 或使用 CLI
vbench evaluate \
  --dimension subject_consistency \
  --videos_path /path/to/your/output_folder/ \
  --mode=custom_input
```

`custom_input` 下支持的维度有：  
`subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`。

多卡可以：

```bash
torchrun --nproc_per_node=${GPUS} --standalone evaluate.py ... 
# 或
vbench evaluate --ngpus=${GPUS} ...
```

**方式 B：按 VBench 标准 prompt 评测（可打榜）**

1. 从 VBench 仓库的 `prompts/` 拿到官方 prompt 列表，用 CausVid 按这些 prompt 生成视频。  
2. 视频命名/目录结构需符合 VBench 约定：  
   `vbench_videos/{model}/{dimension}/{prompt}-{index}.mp4`（详见官方 [prompts 说明](https://github.com/Vchitect/VBench/tree/master/prompts)）。  
3. 然后按标准流程跑各维度并算总分（见下）。

若要做 **VBench-Long**（长视频，CausVid 论文里 84.27 的那个），需要用仓库里的 `vbench2_beta_long`：里面有 `eval_long.py`、`evaluate_long.sh` 和对应的 16 个维度脚本，流程类似，只是针对长视频的配置和 prompt。

---

## 2. 要看哪些指标

### 16 个维度（逐项看）

VBench 把「视频生成质量」拆成 16 个维度，**分数越高越好**：

| 类别 | 维度 |
|------|------|
| **质量相关** | `subject_consistency`（主体一致性）, `background_consistency`（背景一致性）, `temporal_flickering`（时序闪烁）, `motion_smoothness`（运动平滑）, `dynamic_degree`（动态程度）, `aesthetic_quality`（美学质量）, `imaging_quality`（成像质量） |
| **语义相关** | `object_class`, `multiple_objects`, `human_action`, `color`, `spatial_relationship`, `scene`, `temporal_style`, `appearance_style`, `overall_consistency` |

跑全量时可以一次指定多个维度，或写脚本循环调用；每个维度会输出该维度的得分。

### 汇总分数（打榜 / 总览用）

官方用 **Quality Score**、**Semantic Score** 和 **Total Score** 做总览和排行榜：

- **Quality Score**：上面 7 个质量相关维度的加权平均（先按维度的 min/max 做归一化）。
- **Semantic Score**：其余 9 个语义维度的加权平均（同样先归一化）。
- **Total Score**：`Total Score = w1 * Quality Score + w2 * Semantic Score`，即论文/Leaderboard 用的「总分」。

计算方式在 VBench 仓库的 `scripts/constant.py` 里（各维度 min/max、权重等）。本地算总分可以用：

```bash
cd evaluation_results
zip -r ../evaluation_results.zip .
python scripts/cal_final_score.py --zip_file {path_to_evaluation_results.zip} --model_name {your_model_name}
```

这样会得到 Total / Quality / Semantic 三个汇总指标，便于和论文/Leaderboard 对比。

### 评测前注意（temporal_flickering）

评 **temporal_flickering** 前需要先过滤静态视频：

```bash
python static_filter.py --videos_path $VIDEOS_PATH
```

否则静态视频会干扰该维度结果。

---

## 3. 小结（你要做的事）

1. **安装 VBench**（pip 或 clone），需要时装 detectron2。  
2. **用 CausVid 生成视频**：  
   - 自定义评测：继续用你现在的 `output_folder` 即可。  
   - 标准/VBench-Long：用 VBench 或 vbench2_beta_long 的 prompt 列表生成，并按要求组织目录和文件名。  
3. **跑维度**：对 `--videos_path` 用 `evaluate.py` / `vbench evaluate`（自定义就用 `--mode=custom_input`）。  
4. **看指标**：  
   - 细看：16 个维度的分数；  
   - 总览/对比论文：Quality Score、Semantic Score、Total Score（用 `scripts/cal_final_score.py`）。  
5. 若做长视频：用 `vbench2_beta_long` 下的 `eval_long.py` / `evaluate_long.sh`，同样看各维度和汇总分数。

你当前 `jobs_sh/inference.sh` 里已经用 `MovieGenVideoBench_128_ys.txt` 生成到 `jobs_sh/inference_output`，要快速看一批指标，可以直接把 `--videos_path` 指到 `jobs_sh/inference_output`，用 `--mode=custom_input` 先跑上面 6 个支持的维度，再根据需要上全 16 维或 VBench-Long。
