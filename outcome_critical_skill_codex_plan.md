# Outcome-Critical Skill 下一步实施方案（Codex）

## 1. 目标
局部 Skill / Branch 指标已经提升，但最终 DB / Task Success 没有提升。下一步不再继续优化 `Write S/F Pair`，而是定位并学习真正影响最终任务结果的 Skill 决策。

## 2. Task-Level Correction Audit
比较：
```text
Static Low-Rank
vs
Source-Discovered Cluster Binder
```

按 task 分成：
```text
F -> S
S -> F
S -> S
F -> F
```

重点分析 `F -> S`：
1. 找到第一个 action / branch 不同的位置。
2. 记录 task_id、turn、observation、static/corrected skill、branch、action。
3. 标记该修正发生在第一次 DB mutation 前还是后。
4. 按以下类型统计：
```text
Information / Entity Resolution
Verify / Confirm
Commit / Execute
Recovery
Finish
```

输出每类修复 task 数和对最终 DB 的净贡献。

## 3. Outcome-Critical Oracle Decomposition
分别只 Oracle 一类 Skill：

```text
Static
+ Oracle Entity / Information
+ Oracle Verify / Commit
+ Oracle Recovery
+ Oracle Finish
+ Full Oracle Skill
```

主指标：
```text
DB Hash
Task Success
Tool Exec OK
```

只保留真正能提升最终 outcome 的 Skill 类型。

## 4. 计算 Outcome Impact
仅在 train split / environment 上做。

对候选 state `s_t`，强制不同 Skill / branch 并继续 rollout：

\[
I_t = \max_\omega J(s_t,\omega) - \min_\omega J(s_t,\omega)
\]

其中 `J` 使用 Task Success 或 DB correctness。

实现：
```python
impact[state_id] = (
    max(final_outcome_by_skill)
    - min(final_outcome_by_skill)
)
```

只保留：
```text
impact > threshold
```
作为 `Outcome-Critical Skill State`。

## 5. Outcome-Weighted Skill Training
Target Skill Binder 不再所有 state 等权：

\[
L = \sum_t w_t L_{skill,t}
\]

其中：
```python
w_t = normalize(impact[t])
```

优先训练真正有 headroom 的 Skill，如：
```text
Verify / Commit
Recovery
Entity Resolution
```

继续不使用 Target action GT，只使用 Source-derived Skill supervision。

## 6. Deployability Gap 分解
针对：
```text
Source Skill Rerank DB = 72.73%
Deployable Skill DB ≈ 63.64%
```

逐层报告：
```text
P(Target Skill = Source Skill)
P(Correct Branch | Correct Skill)
P(Correct Primitive Action | Correct Branch)
P(Tool Exec OK | Correct Action)
Final DB Hash
```

目标：明确约 9.09 个 DB points 丢在哪一层。

## 7. True Closed-Loop
仅在 Outcome-Critical Skill 确定后执行：

```text
Target history
 -> Target backbone
 -> Static + Skill policy
 -> action
 -> environment
 -> real observation
 -> append history
 -> Target backbone again
 -> next action
```

主指标：
```text
Task Success
DB Hash
Recovery@K
Success After Divergence
```

Action Acc / Branch Acc 仅作为诊断。

## 8. 执行顺序
```text
1. Task-Level Correction Audit
2. Outcome-Critical Oracle Decomposition
3. 计算 Outcome Impact
4. Outcome-Weighted Skill Training
5. Deployability Gap 分解
6. True Closed-Loop
```

## 9. Go / No-Go
继续 Skill 路线的条件：

```text
至少一类 Oracle Skill 能明显提升 DB / Task Success，
且收益集中在少量 outcome-critical states。
```

如果所有单类 Oracle 都几乎不提升最终任务结果，则停止继续优化 Skill Binder。

## 10. 一句话目标
> 从“所有 Skill 都迁移”改成“只迁移少量真正因果影响最终 outcome 的 Policy Skills”。
