# Codex 下一轮执行方案：Hierarchical Branch-First Skill

## 1. 当前结论

最新结果：

- `branch_direction_correct_rate = 96.56%`
- 但 `Oracle Source Branch Response / Oracle Skill / Predicted Skill` 都没有改变 Target top-1 action
- `Write S/F Pair = 0%`
- `Unique Action Prefix = 0%`

说明：

> **Branch-level 方向已经基本正确，当前瓶颈是 branch signal 没有真正作用到 primitive action 决策。**

所以暂时不要继续优化 Binder、K、transition 或更深 MLP。

---

## 2. 核心修改：Branch-First Decoder

不要再做：

\[
l'_a = l_a + alpha * r_{branch(a)}
\]

然后全 action argmax。

改成两层：

### 第一步：选 Branch

Branch：

```text
finish / recover / verify / acquire / execute
```

先从 Static action logits 聚合 branch logits，推荐：

\[
b_j =
logsumexp(l_a, a in branch_j) - log(|branch_j|)
\]

然后：

\[
b'_j = b_j + alpha * r_j
\]

选择：

```python
selected_branch = argmax(corrected_branch_logits)
```

### 第二步：Branch 内选 primitive action

```python
candidate_actions = actions_in_branch(selected_branch)
action = argmax(static_logits[candidate_actions])
```

分工：

> **Skill / Branch 决定高层行为方向，Static Low-Rank 决定 branch 内具体 action。**

---

## 3. 必跑 Oracle 实验

### A. Oracle Branch-First

直接使用 held-out Source branch response：

```text
Static Low-Rank
vs
Oracle Branch + old broadcast
vs
Oracle Branch + hard branch-first
```

重点指标：

```text
Branch Acc Write
Write S/F Pair
Unique Action Prefix
Correct Flip
Harmful Flip
```

如果 Branch-First 能把 `Write S/F Pair` 从 0% 明显拉高，说明之前失败主要是 branch→action 接口问题。

### B. Oracle Skill + Train Branch Operator

使用 Oracle Source Skill ID，但 operator 只能从 train split 学。

### C. Predicted Skill + Train Branch Operator

再替换成：

```python
q(skill | z_target_pre)
```

只有 B 有效后，才值得优化 Binder。

---

## 4. Soft Hierarchical 版本

Hard Branch-First 有效果后，再实现：

\[
p(a|s,o)
=
p(branch(a)|s,o)
*
p(a|branch(a),s)
\]

即：

```python
branch_log_probs = log_softmax(corrected_branch_logits)

within_branch_log_probs = log_softmax(
    static_logits[branch_actions]
)

final_action_log_prob =
    branch_log_prob
    + within_branch_log_prob
```

对比：

```text
hard branch-first
soft hierarchical
old broadcast
```

---

## 5. 收紧 Critical Prefix

现在：

```text
critical_prefix_rate = 90%
source_branch_flip_rate = 12.5%
```

筛选太宽。

改成：

```python
strong_critical =
    source_branch(success) != source_branch(failure)
```

再补充：

```text
branch-margin change top 20%~25%
```

只用：

```text
Strong Critical + Top-Quantile Soft Critical
```

做 Skill clustering。

---

## 6. 扩大 Counterfactual 数据

每个 prefix 不要只有一条 success/failure。

改成每类：

```text
3~5 个 schema-compatible observation variants
```

仅从 Source train 或 simulator 获取。

对同一 observation type 求平均 response，减少模板偶然性。

---

## 7. 新增两层错误诊断

必须输出：

```text
static_branch
corrected_branch
gold/oracle_branch

static_action
corrected_action

Branch Selection Accuracy
Branch Correct Flip
Branch Harmful Flip
Within-Branch Action Accuracy
Primitive Correct Flip Given Branch Correct
```

目的是分清：

- branch 选错
- branch 选对但 primitive action 错
- branch 被错误修改
- branch 修正后 primitive action 是否真正改善

---

## 8. 暂时不要做

先不要优先：

```text
Binder architecture sweep
K sweep
transition T_omega
deeper MLP
full-logit residual alpha sweep
closed-loop
```

开发顺序：

```text
Oracle Branch-First
→ Oracle Skill + Branch Operator
→ Predicted Skill + Branch Operator
→ 再优化 Binder
→ 最后 closed-loop
```

---

## 9. 下一轮成功标准

优先看：

```text
Write S/F Pair > 0
Unique Action Prefix > 0
Branch Correct Flip > Harmful Flip
```

如果 Oracle Branch-First 都无法做到这些，就说明当前 branch definition / Source branch signal 仍需修改。

---

## 10. 一句话目标

> **Functional Skill 负责选择 behavioral branch，Static Low-Rank 负责在 branch 内选择 primitive action。**

这比继续把 branch residual 直接 broadcast 到所有 action logits 更符合当前实验结果。
