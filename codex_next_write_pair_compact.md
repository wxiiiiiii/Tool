# Codex 下一轮执行方案：Write-Branch Pair + Skill Binder

## 1. 当前结论

这轮已经证明 **Branch-First / Soft Hierarchical 是对的**：

- Static Branch Acc All: **68.75%**
- Oracle Hard Branch-First: **78.44%**
- Oracle Soft Hierarchical: **78.75%**
- Predicted Skill + Soft: **73.44%**
- Oracle Soft `Correct Flip=10.31%`，`Harmful Flip=0.31%`
- `Unique Action Prefix` 从 0% 提升到 Oracle Soft **53.75%**

说明之前真正的问题是：**branch signal 没有进入最终 action decision**。  
现在 hierarchical interface 已经打通。

但核心剩余问题是：

```text
Write Success/Failure Pair = 0%
```

即使 Oracle Branch-First 也没有解决 write 场景的 success/failure 成对分支。

---

## 2. 下一轮主线：专攻 Write Success/Failure Pair

暂时固定：

```text
Decoder = Soft Hierarchical
```

不要再继续改 branch-to-action 接口。

重点研究：

```text
write + success -> finish
write + failure -> recover / verify
```

对同一个 prefix，必须同时满足两侧。

新增指标：

```text
Write Success Branch Acc
Write Failure Branch Acc
Write S/F Pair Acc
Write Correct Flip
Write Harmful Flip
```

---

## 3. 先做 Oracle Pair Sanity Check

直接给 Target 正确 branch：

```text
success -> finish
failure -> recover / verify
```

再用 Static Low-Rank 在 branch 内选 primitive action。

如果这个 Oracle 能显著提高 `Write S/F Pair`，说明：

> branch space 没问题，当前 Source-derived response operator 没学到足够强的 paired branch rule。

如果这个 Oracle 仍然低，则需要重新定义 write branch / branch 内 action mapping。

---

## 4. 改成 Pair-Contrastive Write Operator

不要分别学习 success 和 failure 的独立 residual。

直接学习同一 prefix 的成对差异：

\[
r_{sf}
=
b^{failure}
-
b^{success}
\]

目标：

```text
failure:
    recover / verify ↑
    finish ↓

success:
    finish ↑
    recover / verify ↓
```

训练 loss：

```python
loss_success = relu(
    margin
    - finish_score_success
    + recover_score_success
)

loss_failure = relu(
    margin
    - recover_score_failure
    + finish_score_failure
)

loss_pair = loss_success + loss_failure
```

主 loss 用 `loss_pair`，不要再用 full-logit MSE。

---

## 5. 只用 Strong Write-Critical Prefixes

当前不要把所有 critical states 混在一起。

Write Skill 训练集只保留：

```python
is_write_tool(prefix)
and source_branch(success) != source_branch(failure)
```

或 Source success/failure branch margin 差位于 top 20%。

这样避免 routine states 稀释 write recovery signal。

---

## 6. Binder 现在可以开始优化，但只在 Operator 有效后

Oracle Soft: **78.75%**  
Predicted Skill Soft: **73.44%**

存在约 **5.3 pt** 的 Skill Binding Gap。

先固定最好的 write operator，再比较：

```text
ridge binder
prototype binder
low-rank MLP binder
```

输入仍然：

```text
z_target_pre
```

标签仍然：

```text
Source-discovered functional skill
```

不使用 Target action GT。

主要看：

```text
Skill Accuracy
Skill Confidence
Oracle-vs-Predicted Branch Gap
Write S/F Pair
```

---

## 7. 默认采用 Soft Hierarchical

从当前结果看：

```text
Oracle Soft 78.75 > Oracle Hard 78.44
Predicted Soft 73.44 > Predicted Hard 72.50
```

而 harmful flip 仍接近 0。

因此下一轮默认：

```text
Soft Hierarchical
```

Hard Branch-First 保留为 ablation。

---

## 8. 下一轮实验矩阵

```text
1. Static Low-Rank
2. Oracle Correct Write Branch + Soft Hierarchical
3. Oracle Skill + Pair-Contrastive Write Operator
4. Predicted Skill + Pair-Contrastive Write Operator
5. Predicted Skill + improved Binder + Pair Operator
```

主指标：

```text
Branch Acc All
Branch Acc Write
Write Success Acc
Write Failure Acc
Write S/F Pair
Unique Action Prefix
Correct Flip
Harmful Flip
```

---

## 9. 成功标准

优先目标不是继续提高 Branch Acc All，而是：

```text
Write S/F Pair: 0% -> 明显非零
Correct Flip > Harmful Flip
Predicted Skill 保留 Oracle Skill 的大部分收益
```

达到后再进入 true closed-loop，测：

```text
Recovery@K
Success After Divergence
DB Hash
Task Success
```

## 一句话总结

> **Hierarchical branch interface 已经验证有效；下一轮不要再改 decoder，集中解决 write success/failure 的 paired branch rule，并在此基础上缩小 Oracle Skill 与 Predicted Skill 的 binding gap。**
