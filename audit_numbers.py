#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_numbers.py — 交付物自检：核心数字独立回算 + 文档对账

为什么单独一个脚本：
    interview_no_show.py 负责分析，本脚本负责**质疑它**。这里不 import 它，
    所有关键数字用 pandas 独立重算一遍——若两处用同一段代码，同一个错误会被
    复用两次，等于没查（交付文档里那个凭印象写错的均值就是这么漏过去的）。

检查三类：
    A. 核心数字回算：从 data/ 自己算一遍，再要求它同时出现在 output/report.txt
       与 README/AI_LOG 里（少一处就说明文档与产物已经漂移）
    B. 结构对账：文档里**当前声明**的"条数/图数"与产物实际数量比对
       （历史改动记录里被引用来说明的旧值不参与判定，只列进第三类清单）
    C. 数字 token 集合差：文档里出现、report.txt 里算不出来的数字，列出来供人工复核

用法:
    python interview_no_show.py     # 先产出 output/report.txt 与图
    python audit_numbers.py         # 再自检（退出码 0 = 全部通过）

只用发放数据，无外部依赖（scipy 不参与，2x2 检验自行按同一超几何定义重算）。
"""

import math
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/interview_invitations.csv")
REPORT = Path("output/report.txt")
DOCS = [Path("README.md"), Path("AI_LOG.md")]
CLIFF_H = 0.30                    # 18 分钟
COST_NO_SHOW_H = 90.0
COST_CONFIRM_H = 20.0
BREAKEVEN = COST_CONFIRM_H / COST_NO_SHOW_H

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


def auc(score, y):
    r = pd.Series(score).rank().to_numpy()
    y = np.asarray(y)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fisher_two_sided(a, b, c, d):
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c

    def pr(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    obs = pr(a)
    tot = 0.0
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        pv = pr(x)
        if pv <= obs * (1 + 1e-9) + 1e-15:
            tot += pv
    return min(1.0, tot)


# ---------------- 读取产物与文档 ----------------
missing = [p for p in [REPORT] + DOCS if not p.exists()]
if missing:
    print("缺少文件：%s" % ", ".join(str(p) for p in missing))
    sys.exit(2)

rep = REPORT.read_text(encoding="utf-8")
docs = {p.name: p.read_text(encoding="utf-8") for p in DOCS}
doc_all = "\n".join(docs.values())


def norm(t):
    """去掉千分位、统一负号（文档里用的是排版减号 U+2212），便于数字串比对。"""
    return t.replace(",", "").replace("\u3000", " ").replace("\u2212", "-")


rep_n, doc_n = norm(rep), norm(doc_all)


def check_number(name, s):
    """要求某个数字串在 report.txt 和文档里都出现。"""
    in_rep, in_doc = s in rep_n, s in doc_n
    check("%s：%s" % (name, s), in_rep and in_doc,
          "report=%s doc=%s" % ("有" if in_rep else "缺", "有" if in_doc else "缺"))
    return in_rep and in_doc


# ---------------- 独立回算 ----------------
raw = pd.read_csv(SRC)
df = raw.drop_duplicates().copy()
miss = int(df.is_completed.isna().sum())
d = df.dropna(subset=["is_completed"]).copy()
d["y"] = 1 - d.is_completed.astype(int)
d["stale"] = (pd.to_datetime(d.invited_at, utc=True)
              - pd.to_datetime(d.profile_synced_at, utc=True)).dt.total_seconds() / 3600.0

n_all, tot = len(d), int(d.y.sum())
rate_all = 100 * tot / n_all
lo, hi = d[d.stale <= CLIFF_H], d[d.stale > CLIFF_H]
rate_lo, rate_hi = 100 * lo.y.mean(), 100 * hi.y.mean()

# 免疫部门：仍按脚本的口径（陈旧通道内 n>=50 且爽约率最低）独立选出
st = hi
g = st.groupby("department").y.agg(["mean", "size"])
cand = g[g["size"] >= 50].sort_values("mean")
dept = str(cand.index[0])
a = st[st.department == dept]
b = st[st.department != dept]
k_a, n_a, k_b, n_b = int(a.y.sum()), len(a), int(b.y.sum()), len(b)
p_f = fisher_two_sided(k_a, n_a - k_a, k_b, n_b - k_b)

# 推荐规则（陈旧>18min 且非免疫部门）与三个对照口径
m = (d.stale > CLIFF_H) & (d.department != dept)
net_rec = int(d.y[m].sum()) * COST_NO_SHOW_H - int(m.sum()) * COST_CONFIRM_H
net_con = int(hi.y.sum()) * COST_NO_SHOW_H - len(hi) * COST_CONFIRM_H      # 保守版
net_all = tot * COST_NO_SHOW_H - n_all * COST_CONFIRM_H                    # 全量

print("=" * 68)
print("独立回算结果（本脚本自己算的，未 import 分析脚本）")
print("=" * 68)
print("  有效邀约 %d  爽约 %d  整体 %.2f%%" % (n_all, tot, rate_all))
print("  ≤18min n=%d %.2f%%   >18min n=%d %.2f%%" % (len(lo), rate_lo, len(hi), rate_hi))
print("  免疫部门 = %s（陈旧通道内 n=%d %.1f%%）vs 其他 n=%d %.1f%%  Fisher p=%.2g"
      % (dept, n_a, 100 * k_a / n_a, n_b, 100 * k_b / n_b, p_f))
print("  净收益：全量 %+d h ｜ 保守版 %+d h ｜ 推荐 %+d h" % (net_all, net_con, net_rec))
print("-" * 68)

# A. 核心数字回算 -> 同时要求在 report.txt 与文档中出现
check_number("有效条数", str(n_all))
check_number("爽约条数", str(tot))
check_number("整体爽约率", "%.2f%%" % rate_all)
check_number("新鲜通道 n", str(len(lo)))
check_number("陈旧通道 n", str(len(hi)))
check_number("新鲜通道爽约率", "%.2f%%" % rate_lo)
check_number("陈旧通道爽约率", "%.2f%%" % rate_hi)
for col, s_ in [("intent_score", d.intent_score), ("pool_size", d.pool_size),
                ("stale_h", d.stale)]:
    check_number("AUC(%s)" % col, "%.3f" % auc(s_, d.y))
check_number("免疫部门陈旧通道 n", str(n_a))
check_number("免疫部门爽约率", "%.1f%%" % (100 * k_a / n_a))
check_number("其余部门 n", str(n_b))
check_number("其余部门爽约率", "%.1f%%" % (100 * k_b / n_b))
check_number("净收益(全量)", "%+d" % net_all)
check_number("净收益(保守版)", "%+d" % net_con)
check_number("净收益(推荐)", "%+d" % net_rec)
check_number("保本率", "%.2f%%" % (BREAKEVEN * 100))

# B. 结构对账：文档声明的数量 vs 产物实际数量
rule_lines = re.findall(r"^  [①②③④⑤⑥⑦⑧⑨].*?净收益\s*([+-]?\d+) h", rep, re.M)
n_rules_real = len(rule_lines)
declared = re.findall(r"(\d+)\s*条候选规则", doc_all)
check("候选规则条数：文档声明 %s vs 产物实际 %d" % (declared or "未声明", n_rules_real),
      declared and all(int(x) == n_rules_real for x in declared),
      "文档需写成 %d 条（逐条规则行见 report.txt 第五节）" % n_rules_real)

figs = sorted(p.name for p in Path("output").glob("fig_*.png"))
py_doc = Path("interview_no_show.py").read_text(encoding="utf-8")
decl_py = re.findall(r"(\d+)\s*张图", py_doc)
decl_rd = re.findall(r"\+\s*(\d+)\s*张图", docs.get("README.md", ""))
check("图表数量：脚本声明 %s / README 产出声明 %s vs 实际 %d 个"
      % (decl_py or "未声明", decl_rd or "未声明", len(figs)),
      bool(decl_py) and bool(decl_rd) and all(int(x) == len(figs) for x in decl_py + decl_rd),
      "实际：%s" % ", ".join(figs))

# C. 数字 token 集合差（人工复核清单，不直接判失败）
rep_tokens = set(re.findall(r"\d+(?:\.\d+)?", rep_n))
doc_tokens = set(re.findall(r"\d+(?:\.\d+)?", norm("\n".join(docs.values()))))
only_doc = sorted(t for t in doc_tokens - rep_tokens if len(t) >= 3)
print("文档里出现、report.txt 里算不出来的数字 token（%d 个，人工复核；"
      "日期/条目号/commit 号属正常）：" % len(only_doc))
print("  " + (" ".join(only_doc[:40]) if only_doc else "无"))
print("-" * 68)

# ---------------- 汇总 ----------------
bad = [c for c in CHECKS if not c[1]]
for name, ok, detail in CHECKS:
    print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  ← %s" % detail) if (detail and not ok) else ""))
print("=" * 68)
print("共 %d 项，通过 %d 项，失败 %d 项" % (len(CHECKS), len(CHECKS) - len(bad), len(bad)))
sys.exit(1 if bad else 0)
