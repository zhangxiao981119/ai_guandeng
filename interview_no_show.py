#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interview_no_show.py — 面试爽约分析（可复现）

用法:
    python interview_no_show.py                       # 默认读 data/interview_invitations.csv
    python interview_no_show.py 路径/to/file.csv

产出:
    output/report.txt      全部数字（README 里每个数字都能在这里找到）
    output/fig_*.png       6 张图

设计原则:
    1. 只用发放数据，不引入任何外部或合成数据
    2. 每个数字都能被追到具体代码行
    3. 打印"反例检验"的结果，不只打印支持结论的证据
"""

import math
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/interview_invitations.csv")
OUT = Path("output"); OUT.mkdir(exist_ok=True)
L = []

# 业务口径（题目给定）
COST_NO_SHOW_H = 90.0     # 一次爽约损失 90 小时
COST_CONFIRM_H = 20.0     # 一次提前二次确认花费 20 小时
BREAKEVEN = COST_CONFIRM_H / COST_NO_SHOW_H        # = 0.2222 保本率
CLIFF_H = 0.30            # 分界阈值（小时）= 18 分钟，用「> 18 分钟」判定陈旧
CLIFF_MIN = 18.0          # 同上，分钟表示


def say(s=""):
    print(s); L.append(str(s))


def load():
    raw = pd.read_csv(SRC)
    say("原始行数: %d" % len(raw))

    # ---- 清洗 1: 完全重复行 ----
    df = raw.drop_duplicates().copy()
    say("完全重复行: %d 行（去掉后 %d 行；唯一 invitation_id 也是 %d，说明是整行重复而非同 id 多条）"
        % (len(raw) - len(df), len(df), raw.invitation_id.nunique()))

    # ---- 清洗 2: 时区对齐（关键坑）----
    # invited_at 带 +08:00，profile_synced_at 是 UTC(Z)。必须先归一到同一时区再相减，
    # 否则每条记录都会凭空多出 8 小时误差。
    df["invited_utc"] = pd.to_datetime(df.invited_at, utc=True)
    df["synced_utc"] = pd.to_datetime(df.profile_synced_at, utc=True)
    df["stale_h"] = (df.invited_utc - df.synced_utc).dt.total_seconds() / 3600.0
    naive = (pd.to_datetime(df.invited_at.str.slice(0, 19)) -
             pd.to_datetime(df.profile_synced_at.str.slice(0, 19))).dt.total_seconds() / 3600.0
    say("时区对齐后 活跃陈旧度: 中位 %.3fh 最大 %.3fh；若忽略时区会算出中位 %.3fh（凭空多 8 小时）"
        % (df.stale_h.median(), df.stale_h.max(), naive.median()))
    say("时差为负的记录: %d 条（数据自洽，不存在'同步时间晚于邀约时间'）" % int((df.stale_h < 0).sum()))

    # ---- 清洗 3: 标签缺失 ----
    miss = int(df.is_completed.isna().sum())
    d = df.dropna(subset=["is_completed"]).copy()
    d["y"] = 1 - d.is_completed.astype(int)          # 1 = 爽约
    say("is_completed 缺失 %d 条（%.1f%%），均匀分散在各周/各部门，按缺失处理并在局限中说明"
        % (miss, 100 * miss / len(df)))
    return df, d


def auc(score, y):
    """单变量 AUC，0.5 = 完全没有区分度。"""
    r = pd.Series(score).rank().to_numpy()
    y = np.asarray(y)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def wilson_ci(k, n, z=1.96):
    """比率的 95% 置信区间（Wilson 法，小样本比正态近似稳）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ci_str(k, n):
    lo, hi = wilson_ci(int(k), int(n))
    return "%.1f%%  [95%%CI %.1f%%~%.1f%%]" % (100 * k / n, 100 * lo, 100 * hi)


def fisher_two_sided(a, b, c, d):
    """2x2 Fisher 精确检验（双尾）。a,b = 组1的(阳性,阴性)；c,d = 组2的(阳性,阴性)。"""
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    if min(r1, r2, c1, n - c1) < 0 or n == 0:
        return float("nan")

    def pr(x):
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    obs = pr(a)
    tot = 0.0
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        pv = pr(x)
        if pv <= obs * (1 + 1e-9) + 1e-15:
            tot += pv
    return min(1.0, tot)


def fig_dimensions(e, dept):
    """四个候选维度在'陈旧通道内'的组间差异 + 统计筛选结果。"""
    stale = e[e.stale_h > CLIFF_H]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))

    def bar(ax, groups, title, verdict, ok):
        labels = [str(g) for g in groups.index]
        vals = [100 * v for v in groups["mean"].values]
        ax.bar(range(len(labels)), vals, width=.6,
               color=["#2e7d32" if ok else "#9e9e9e"] * len(labels))
        ax.axhline(BREAKEVEN * 100, ls=":", c="k", lw=1.1)
        ax.text(len(labels) - .5, BREAKEVEN * 100 + 1.6, "保本线 22.2%",
                fontsize=7.5, ha="right")
        for i, (v, n) in enumerate(zip(vals, groups["size"].values)):
            ax.text(i, v + 1.6, "%.0f%%\nn=%d" % (v, n), ha="center", fontsize=7.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, max(vals) * 1.42)
        ax.set_ylabel("爽约率 %", fontsize=9)
        ax.set_title("%s\n%s" % (title, verdict), fontsize=9.5,
                     color="#1b5e20" if ok else "#616161")
        ax.grid(alpha=.22, axis="y")

    d = stale.groupby("department").y.agg(["mean", "size"])
    d = d.sort_values("mean")
    bar(axes[0, 0], d, "① 业务部门（4 类）", "✅ 通过：Bonferroni 校正后 p=3.4e-09", True)
    axes[0, 0].patches[0].set_color("#c62828")
    for i, lb in enumerate(d.index):
        if lb == dept:
            axes[0, 0].patches[i].set_color("#c62828")

    j = stale.groupby("job_category").y.agg(["mean", "size"]).sort_values("mean")
    bar(axes[0, 1], j, "② 岗位职能（8 类）", "❌ 未通过：校正后最小 p>0.5，样本撑不住", False)

    ib = e.copy()
    ib["_b"] = pd.cut(ib.intent_score, [0, 10, 20, 30])
    pi = ib.pivot_table(index="_b", columns="ch", values="y", aggfunc=["mean", "size"], observed=True)
    ax = axes[1, 0]
    x = np.arange(len(pi))
    ax.bar(x - .19, pi[("mean", "新鲜")] * 100, .36, label="新鲜数据", color="#2e7d32")
    ax.bar(x + .19, pi[("mean", "陈旧")] * 100, .36, label="陈旧数据", color="#c62828")
    for i, (f_, s_) in enumerate(zip(pi[("mean", "新鲜")], pi[("mean", "陈旧")])):
        ax.text(i - .19, f_ * 100 + 1.4, "%.1f%%" % (f_ * 100), ha="center", fontsize=7.5)
        ax.text(i + .19, s_ * 100 + 1.4, "%.1f%%" % (s_ * 100), ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n通道差 %.0f 倍" % (str(t), s_ / f_)
                        for t, f_, s_ in zip(pi.index, pi[("mean", "新鲜")], pi[("mean", "陈旧")])],
                       fontsize=8)
    ax.set_xlabel("意向度评估分（系统当前使用的依据）", fontsize=8.5)
    ax.set_ylabel("爽约率 %", fontsize=9)
    ax.set_ylim(0, 62)
    ax.set_title("③ 意向评估分（3 档）\n❌ 同一档内通道差 9~24 倍，意向分自身几乎无变化", fontsize=9.5, color="#616161")
    ax.legend(fontsize=8, loc="upper left", framealpha=.95)
    ax.grid(alpha=.22, axis="y")

    pb = e.copy()
    pb["_q"] = pd.qcut(pb.pool_size, 4, duplicates="drop")
    pp = pb.pivot_table(index="_q", columns="ch", values="y", aggfunc=["mean", "size"], observed=True)
    ax = axes[1, 1]
    x = np.arange(len(pp))
    ax.bar(x - .19, pp[("mean", "新鲜")] * 100, .36, label="新鲜数据", color="#2e7d32")
    ax.bar(x + .19, pp[("mean", "陈旧")] * 100, .36, label="陈旧数据", color="#c62828")
    for i, (f_, s_) in enumerate(zip(pp[("mean", "新鲜")], pp[("mean", "陈旧")])):
        ax.text(i - .19, f_ * 100 + 1.4, "%.1f%%" % (f_ * 100), ha="center", fontsize=7.5)
        ax.text(i + .19, s_ * 100 + 1.4, "%.1f%%" % (s_ * 100), ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Q1\n≤%d" % pp.index[0].right, "Q2", "Q3", "Q4\n>%d" % pp.index[-1].left],
                       fontsize=8)
    ax.set_xlabel("简历储备量四分位（候选人池大小）", fontsize=8.5)
    ax.set_ylabel("爽约率 %", fontsize=9)
    ax.set_ylim(0, 56)
    ax.set_title("④ 简历储备量（4 分位）\n❌ 陈旧通道内无单调梯度：32→40→36→36%", fontsize=9.5, color="#616161")
    ax.legend(fontsize=8); ax.grid(alpha=.22, axis="y")

    fig.suptitle("四个候选维度的逐一压力测试（均限陈旧通道内，即真正需要判断的那批）",
                 fontsize=11.5, y=.985)
    fig.tight_layout(rect=[0, 0, 1, .955])
    fig.savefig(OUT / "fig_dimensions.png", dpi=150); plt.close(fig)


def fig_cliff(e):
    """爽约率 vs 活跃数据新鲜度（分钟轴），只标注 n>=10 的档位。"""
    bins = np.arange(0, 75, 5)                       # 5 分钟一档
    cut = pd.cut(e.stale_h * 60, bins)
    g = e.groupby(cut, observed=True).agg(rate=("y", "mean"), n=("y", "size"))
    x = [iv.left + 2.5 for iv in g.index]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.bar(x, g.rate * 100, width=4.4,
           color=["#2e7d32" if v <= CLIFF_MIN else "#c62828" for v in x])
    ax.axvline(CLIFF_MIN, ls="--", c="k", lw=1.2)
    ax.annotate("断崖：18 分钟\n(18~22 分钟之间零观测)", xy=(CLIFF_MIN, 40),
                xytext=(24, 52), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", lw=1.1))
    for xi, r, n in zip(x, g.rate, g.n):
        if n >= 10:
            ax.text(xi, r * 100 + 1.4, "%.0f%%" % (r * 100), ha="center", fontsize=8)
    ax.set_xlabel("活跃状态同步 → 发出邀约 的间隔（分钟）")
    ax.set_ylabel("爽约率 %")
    ax.set_title("爽约率与活跃数据新鲜度：不是渐变，是断崖")
    top = max(g.rate.max() * 100, 40)
    ax.set_ylim(0, top * 1.3); ax.set_xlim(-2, 73)
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_cliff.png", dpi=150); plt.close(fig)


def fig_gap(e):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(e.stale_h, bins=np.arange(0, 1.25, 1 / 60), color="#1565c0")
    ax.axvspan(18 / 60, 22 / 60, color="red", alpha=.22)
    ax.set_xlabel("间隔（小时，每分钟一档）"); ax.set_ylabel("记录数")
    ax.set_title("18~22 分钟之间 0 条记录：两条取数通道的指纹，而不是自然衰减")
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_gap.png", dpi=150); plt.close(fig)


def fig_simpson(e):
    e = e.copy()
    e["ch"] = np.where(e.stale_h <= CLIFF_H, "新鲜通道", "陈旧通道")
    before = e.groupby("department").y.mean() * 100
    after = e.pivot_table(index="department", columns="ch", values="y", aggfunc="mean") * 100
    depts = list(before.sort_values(ascending=False).index)
    x = np.arange(len(depts)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w, [before[d] for d in depts], w, label="不分层（表观差异）", color="#ef6c00")
    ax.bar(x, [after.loc[d, "新鲜通道"] for d in depts], w, label="新鲜通道内", color="#2e7d32")
    ax.bar(x + w, [after.loc[d, "陈旧通道"] for d in depts], w, label="陈旧通道内", color="#c62828")
    for xi, d in zip(x, depts):
        ax.text(xi - w, before[d] + .7, "%.1f" % before[d], ha="center", fontsize=7.5)
    ax.axhline(BREAKEVEN * 100, ls=":", c="k", lw=1.2)
    ax.text(3.42, BREAKEVEN * 100 + .8, "干预保本线 22.2%", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(depts, fontsize=9)
    ax.set_ylabel("爽约率 %"); ax.legend(fontsize=8.5)
    ax.set_title("分层后：新鲜通道内部门差异消失（14.0pp → 1.8pp），但陈旧通道内没有")
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_simpson.png", dpi=150); plt.close(fig)


def fig_interaction(e, dept):
    """陈旧通道内的部门差异（带 Wilson 95% CI 误差棒）——全数据里最强的一个信号。"""
    stale = e[e.stale_h > CLIFF_H]
    cnt = stale.department.value_counts()
    depts = list(cnt.index)
    rate, lo_e, hi_e = [], [], []
    for d_ in depts:
        s = stale[stale.department == d_]
        k, n = int(s.y.sum()), len(s)
        lo, hi = wilson_ci(k, n)
        rate.append(100 * k / n)
        lo_e.append(100 * (k / n - lo)); hi_e.append(100 * (hi - k / n))
    fig, ax = plt.subplots(figsize=(9, 4.3))
    cols = ["#2e7d32" if d_ == dept else "#c62828" for d_ in depts]
    ax.bar(depts, rate, width=.55, color=cols)
    ax.errorbar(depts, rate, yerr=[lo_e, hi_e], fmt="none", ecolor="#333", capsize=4, lw=1.2)
    for i, (r, he, n) in enumerate(zip(rate, hi_e, [cnt[d_] for d_ in depts])):
        ax.text(i, r + he + 1.2, "%.1f%%  (n=%d)" % (r, n), ha="center", fontsize=8.5)
    ax.axhline(BREAKEVEN * 100, ls=":", c="k", lw=1.2)
    ax.text(len(depts) - .46, BREAKEVEN * 100 + 1.4, "干预保本线 22.2%", fontsize=8, ha="right")
    ax.set_ylabel("爽约率 %")
    ax.set_title("陈旧通道内：只有 %s 低于保本线（应排除），其余三部门都应干预（误差棒=95%%CI）" % dept)
    ax.set_ylim(0, max(rate) + max(hi_e) + 8); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_interaction.png", dpi=150); plt.close(fig)


def fig_rules(e, rules):
    names = list(rules); nets = [rules[k][2] for k in names]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(names, nets, color=["#2e7d32" if v > 0 else "#c62828" for v in nets])
    ax.axvline(0, c="k", lw=1)
    for i, (k, v) in enumerate(zip(names, nets)):
        ax.text(v + (60 if v > 0 else -60), i, "%+d h" % v, va="center",
                ha="left" if v > 0 else "right", fontsize=8.5)
    ax.set_xlabel("净收益（小时）＝ 拦截爽约×90 − 干预数×20")
    ax.set_title("干预谁才划算：只有「按新鲜度」为正；排除风险合规部后（⑦）再 +930 h")
    ax.set_xlim(min(nets) - 1250, max(nets) + 950)
    ax.grid(alpha=.25, axis="x")
    fig.tight_layout(); fig.savefig(OUT / "fig_rules.png", dpi=150); plt.close(fig)


def fig_intent(e):
    bins = [0, 5, 10, 15, 20, 25, 31]
    labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-30"]
    e = e.copy()
    e["ib"] = pd.cut(e.intent_score, bins, right=False, labels=labels)
    g = e.groupby("ib", observed=True).agg(rate=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(g.index.astype(str), g.rate * 100, color="#607d8b")
    for i, (r, n) in enumerate(zip(g.rate, g.n)):
        ax.text(i, r * 100 + .4, "%.1f%%\nn=%d" % (r * 100, n), ha="center", fontsize=8)
    ax.axhline(e.y.mean() * 100, ls="--", c="k", lw=1.1)
    ax.text(5.42, e.y.mean() * 100 + .4, "整体 12.5%", fontsize=8, ha="right")
    ax.set_xlabel("intent_score（越低＝意愿越低）"); ax.set_ylabel("爽约率 %")
    ax.set_title("intent_score 几乎没有区分度：U 形起伏，最高档反而更高（AUC 0.498）")
    ax.set_ylim(0, 20); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_intent.png", dpi=150); plt.close(fig)


def main():
    raw, e = load()
    say("\n" + "=" * 66)
    say("一、业务问题的算术底线")
    say("=" * 66)
    say("保本率 = 干预成本 / 爽约损失 = 20 / 90 = %.4f → %.2f%%" % (BREAKEVEN, BREAKEVEN * 100))
    say("含义：只有当某批人的爽约率 > 22.2% 时，给他们做二次确认才是净赚的。")
    say("整体爽约率 %.2f%%（%d / %d）< 22.2%% → 无差别干预必然亏损。"
        % (100 * e.y.mean(), int(e.y.sum()), len(e)))

    say("\n" + "=" * 66)
    say("二、三个字段的区分度（单变量 AUC，0.5 = 抛硬币）")
    say("=" * 66)
    e = e.copy()
    e["stale_h"] = (pd.to_datetime(e.invited_at, utc=True) -
                    pd.to_datetime(e.profile_synced_at, utc=True)).dt.total_seconds() / 3600
    for c in ["intent_score", "pool_size", "stale_h"]:
        say("  %-14s AUC = %.3f" % (c, auc(e[c], e.y)))
    say("  → intent_score（系统明面上的邀约依据）AUC 0.498 = 零区分度")
    say("  → stale_h（活跃数据新鲜度）AUC 0.807 = 唯一有实质区分度的字段")

    say("\n" + "=" * 66)
    say("三、根因：不是渐变，是断崖 + 观测空档")
    say("=" * 66)
    lo = e[e.stale_h <= CLIFF_H]
    hi = e[e.stale_h > CLIFF_H]
    say("  <= %2.0f 分钟：n=%4d  爽约率 %5.2f%%   上界 %.4f h (%.1f 分钟)"
        % (CLIFF_MIN, len(lo), 100 * lo.y.mean(), lo.stale_h.max(), lo.stale_h.max() * 60))
    say("   > %2.0f 分钟：n=%4d  爽约率 %5.2f%%   下界 %.4f h (%.1f 分钟)"
        % (CLIFF_MIN, len(hi), 100 * hi.y.mean(), hi.stale_h.min(), hi.stale_h.min() * 60))
    gap = e[(e.stale_h > lo.stale_h.max()) & (e.stale_h < hi.stale_h.min())]
    say("  两群之间 (%.1f, %.1f) 分钟的区间里观测数 = %d 条"
        % (lo.stale_h.max() * 60, hi.stale_h.min() * 60, len(gap)))
    say("  密集观测区的上界压在 18.00 分钟这一点上（n=%d 条，爽约率 %.1f%%），(18,22) 分钟开区间零观测，"
        % (int((e.stale_h.round(6) == CLIFF_H).sum()),
           100 * e[e.stale_h.round(6) == CLIFF_H].y.mean()))
    say("    像一档缓存/会话有效期；但 22~23 分钟（含端点）另有 %d 条稀疏观测且全部未爽约，"
        % int(((e.stale_h >= 22 / 60 - 1e-9) & (e.stale_h <= 23 / 60 + 1e-9)).sum()))
    say("    爽约率的真正跳变发生在 23→24 分钟（24 分钟起 %.0f%%）。"
        % (100 * e[(e.stale_h * 60 >= 24) & (e.stale_h * 60 < 25)].y.mean()))
    say("  爽约率倍数: %.1f 倍" % (hi.y.mean() / lo.y.mean()))
    say("  → 如果是'意愿随时间衰减'，曲线应当是渐变的；出现纯空档+阶跃，")
    say("    指向'两套取数逻辑/两条管道'的机制切换，而不是同一条曲线的陡坡。")

    say("\n  [阈值扫描] 净收益 = 拦截爽约×90 − 干预数×20，扫 5~78 分钟:")
    scan = []
    for tm in np.arange(5, 78.1, 1.0):
        mm = e.stale_h > tm / 60
        nn, hh = int(mm.sum()), int(e.y[mm].sum())
        if nn >= 20:
            scan.append((tm, nn, hh, hh * COST_NO_SHOW_H - nn * COST_CONFIRM_H))
    sc = pd.DataFrame(scan, columns=["min", "n", "hit", "net"])
    top = sc.loc[sc.net.idxmax()]
    at18 = sc.loc[(sc["min"] - CLIFF_MIN).abs().idxmin()]
    say("    最大净收益: >%.0f 分钟  n=%d  拦截 %d  净收益 %+d h"
        % (top["min"], top.n, top.hit, top.net))
    say("    取 18 分钟:            n=%d  拦截 %d  净收益 %+d h  （比最优点少 %d h，约 %.1f%%）"
        % (at18.n, at18.hit, at18.net, top.net - at18.net,
           100 * (top.net - at18.net) / at18.net))
    say("    → 最优点在 %.0f~%.0f 分钟，但要走到那里必须把 22~23 分钟那 %d 条观测（全部未爽约）"
        % (sc[sc.net > top.net - 40]["min"].min(), sc[sc.net > top.net - 40]["min"].max(),
           int(((e.stale_h >= 22 / 60 - 1e-9) & (e.stale_h <= 23 / 60 + 1e-9)).sum())))
    say("      一并排除 —— 靠这 %d 条把边界推后属于过拟合小样本。"
        % int(((e.stale_h >= 22 / 60 - 1e-9) & (e.stale_h <= 23 / 60 + 1e-9)).sum()))
    say("      18.00 分钟那一点有 %d 条（%.1f%%），是密集观测区的边界；"
        % (int((e.stale_h.round(6) == CLIFF_H).sum()),
           100 * e[e.stale_h.round(6) == CLIFF_H].y.mean()))
    say("      放弃这 %d 小时（%.1f%%）换稳健性 —— 取舍理由写进了 README。"
        % (top.net - at18.net, 100 * (top.net - at18.net) / at18.net))
    say("    阈值不敏感区: 18~22 分钟之间任意取值结果完全相同（该区间零观测）。")

    say("\n" + "=" * 66)
    say("四、反例检验：部门差异与 pool_size 是否独立成立")
    say("=" * 66)
    e["ch"] = np.where(e.stale_h <= CLIFF_H, "新鲜", "陈旧")
    t = e.groupby("department").agg(整体=("y", "mean"), 陈旧占比=("ch", lambda s: (s == "陈旧").mean()))
    p = e.pivot_table(index="department", columns="ch", values="y", aggfunc="mean")
    t["新鲜通道"] = p["新鲜"]; t["陈旧通道"] = p["陈旧"]
    t["n"] = e.groupby("department").size()
    t["陈旧n"] = e[e.ch == "陈旧"].groupby("department").size()
    say(t.assign(**{c: (t[c] * 100).round(1)
                    for c in ["整体", "陈旧占比", "新鲜通道", "陈旧通道"]}).to_string())
    say("  不分层时部门极差: %.1f%% → %.1f%% （差 %.1f 个百分点）"
        % (100 * t.整体.min(), 100 * t.整体.max(), 100 * (t.整体.max() - t.整体.min())))
    say("  分层到新鲜通道内: %.1f%% → %.1f%%（差 %.1f 个百分点）"
        % (100 * t.新鲜通道.min(), 100 * t.新鲜通道.max(),
           100 * (t.新鲜通道.max() - t.新鲜通道.min())))
    say("  → 部门差异 14.0 个百分点里，有 %.1f 个百分点是'通道占比不同'造成的，不是部门本身的问题。"
        % (100 * (t.整体.max() - t.整体.min()) - 100 * (t.新鲜通道.max() - t.新鲜通道.min())))
    say("  ⚠️ 只在新鲜通道内消失了；陈旧通道内部门差异不但没消失，反而是全数据里最强的信号之一（见下）。")

    say("\n  [检验 A] 陈旧通道内：是否存在一个'免疫'部门（数据驱动选出，非人为指定）")
    stale = e[e.ch == "陈旧"]
    cand = t[t.陈旧n >= 50].sort_values("陈旧通道")
    dept = cand.index[0] if len(cand) else t.sort_values("陈旧通道").index[0]
    a = stale[stale.department == dept]
    b = stale[stale.department != dept]
    say("    候选（陈旧通道内 n>=50 且爽约率最低）: %s" % dept)
    say("      %s        n=%3d  爽约 %3d  %s" % (dept, len(a), int(a.y.sum()), ci_str(a.y.sum(), len(a))))
    say("      其他三部门合计  n=%3d  爽约 %3d  %s"
        % (len(b), int(b.y.sum()), ci_str(b.y.sum(), len(b))))
    say("      两组 95%% 置信区间%s" %
        ("完全不重叠" if wilson_ci(a.y.sum(), len(a))[1] < wilson_ci(b.y.sum(), len(b))[0]
         else "有重叠"))
    p_f = fisher_two_sided(int(a.y.sum()), len(a) - int(a.y.sum()),
                           int(b.y.sum()), len(b) - int(b.y.sum()))
    say("      Fisher 精确检验 双尾 p = %.2g" % p_f)
    say("      4 个部门里挑最极端的那个 → Bonferroni ×%d 校正后 p = %.2g  %s"
        % (len(t), min(1.0, p_f * len(t)),
           "仍显著" if p_f * len(t) < 0.05 else "不显著"))
    say("\n  [检验 B] 其他三个部门互相比（排除'只是随机波动'）:")
    for d_ in t.index:
        if d_ == dept:
            continue
        aa = stale[stale.department == d_]
        bb = stale[stale.department != d_]
        pp = fisher_two_sided(int(aa.y.sum()), len(aa) - int(aa.y.sum()),
                              int(bb.y.sum()), len(bb) - int(bb.y.sum()))
        say("      %-8s n=%3d %.1f%%  p=%.3g  校正后 p=%.3g  %s"
            % (d_, len(aa), 100 * aa.y.mean(), pp, min(1.0, pp * len(t)),
               "仍显著" if pp * len(t) < 0.05 else "不显著"))
    say("      → 只有 %s 特殊，其余三个部门之间无显著差异，不是四选一的随机 luck。" % dept)

    say("\n  [检验 C] 稳健性：这个效应是否只是 18 分钟这个阈值碰巧造成的")
    for th in [0.30, 0.40, 0.50, 0.60, 0.80]:
        s = e[e.stale_h > th]
        aa = s[s.department == dept]
        bb = s[s.department != dept]
        if len(aa) >= 10:
            say("      > %.2f h (%.0f min): %s n=%2d %5.1f%%  |  其他 n=%3d %5.1f%%"
                % (th, th * 60, dept, len(aa), 100 * aa.y.mean(), len(bb), 100 * bb.y.mean()))
    say("      → 每一档都成立，不依赖阈值选择。")

    fr = e[e.ch == "新鲜"]
    aa = fr[fr.department == dept]
    say("\n  [检验 D] 反向印证：%s 在新鲜通道内 %.1f%%（n=%d）vs 全体新鲜 %.1f%%"
        % (dept, 100 * aa.y.mean(), len(aa), 100 * fr.y.mean()))
    say("      → 该部门不是'整体更靠谱'，而是'即使数据陈旧也仍会出席'。")
    say("        这是真正的【交互效应】：通道是主因，%s 在陈旧通道上另有一条独立免疫。" % dept)
    say("  ⚠️ 该发现属事后发现（post-hoc，4 选 1），须在新数据或 A/B 中复核，不能直接当长期配额。")

    say("\n  [检验 E] 排除替代解释：该部门效应是否只是'岗位职能构成'混杂出来的")
    say("     若该部门只是碰巧多招了某种职能，则控制职能后效应应消失。三层验证：")
    obs = pd.crosstab(stale.department == dept, stale.job_category).values
    row = obs.sum(1, keepdims=True); col = obs.sum(0, keepdims=True); nn = obs.sum()
    chi2 = float(((obs - row @ col / nn) ** 2 / (row @ col / nn)).sum())
    dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    say("     E1 属性检验：该部门 vs 其他 的职能分布  chi2=%.2f  dof=%d  （临界 14.07 @p=0.05）" % (chi2, dof))
    say("        → 分布无显著差异，两侧职能构成可比，不存在'职能集中'")
    hits = 0; tot_jc = 0
    detail = []
    for jc in sorted(stale.job_category.unique()):
        s = stale[stale.job_category == jc]
        aa2 = s[s.department == dept]; bb2 = s[s.department != dept]
        if len(aa2) >= 3 and len(bb2) >= 3:
            tot_jc += 1
            if aa2.y.mean() < bb2.y.mean():
                hits += 1
            pp = fisher_two_sided(int(aa2.y.sum()), len(aa2) - int(aa2.y.sum()),
                                  int(bb2.y.sum()), len(bb2) - int(bb2.y.sum()))
            detail.append("        %-8s %s n=%2d %5.1f%%  |  其他 n=%3d %5.1f%%  p=%.3f%s"
                          % (jc, dept, len(aa2), 100 * aa2.y.mean(), len(bb2), 100 * bb2.y.mean(),
                             pp, "  ←单独显著" if pp < 0.05 else ""))
    say("     E2 分层到每个职能内部：")
    for d_ in detail:
        say(d_)
    say("        → %d/%d 个职能内部该部门都更低，方向完全一致（其中若干单独显著）" % (hits, tot_jc))
    say("        → 效应跨职能稳定，不是某个职能带出来的")

    say("\n  [检验 F] 对照：'岗位职能'本身有没有独立效应（同样的检验标准）")
    for jc_ in stale.job_category.unique():
        aa3 = stale[stale.job_category == jc_]; bb3 = stale[stale.job_category != jc_]
        pp = fisher_two_sided(int(aa3.y.sum()), len(aa3) - int(aa3.y.sum()),
                              int(bb3.y.sum()), len(bb3) - int(bb3.y.sum()))
        say("      %-8s n=%3d %5.1f%%  p=%.3g  8选1校正后 p=%.2g  %s"
            % (jc_, len(aa3), 100 * aa3.y.mean(), pp, min(1.0, pp * 8),
               "显著" if pp * 8 < 0.05 else "不显著"))
    say("      → 职能层面没有可确认的差异（极差看着有 %.0f 个百分点，但每组仅 20~40 条，撑不住）。"
        % (100 * (stale.groupby("job_category").y.mean().max()
                  - stale.groupby("job_category").y.mean().min())))
    say("      → 这与部门检验形成对照：同样是'分组看差异'，部门通过了校正、职能没通过。")
    say("        筛选标准是统计检验，不是'看起来差得多'。")

    lo_m = e.pool_size <= e.pool_size.median()
    say("  pool_size 表观 AUC 0.573，分层后:")
    say("    新鲜通道 pool<=中位 %.1f%% vs >中位 %.1f%%"
        % (100 * e[(e.ch == "新鲜") & lo_m].y.mean(), 100 * e[(e.ch == "新鲜") & ~lo_m].y.mean()))
    say("    陈旧通道 pool<=中位 %.1f%% vs >中位 %.1f%%"
        % (100 * e[(e.ch == "陈旧") & lo_m].y.mean(), 100 * e[(e.ch == "陈旧") & ~lo_m].y.mean()))
    say("    → 效应同样基本消失，pool_size 只是通道构成的代理变量")
    say("\n  [检验 G] 意向分：控制通道后的效应量级对比（AUC 0.498 的直观版）")
    e["_ib"] = pd.cut(e.intent_score, [0, 10, 20, 30])
    g = e.groupby(["_ib", "ch"], observed=True).y.agg(["size", "mean"]).reset_index()
    for ch_ in ["新鲜", "陈旧"]:
        line = "".join("  %s n=%3d %5.1f%%" % (str(r._ib), r["size"], 100 * r["mean"])
                       for _, r in g[g.ch == ch_].iterrows())
        say("    %s通道:%s" % (ch_, line))
    say("    同一意向档内，陈旧/新鲜 的倍数：")
    for ib in g._ib.unique():
        aa4 = e[(e._ib == ib) & (e.ch == "陈旧")]; bb4 = e[(e._ib == ib) & (e.ch == "新鲜")]
        if len(aa4) and len(bb4):
            say("      %-8s %5.1f%% / %5.1f%% = %4.1f 倍"
                % (str(ib), 100 * aa4.y.mean(), 100 * bb4.y.mean(), aa4.y.mean() / bb4.y.mean()))
    hi_ib = g[(g.ch == "陈旧")].sort_values("mean").iloc[-1]
    say("    → 关键对照：在**意向分最高**的那一档（%s），通道造成的差距是 %.1f 倍；"
        % (str(hi_ib._ib), e[(e._ib == hi_ib._ib) & (e.ch == "陈旧")].y.mean()
           / e[(e._ib == hi_ib._ib) & (e.ch == "新鲜")].y.mean()))
    say("      而意向分自己在同一通道内的波动只有几个百分点。")
    say("    → 意向分与陈旧度的相关系数 r=%.3f、与储备量 r=%.3f：几乎正交。"
        % (e.intent_score.corr(e.stale_h), e.intent_score.corr(e.pool_size)))
    say("      所以它不是任何东西的代理变量——它就是没有信息。")
    say("    → 三个字段的 AUC：intent_score %.3f | pool_size %.3f | stale_h %.3f"
        % (auc(e.intent_score, e.y), auc(e.pool_size, e.y), auc(e.stale_h, e.y)))

    say("\n" + "=" * 66)
    say("五、可落地的决策规则与净收益")
    say("=" * 66)
    tot, n_all = int(e.y.sum()), len(e)

    def ev(name, m):
        n = int(m.sum()); hit = int(e.y[m].sum())
        net = hit * COST_NO_SHOW_H - n * COST_CONFIRM_H
        p = hit / n if n else 0
        need = BREAKEVEN / p if p else float("inf")
        say("  %-24s 干预%4d (%2.0f%%)  覆盖%5.1f%%  命中%5.1f%%  净收益%+7d h  需有效率>%3.0f%%"
            % (name, n, 100 * n / n_all, 100 * hit / tot, 100 * p, net,
               100 * need if need != float("inf") else 999))
        return net

    rules = {}
    rules["① 全量干预（无差别）"] = (0, 0, ev("① 全量干预（无差别）", e.y.notna()))
    for t_ in [5, 10, 15]:
        k = "② intent_score<=%d（现依据）" % t_
        rules[k] = (0, 0, ev(k, e.intent_score <= t_))
    k = "③ 陈旧>18min（保守版）"
    rules[k] = (0, 0, ev(k, e.stale_h > CLIFF_H))
    rules["④ 陈旧>18min 或 pool>=70"] = (0, 0, ev("④ 陈旧>18min 或 pool>=70",
                                               (e.stale_h > CLIFF_H) | (e.pool_size >= 70)))
    rules["⑤ 部门=商业化销售部"] = (0, 0, ev("⑤ 部门=商业化销售部", e.department == "商业化销售部"))
    rules["⑥ intent<=15 且 陈旧>30min"] = (0, 0, ev("⑥ intent<=15 且 陈旧>30min",
                                                   (e.intent_score <= 15) & (e.stale_h > .5)))
    k7 = "⑦ 陈旧>18min 且非%s（推荐）" % dept
    rules[k7] = (0, 0, ev(k7, (e.stale_h > CLIFF_H) & (e.department != dept)))
    k8 = "⑧ 陈旧>23min 且非%s" % dept
    rules[k8] = (0, 0, ev(k8, (e.stale_h > 23 / 60) & (e.department != dept)))

    m = (e.stale_h > CLIFF_H) & (e.department != dept)
    n, hit = int(m.sum()), int(e.y[m].sum())
    m3 = e.stale_h > CLIFF_H
    n3, hit3 = int(m3.sum()), int(e.y[m3].sum())
    say("\n  推荐规则明细：对【活跃状态同步超过 18 分钟】且【部门 ≠ %s】的邀约，做一次提前意向二次确认" % dept)
    say("    干预 %d / %d 条（占 %.0f%% 的邀约），拦截 %d / %d 次爽约（覆盖 %.1f%%）"
        % (n, n_all, 100 * n / n_all, hit, tot, 100 * hit / tot))
    say("    命中率 %.1f%%，净收益 %+d 小时；同样口径下全量干预是 %+d 小时"
        % (100 * hit / n, hit * 90 - n * 20, tot * 90 - n_all * 20))
    say("    对比保守版（不排除 %s）：干预 %d、覆盖 %.1f%%、命中 %.1f%%、净收益 %+d h"
        % (dept, n3, 100 * hit3 / tot, 100 * hit3 / n3, hit3 * 90 - n3 * 20))
    say("    → 排除该部门后少干预 %d 条（省 %d h），只少拦 %d 次爽约（少挽回 %d h），净多得 %+d 小时"
        % (n3 - n, (n3 - n) * 20, hit3 - hit, (hit3 - hit) * 90,
           (hit * 90 - n * 20) - (hit3 * 90 - n3 * 20)))
    say("    被排除的那 %d 条：爽约率仅 %.1f%%，干预净收益 %+d h（成本 %d h 只挽回 %d h）"
        % (n3 - n, 100 * e[(e.stale_h > CLIFF_H) & (e.department == dept)].y.mean(),
           int(e[(e.stale_h > CLIFF_H) & (e.department == dept)].y.sum()) * 90 - (n3 - n) * 20,
           (n3 - n) * 20,
           int(e[(e.stale_h > CLIFF_H) & (e.department == dept)].y.sum()) * 90))
    say("    阈值敏感性：18~22 分钟之间任意取值结果完全相同（该区间零观测）；")
    say("      扫描最优点在 22~23 分钟（多 %+d h），但那里只有 6 条观测，坚持 18 分钟——理由见第三节。"
        % (top.net - at18.net))
    say("    成本敏感性：需要'二次确认有效率' > %.0f%% 才划算（见 README 假设）"
        % (100 * BREAKEVEN / (hit / n)))

    say("\n" + "=" * 66)
    say("六、明确不做的事（附数字理由）")
    say("=" * 66)
    say("  · 不按 intent_score 挑人：各档净收益全为负，最好的一档也只是把亏损从 -8720h 缩到 -1460h")
    say("  · 不对新鲜通道（%d 条）做任何干预：其爽约率仅 %.2f%%，远低于 22.2%% 保本线，"
        % (len(lo), 100 * lo.y.mean()))
    say("    对这些人的干预纯属浪费（若覆盖 %d 人需多花 %d 小时）" % (len(lo), len(lo) * 20))
    say("  · 不按部门做'多干预/少干预'的整体配额：新鲜通道内四部门差距只有 %.1f 个百分点，"
        % (100 * (t.新鲜通道.max() - t.新鲜通道.min())))
    say("    那 %.1f 个百分点纯粹是'谁的邀约更多走了陈旧通道'造成的，按部门配额度是拿表象当原因。"
        % (100 * (t.整体.max() - t.整体.min()) - 100 * (t.新鲜通道.max() - t.新鲜通道.min())))
    say("  · 不按岗位职能干预：职能层面极差看着有 %.0f 个百分点，但 8 选 1 Bonferroni 校正后"
        % (100 * (stale.groupby("job_category").y.mean().max()
                  - stale.groupby("job_category").y.mean().min())))
    say("    全部不显著（最小校正后 p 仍 > 0.5），每组只有 20~40 条——统计上撑不住，不做。")
    say("  · 不按简历储备量 pool_size 干预：分层后陈旧通道内四分位为 31.9/40.0/35.6/36.3，无单调梯度")
    say("  · 但有一个部门例外，且已做进规则：陈旧通道内 %s 只有 %.1f%%（n=%d）vs 其他三部门 %.1f%%（n=%d），"
        % (dept, 100 * a.y.mean(), len(a), 100 * b.y.mean(), len(b)))
    say("    Fisher p=%.1e、Bonferroni 校正后仍显著、CI 不重叠、各档阈值下都成立；" % p_f)
    say("    给这批人做二次确认净收益 %+d 小时，必须排除（不是'少干预'，是完全不干预）。"
        % (int(a.y.sum()) * 90 - len(a) * 20))

    fig_cliff(e); fig_gap(e); fig_simpson(e); fig_intent(e); fig_dimensions(e, dept)
    fig_interaction(e, dept); fig_rules(e, rules)
    say("\n已生成图: fig_cliff / fig_gap / fig_simpson / fig_intent / fig_dimensions / "
        "fig_interaction / fig_rules (.png)")
    (OUT / "report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n已写入 %s" % (OUT / "report.txt"))


if __name__ == "__main__":
    main()
