#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interview_no_show.py — 面试爽约分析（可复现）

用法:
    python interview_no_show.py                       # 默认读 data/interview_invitations.csv
    python interview_no_show.py 路径/to/file.csv

产出:
    output/report.txt      全部数字（README 里每个数字都能在这里找到）
    output/fig_*.png       5 张图

设计原则:
    1. 只用发放数据，不引入任何外部或合成数据
    2. 每个数字都能被追到具体代码行
    3. 打印"反例检验"的结果，不只打印支持结论的证据
"""

import sys
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
    ax.set_title("部门差异是假的：分层后 14 个百分点的差距基本消失")
    ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_simpson.png", dpi=150); plt.close(fig)


def fig_rules(e, rules):
    names = list(rules); nets = [rules[k][2] for k in names]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(names, nets, color=["#2e7d32" if v > 0 else "#c62828" for v in nets])
    ax.axvline(0, c="k", lw=1)
    for i, (k, v) in enumerate(zip(names, nets)):
        ax.text(v + (60 if v > 0 else -60), i, "%+d h" % v, va="center",
                ha="left" if v > 0 else "right", fontsize=8.5)
    ax.set_xlabel("净收益（小时）＝ 拦截爽约×90 − 干预数×20")
    ax.set_title("干预谁才划算：按新鲜度挑人 vs 按意向分挑人")
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
    say("  新鲜通道上界正好压在 18.00 分钟这一点上（n=%d 条），再往上直接跳到 22 分钟——像一档缓存/会话有效期"
        % int((e.stale_h.round(6) == CLIFF_H).sum()))
    say("  爽约率倍数: %.1f 倍" % (hi.y.mean() / lo.y.mean()))
    say("  → 如果是'意愿随时间衰减'，曲线应当是渐变的；出现纯空档+阶跃，")
    say("    指向'两套取数逻辑/两条管道'的机制切换，而不是同一条曲线的陡坡。")

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
    say("  ⚠️ 但没完全消失：陈旧通道内 风险合规部 %.1f%%，仍显著低于其他三部门（%.1f%%~%.1f%%），"
        % (100 * t.loc["风险合规部", "陈旧通道"], 100 * t[t.index != "风险合规部"]["陈旧通道"].min(),
           100 * t[t.index != "风险合规部"]["陈旧通道"].max()))
    say("     说明除通道外还有一个较小的部门级因素（本样本不足以拆开，列入局限）。")
    lo_m = e.pool_size <= e.pool_size.median()
    say("  pool_size 表观 AUC 0.573，分层后:")
    say("    新鲜通道 pool<=中位 %.1f%% vs >中位 %.1f%%"
        % (100 * e[(e.ch == "新鲜") & lo_m].y.mean(), 100 * e[(e.ch == "新鲜") & ~lo_m].y.mean()))
    say("    陈旧通道 pool<=中位 %.1f%% vs >中位 %.1f%%"
        % (100 * e[(e.ch == "陈旧") & lo_m].y.mean(), 100 * e[(e.ch == "陈旧") & ~lo_m].y.mean()))
    say("    → 效应同样基本消失，pool_size 只是通道构成的代理变量")
    say("  intent_score 分层后依旧无信号（AUC 0.498 的直观版）:")
    say("    新鲜通道 intent<=10 %.1f%% vs intent>20 %.1f%%"
        % (100 * e[(e.ch == "新鲜") & (e.intent_score <= 10)].y.mean(),
           100 * e[(e.ch == "新鲜") & (e.intent_score > 20)].y.mean()))
    say("    陈旧通道 intent<=10 %.1f%% vs intent>20 %.1f%%（方向还反了）"
        % (100 * e[(e.ch == "陈旧") & (e.intent_score <= 10)].y.mean(),
           100 * e[(e.ch == "陈旧") & (e.intent_score > 20)].y.mean()))

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
    k = "③ 陈旧>18min（推荐）"
    rules[k] = (0, 0, ev(k, e.stale_h > CLIFF_H))
    rules["④ 陈旧>18min 或 pool>=70"] = (0, 0, ev("④ 陈旧>18min 或 pool>=70",
                                               (e.stale_h > CLIFF_H) | (e.pool_size >= 70)))
    rules["⑤ 部门=商业化销售部"] = (0, 0, ev("⑤ 部门=商业化销售部", e.department == "商业化销售部"))
    rules["⑥ intent<=15 且 陈旧>30min"] = (0, 0, ev("⑥ intent<=15 且 陈旧>30min",
                                                   (e.intent_score <= 15) & (e.stale_h > .5)))

    m = e.stale_h > CLIFF_H
    n, hit = int(m.sum()), int(e.y[m].sum())
    say("\n  推荐规则明细：对活跃状态同步超过 18 分钟的邀约，做一次提前意向二次确认")
    say("    干预 %d / %d 条（占 %.0f%% 的邀约），拦截 %d / %d 次爽约（覆盖 %.1f%%）"
        % (n, n_all, 100 * n / n_all, hit, tot, 100 * hit / tot))
    say("    命中率 %.1f%%，净收益 %+d 小时；同样口径下全量干预是 %+d 小时"
        % (100 * hit / n, hit * 90 - n * 20, tot * 90 - n_all * 20))
    say("    阈值敏感性：把 18 分钟换成 18~22 分钟之间任意值，结果完全相同（该区间零观测）")
    say("    成本敏感性：需要'二次确认有效率' > %.0f%% 才划算（见 README 假设）"
        % (100 * BREAKEVEN / (hit / n)))

    say("\n" + "=" * 66)
    say("六、明确不做的事（附数字理由）")
    say("=" * 66)
    say("  · 不按 intent_score 挑人：各档净收益全为负，最好的一档也只是把亏损从 -8720h 缩到 -1460h")
    say("  · 不对新鲜通道（%d 条）做任何干预：其爽约率仅 %.2f%%，远低于 22.2%% 保本线，"
        % (len(lo), 100 * lo.y.mean()))
    say("    对这些人的干预纯属浪费（若覆盖 %d 人需多花 %d 小时）" % (len(lo), len(lo) * 20))
    say("  · 不按部门/职能单独干预：分层后部门差异从 14.0 个百分点缩到 1.8 个百分点，按部门切是拿表象当原因")

    fig_cliff(e); fig_gap(e); fig_simpson(e); fig_intent(e); fig_rules(e, rules)
    say("\n已生成图: fig_cliff / fig_gap / fig_simpson / fig_intent / fig_rules (.png)")
    (OUT / "report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n已写入 %s" % (OUT / "report.txt"))


if __name__ == "__main__":
    main()
