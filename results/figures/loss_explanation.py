"""Render the GRPO loss derivation as a PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager as fm

# Try to find a CJK-capable font
candidates = ["Noto Sans CJK SC", "Droid Sans Fallback", "WenQuanYi Zen Hei",
              "DejaVu Sans"]
for c in candidates:
    try:
        fm.findfont(c, fallback_to_default=False)
        cjk_font = c
        break
    except Exception:
        cjk_font = "DejaVu Sans"

rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = ['Droid Sans Fallback', 'DejaVu Serif']

fig = plt.figure(figsize=(13, 13))
fig.patch.set_facecolor('white')
ax = fig.add_axes([0.04, 0.02, 0.92, 0.96])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

y = 0.97
def title(t, size=15, weight='bold'):
    global y
    ax.text(0.0, y, t, fontsize=size, fontweight=weight); y -= 0.045
def line(t, size=12):
    global y
    ax.text(0.02, y, t, fontsize=size); y -= 0.034
def math(t, size=15):
    global y
    ax.text(0.05, y, t, fontsize=size); y -= 0.05
def gap(h=0.018):
    global y; y -= h

# 1. Definition
title("1.  Loss 定义", size=16)
line("一条轨迹 i (在第 g 组里) 的 PG loss:")
math(r"$L_{g,i} \;=\; -\;A_{g,i}\;\cdot\;\overline{\log p_\theta}\,(\mathrm{traj}\;i\;\mathrm{actions})$")
gap()
line("其中:")
math(r"$A_{g,i} \;=\; \dfrac{\,r_{g,i} \,-\, \bar r_g\,}{\sigma_{r,g} + \varepsilon}$    "
     r"= group 内 z-score 标准化的 reward")
math(r"$\overline{\log p_\theta} \;=\; \dfrac{1}{T}\,\sum_{t=1}^{T} \log p_\theta(\mathrm{token}_t \,|\, \mathrm{prefix})$    "
     r"= 模型对这条轨迹动作的对数概率均值")
gap()
math(r"$L_{\mathrm{total}} \;=\; \dfrac{1}{n_{\mathrm{items}}}\;\sum_{g,\,i}\, L_{g,i}$")

ax.axhline(y - 0.015, color='gray', alpha=0.3); y -= 0.04

# 2. Why near zero
title("2.  为什么 loss 在 0 附近", size=16)
line("具体例子: 一个 group, G=8, 最终 rewards =")
math(r"$\mathbf{r}\;=\;[\,1,\,1,\,0,\,0,\,1,\,0,\,1,\,0\,]$")
math(r"$\bar r = 0.5,\quad \sigma_r = 0.5$")
math(r"$A \;=\; (r-0.5)/0.5 \;=\; [\,+1,\,+1,\,-1,\,-1,\,+1,\,-1,\,+1,\,-1\,]$")
line("注意: sum(A_i) = 0   <-- z-score 中心化的强制结果")
gap()
line("假设每条轨迹的 mean_logp ~= -3 (7B 模型给一串动作 token 的对数概率均值):")
math(r"$\mathbf{logp}\;\approx\;[\,-3.1,\,-2.9,\,-3.0,\,-3.2,\,-2.8,\,-3.1,\,-2.9,\,-3.0\,]$")
gap()
line("每条 loss_i = -A_i * logp_i:")
math(r"$[\,+3.1,\,+2.9,\,-3.0,\,-3.2,\,+2.8,\,-3.1,\,+2.9,\,-3.0\,]$")
math(r"$\sum_i = -0.6,\quad \div\, n_{\mathrm{items}}=64\;\Rightarrow\; "
     r"L_{\mathrm{total}}\approx -0.009$")

ax.axhline(y - 0.015, color='gray', alpha=0.3); y -= 0.04

# 3. Insight
title("3.  关键直觉", size=16)
line("每条 loss_i 量级 ~= +/- 3   (因 |A| ~= 1, |logp| ~= 3)")
line("但 sum(A) = 0, 把 +/- 3 的主体几乎完全抵消")
gap()
line("剩下的 \"残余\" = advantage 跟 logp 偏离均值的相关性:")
math(r"$L_{\mathrm{total}} \;\approx\; \sum_i A_i \cdot ( \overline{\log p_\theta}_i \,-\, \mathrm{mean}(\overline{\log p_\theta}) )$",
     size=14)
line("这才是真正的训练信号.  所以 loss 是 ~0.01 的小数, 不是 bug.")
gap()
line("* 监督学习 loss = \"模型当前有多错\"  ->  单调降到 0")
line("* PG loss      = \"advantage 加权 logp\", 期望值 = 0, 围绕 0 震荡是预期行为")
line("* 收敛要看的是 grad-norm 趋势 + 下游 reward, 不是 loss 本身")

plt.savefig('/home/zhiyuanzhai/selective rollout/results/figures/loss_explanation.png',
            dpi=140, bbox_inches='tight', facecolor='white')
print("wrote results/figures/loss_explanation.png")
