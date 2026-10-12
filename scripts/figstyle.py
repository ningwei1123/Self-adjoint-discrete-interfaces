# -*- coding: utf-8 -*-
"""
figstyle.py — 投稿级图件统一样式 (Elsevier / CMAME)
===================================================
依据 Elsevier 官方 artwork 规范:
  * finished printed lettering: 7 pt 常规, >=6 pt 上下标
  * 双栏整幅宽 190 mm (7.48 in), 单栏 90 mm
  * 分辨率: halftone 300 / combination 500 / line art 1000 dpi
  * 线宽 0.25-1.5 pt
  * 提交格式: PDF/EPS (矢量, 首选) + TIFF (LZW)

设计宽度取 190 mm, 字体名义值即印刷实际值 (不用再被缩小).
所有图统一使用 constrained layout 以避免标签/标题互相遮挡.
"""
import _paths  # noqa: F401  (adds src/ to the path)
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- 版面常数 ----------------
MM = 1.0 / 25.4
W_DOUBLE = 190 * MM        # 双栏整幅 = 7.480 in
W_ONEHALF = 140 * MM       # 1.5 栏
W_SINGLE = 90 * MM         # 单栏

# ---------------- 色盲安全色板 ----------------
CM_SEQ = 'viridis'         # 幅值场 |u|, sigma_vm
CM_ERR = 'inferno'         # 误差场 |Delta sigma| (深底: 适合误差处处非零)
CM_ERR_LIGHT = 'YlOrRd'    # 误差场 (浅底: 零值清晰, 越深=越大)
# 浅蓝 -> 深红: 单调非负误差场的双色梯度。低端取近白蓝(而非纯白),
# 以便"接近零"与纸面背景仍可区分; 高端取深红以突出强失配区。
# 节点配比: 蓝段占据低到中程 (对数刻度上 5 个数量级里的大部分), 之后经
# 橙色过渡到深红, 使渐变均匀而不是"浅蓝直跳红"。
CM_ERR_BR = [(0.00, '#eef5fc'), (0.18, '#c3d9ef'), (0.38, '#7fb0dc'),
             (0.58, '#4a86bd'), (0.74, '#e8a33d'), (0.88, '#d4552f'),
             (1.00, '#7d1414')]
CM_LOG = 'magma'           # 对数热图
CM_DIV = 'RdBu_r'          # 有符号 / 对称性

C_GAL = '#1f6feb'          # Galerkin
C_COL = '#d1495b'          # Collocation
C_ACC = '#e08e0b'          # 强调 / 界面
C_GRN = '#2b8a3e'          # CPINN
C_GRY = '#7a7a7a'          # 基线
C_REF = '#222222'

_PALETTE = [C_GAL, C_COL, C_ACC, C_GRN, C_GRY]

# 投稿图件唯一字族: DejaVu Sans 覆盖全部希腊字母/数学符号, 无需回退到中文衬线体,
# 保证同一篇稿件内所有图件字面完全一致 (Elsevier artwork 硬性要求)。
PUB_FONT = 'DejaVu Sans'
_CJK_CANDIDATES = ['Microsoft YaHei', 'SimHei', 'STSong', 'Arial Unicode MS']


def use_cjk_fallback():
    """让中文标签可渲染, 但**绝不覆盖**已显式设定的出版字体。

    历史坑: 某些求解器模块在 import 时无条件把 ``font.sans-serif`` 改成中文字体,
    导致之后生成的图件全部掺入 Microsoft YaHei (同一篇稿件出现两种字面)。
    这里改为「仅当调用方尚未配置字体时才插入中文回退」, 因此
    ``apply()`` 之后再 import 这些模块不会污染投稿图件, 而旧的中文标注脚本
    (不调用 apply()) 仍能正常显示汉字。
    """
    from matplotlib import font_manager
    default = list(matplotlib.rcParamsDefault['font.sans-serif'])
    cur = list(plt.rcParams['font.sans-serif'])
    if cur != default:          # 调用方已显式设置 -> 保持不动
        return False
    for cand in _CJK_CANDIDATES:
        try:
            font_manager.findfont(cand, fallback_to_default=False)
        except Exception:
            continue
        plt.rcParams['font.sans-serif'] = [cand] + cur
        plt.rcParams['axes.unicode_minus'] = False
        return True
    return False


def apply(base=9.0):
    """全局 rcParams。base = 正文字号 (>=7 pt 印刷下限, 取 9 保险)。"""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        # 单一字族贯穿正文与公式, 避免 Arial/YaHei/CM 混排 (Elsevier: 全刊图件应统一字面)
        'font.sans-serif': [PUB_FONT],
        'mathtext.fontset': 'dejavusans',
        'font.size': base,
        'axes.labelsize': base,
        'axes.titlesize': base,
        'xtick.labelsize': base - 0.5,
        'ytick.labelsize': base - 0.5,
        'legend.fontsize': base - 0.5,
        'figure.titlesize': base + 0.5,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.22,
        'grid.linewidth': 0.5,
        'grid.color': '#9aa0a6',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'legend.frameon': False,
        'legend.handlelength': 1.6,
        'legend.labelspacing': 0.35,
        'legend.borderaxespad': 0.3,
        'lines.linewidth': 1.3,
        'lines.markersize': 4.0,
        'figure.dpi': 130,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        # 用真正的减号 U+2212 (DejaVu 内含), 而非 ASCII 连字符, 符合排版规范
        'axes.unicode_minus': True,
        'axes.axisbelow': True,
        'figure.autolayout': False,
    })


# 标注允许的最小字号。设计宽 190 mm, 投稿 docx 版心 158 mm -> 缩放 0.83,
# 8.5 pt 落到 7.1 pt, 仍高于 Elsevier 的 7 pt 下限。
MIN_ANNOT = 8.5


def new(figsize, **kw):
    """建图, 默认开启 constrained layout (自动防重叠)。"""
    kw.setdefault('layout', 'constrained')
    fig = plt.figure(figsize=figsize, **kw)
    return fig


def panel(ax, letter, x=-0.02, y=1.06):
    """统一的面板标签 (粗体小写, 置于左上角外侧)。"""
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='bottom', ha='left')


def finish(ax, title=None, xlabel=None, ylabel=None, letter=None,
           grid_axis='both', legend=None):
    """一次性收口: 标题 / 轴标签 / 面板标签 / 图例。"""
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if letter is not None:
        panel(ax, letter)
    if grid_axis is not None:
        if grid_axis == 'both':
            ax.grid(True, alpha=0.22)
        else:
            ax.grid(True, alpha=0.22, axis=grid_axis)
    if legend is not None:
        ax.legend(**legend)
    return ax


def cbar(fig, mappable, ax, label=None, size=8.0, **kw):
    """统一 colorbar: 必带 label, 刻度字号统一。"""
    size = max(size, MIN_ANNOT)          # 任何调用方都不得低于投稿下限
    kw.setdefault('pad', 0.02)
    kw.setdefault('fraction', 0.046)
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.ax.tick_params(labelsize=size - 0.5)
    if label:
        cb.set_label(label, fontsize=size)
    return cb


def save(fig, name, outdir, png_dpi=600, tiff=True, pdf=True):
    """三种产物: PDF (矢量, 投稿首选) + PNG (预览/docx) + TIFF (生产)。"""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    if pdf:
        p = os.path.join(outdir, name + '.pdf')
        fig.savefig(p)                      # 矢量, 字体 Type42 嵌入
        paths.append(p)
    p = os.path.join(outdir, name + '.png')
    fig.savefig(p, dpi=png_dpi)
    paths.append(p)
    if tiff:
        p = os.path.join(outdir, name + '.tif')
        try:
            from PIL import Image
            tmp = os.path.join(outdir, name + '_tmp.png')
            fig.savefig(tmp, dpi=500)
            im = Image.open(tmp).convert('RGB')
            im.save(p, compression='tiff_lzw')
            os.remove(tmp)
            paths.append(p)
        except Exception as e:                                   # pragma: no cover
            print(f'  [warn] TIFF 写出失败: {e}')
    plt.close(fig)
    print(f'[出图] {name}  ->  ' + ', '.join(os.path.basename(x) for x in paths))
    return paths
