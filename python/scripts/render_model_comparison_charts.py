#!/usr/bin/env python3
"""Scan checkpoint eval JSON and render OOS walk-forward comparison as SVG (no extra deps)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Row:
    label: str
    symbol: str  # BTC or ETH
    mean_return: float
    std_return: float
    mean_max_dd: float
    win_rate: float
    source: str


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def _sym_folder(name: str) -> str:
    return "BTC" if "BTC" in name else "ETH"


def collect_rows(checkpoints: Path) -> list[Row]:
    rows: list[Row] = []

    for path in sorted(checkpoints.glob("**/wf_14d.eval.json")):
        rel = path.relative_to(checkpoints)
        if len(rel.parts) < 3:
            continue
        run, sym_folder, _ = rel.parts[0], rel.parts[1], rel.parts[2]
        s = _load_summary(path)
        rows.append(
            Row(
                label=run,
                symbol=_sym_folder(sym_folder),
                mean_return=float(s["mean_return"]),
                std_return=float(s["std_return"]),
                mean_max_dd=float(s["mean_max_dd"]),
                win_rate=float(s["win_rate"]),
                source=str(path.relative_to(checkpoints.parent)),
            )
        )

    for path in sorted(checkpoints.glob("**/wf_14d_best.eval.json")):
        rel = path.relative_to(checkpoints)
        run, sym_folder, _ = rel.parts[0], rel.parts[1], rel.parts[2]
        s = _load_summary(path)
        rows.append(
            Row(
                label=f"{run} (best ckpt)",
                symbol=_sym_folder(sym_folder),
                mean_return=float(s["mean_return"]),
                std_return=float(s["std_return"]),
                mean_max_dd=float(s["mean_max_dd"]),
                win_rate=float(s["win_rate"]),
                source=str(path.relative_to(checkpoints.parent)),
            )
        )

    for path in sorted(checkpoints.glob("v7_long/*/best_agent.eval.json")):
        rel = path.relative_to(checkpoints)
        sym_folder = rel.parts[1]
        s = _load_summary(path)
        rows.append(
            Row(
                label="v7_long",
                symbol=_sym_folder(sym_folder),
                mean_return=float(s["mean_return"]),
                std_return=float(s["std_return"]),
                mean_max_dd=float(s["mean_max_dd"]),
                win_rate=float(s["win_rate"]),
                source=str(path.relative_to(checkpoints.parent)),
            )
        )

    return rows


def composite_score(r: Row) -> float:
    """Higher is better: return net of drawdown & fold dispersion."""
    return r.mean_return - 0.25 * r.mean_max_dd - 0.02 * r.std_return


def escape_xml(t: str) -> str:
    return (
        t.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def bar_panel(
    title: str,
    rows: list[Row],
    value_key: str,
    y_min: float,
    y_max: float,
    x0: float,
    y0: float,
    w: float,
    h: float,
    winner_label: str | None,
) -> str:
    parts: list[str] = []
    parts.append(
        f'<text x="{x0}" y="{y0 - 6}" font-size="13" font-family="system-ui,sans-serif" '
        f'font-weight="600">{escape_xml(title)}</text>'
    )
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#fafafa" stroke="#ddd"/>'
    )
    n = len(rows)
    if n == 0:
        return "".join(parts)
    gap = 4.0
    bw = (w - gap * (n + 1)) / n
    scale = h / (y_max - y_min) if y_max > y_min else 1.0
    zero_y = y0 + h - (0.0 - y_min) * scale

    if value_key == "ret":
        parts.append(
            f'<line x1="{x0}" y1="{zero_y}" x2="{x0 + w}" y2="{zero_y}" '
            f'stroke="#999" stroke-dasharray="4 3"/>'
        )

    for i, r in enumerate(rows):
        bx = x0 + gap + i * (bw + gap)
        if value_key == "ret":
            val_pct = r.mean_return * 100.0
            if val_pct >= 0:
                rect_h = (val_pct - 0.0) * scale
                rect_y = zero_y - rect_h
            else:
                rect_h = (0.0 - val_pct) * scale
                rect_y = zero_y
            fill = "#2d6cdf" if r.symbol == "BTC" else "#c44eb5"
        else:
            val_pct = r.mean_max_dd * 100.0
            rect_h = (val_pct - y_min) * scale
            rect_y = y0 + h - rect_h
            fill = "#e07a2f"

        stroke = "#d4af37" if winner_label and r.label == winner_label else "#333"
        sw = 2.5 if stroke == "#d4af37" else 0.6
        parts.append(
            f'<rect x="{bx:.1f}" y="{rect_y:.1f}" width="{bw:.1f}" height="{max(rect_h, 0.5):.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        short = r.label.replace("v8_seq_", "").replace("v8_", "")
        if len(short) > 14:
            short = short[:12] + "…"
        tx = bx + bw / 2
        parts.append(
            f'<text x="{tx:.1f}" y="{y0 + h + 14}" font-size="8" font-family="system-ui,sans-serif" '
            f'text-anchor="end" transform="rotate(-52 {tx:.1f} {y0 + h + 14})">'
            f"{escape_xml(short)}</text>"
        )

    parts.append(
        f'<text x="{x0 + w - 4}" y="{y0 + 12}" font-size="9" font-family="system-ui,sans-serif" '
        f'text-anchor="end" fill="#666">{y_max:.1f}%</text>'
    )
    parts.append(
        f'<text x="{x0 + w - 4}" y="{y0 + h - 4}" font-size="9" font-family="system-ui,sans-serif" '
        f'text-anchor="end" fill="#666">{y_min:.1f}%</text>'
    )
    return "".join(parts)


def build_svg(rows: list[Row]) -> str:
    btc = [r for r in rows if r.symbol == "BTC"]
    eth = [r for r in rows if r.symbol == "ETH"]
    btc_w = max(btc, key=composite_score).label if btc else None
    eth_w = max(eth, key=composite_score).label if eth else None

    def sort_key(r: Row) -> tuple[float, str]:
        return (-composite_score(r), r.label)

    btc_s = sorted(btc, key=sort_key)
    eth_s = sorted(eth, key=sort_key)

    ret_min = min((r.mean_return * 100 for r in rows), default=-10)
    ret_max = max((r.mean_return * 100 for r in rows), default=10)
    pad_r = max(3.0, (ret_max - ret_min) * 0.12)
    r_lo, r_hi = ret_min - pad_r, ret_max + pad_r

    m_min = 0.0
    m_max = max((r.mean_max_dd * 100 for r in rows), default=30) * 1.08

    W, H = 900, 640
    pw, ph = 400, 200
    body = [
        bar_panel("BTC — 折均收益 %", btc_s, "ret", r_lo, r_hi, 40, 50, pw, ph, btc_w),
        bar_panel("BTC — 折均 MaxDD %", btc_s, "mdd", m_min, m_max, 460, 50, pw, ph, btc_w),
        bar_panel("ETH — 折均收益 %", eth_s, "ret", r_lo, r_hi, 40, 300, pw, ph, eth_w),
        bar_panel("ETH — 折均 MaxDD %", eth_s, "mdd", m_min, m_max, 460, 300, pw, ph, eth_w),
    ]
    sc_parts = [
        '<text x="40" y="514" font-size="13" font-family="system-ui,sans-serif" '
        'font-weight="600">风险–收益散点（金边 = 该币种综合得分最高）</text>',
        f'<rect x="40" y="520" width="820" height="90" fill="#fafafa" stroke="#ddd"/>',
    ]
    max_x = max(r.mean_max_dd * 100 for r in rows) * 1.12 if rows else 30
    all_r = [r.mean_return * 100 for r in rows]
    min_r, max_r = (min(all_r), max(all_r)) if all_r else (-5, 5)
    pad = max(2.0, (max_r - min_r) * 0.15)
    y_lo, y_hi = min_r - pad, max_r + pad
    sx = 820 / max_x
    sy = 90 / (y_hi - y_lo)
    sc_parts.append(
        '<text x="54" y="536" font-size="9" fill="#666" font-family="system-ui,sans-serif">MaxDD% →</text>'
    )
    for r in sorted(rows, key=lambda r: (r.symbol, r.label)):
        px = 40 + r.mean_max_dd * 100 * sx
        py = 520 + 90 - (r.mean_return * 100 - y_lo) * sy
        fill = "#2d6cdf" if r.symbol == "BTC" else "#c44eb5"
        win = (r.symbol == "BTC" and btc_w and r.label == btc_w) or (
            r.symbol == "ETH" and eth_w and r.label == eth_w
        )
        stroke = "#d4af37" if win else "#222"
        sw = 3 if win else 1.2
        sc_parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"/>'
        )
    sc_parts.append(
        '<text x="450" y="628" font-size="10" text-anchor="middle" '
        'font-family="system-ui,sans-serif" fill="#444">'
        "横轴折均 MaxDD，纵轴折均收益；蓝=BTC 紫=ETH</text>"
    )
    sc = "".join(sc_parts)

    note = (
        f"综合得分 = 折均收益 − 0.25×折均MaxDD − 0.02×折均收益标准差（越高越好）。"
        f"当前最优：BTC → {escape_xml(btc_w or '—')}；ETH → {escape_xml(eth_w or '—')}。"
    )
    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="450" y="28" font-size="16" font-family="system-ui,sans-serif" font-weight="700"
        text-anchor="middle">RL 模型 OOS 对比（walk-forward 20 折 × 14 天）</text>
  <text x="450" y="46" font-size="10" font-family="system-ui,sans-serif" fill="#555" text-anchor="middle">
    {escape_xml(note)}
  </text>
  {''.join(body[:4])}
  {sc}
  <text x="450" y="{H - 8}" font-size="9" fill="#888" text-anchor="middle" font-family="system-ui,sans-serif">
    数据来源：各 run 下 wf_14d.eval.json / wf_14d_best.eval.json；v7_long 为 best_agent.eval.json
  </text>
</svg>
"""
    return svg


def main() -> None:
    python_root = Path(__file__).resolve().parents[1]
    repo_root = python_root.parent
    checkpoints = python_root / "checkpoints"
    out_dir = checkpoints / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(checkpoints)
    if not rows:
        raise SystemExit("No eval JSON found under checkpoints/")
    svg = build_svg(rows)
    out = out_dir / "model_oos_comparison.svg"
    out.write_text(svg, encoding="utf-8")
    docs_asset = repo_root / "docs" / "assets" / "model_oos_comparison.svg"
    docs_asset.parent.mkdir(parents=True, exist_ok=True)
    docs_asset.write_text(svg, encoding="utf-8")
    print(f"Wrote {out} and {docs_asset} ({len(rows)} series)")


if __name__ == "__main__":
    main()
