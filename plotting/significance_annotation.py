import math

import numpy as np


def _annotate_significance_panel(
    ax,
    panel_df,
    x,
    y,
    yerr,
    hue,
    hue_order,
    x_labels,
    test_pairs,
    symbols,
    dodge_width=0.8,
    alpha=0.05,
    vertical_pad_frac=0.02,
    horiz_spacing_frac=0.6,
    encode_strength=True,
):
    """
    Annotate a single panel (single Axes) with significance markers.
    panel_df must already be subsetted for this facet.
    """

    # ---------- Statistical helper ----------
    def p_from_z(z):
        return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    def is_significant(m1, se1, m2, se2):
        denom = math.sqrt(se1**2 + se2**2)
        if denom == 0:
            return False, 1.0
        z = (m1 - m2) / denom
        p = p_from_z(z)
        return p < alpha, p

    def symbol_from_p(p, base_symbol):
        if not encode_strength:
            return base_symbol
        if p < 0.001:
            return base_symbol * 3
        if p < 0.01:
            return base_symbol * 2
        if p < 0.05:
            return base_symbol
        return base_symbol

    # ---------- X positions ----------
    x_to_index = {lab: i for i, lab in enumerate(x_labels)}
    x_base_positions = np.arange(len(x_labels))

    # ---------- Dodge math (EXACTLY like your errorbar function) ----------
    if hue and hue_order:
        num_hues = len(hue_order)
        bar_width = dodge_width / num_hues
        dodge_distances = np.linspace(
            -dodge_width / 2 + bar_width / 2,
            dodge_width / 2 - bar_width / 2,
            num_hues,
        )
    else:
        num_hues = 1
        bar_width = dodge_width
        dodge_distances = np.array([0.0])

    # ---------- Build lookup: (x_label, hue) -> position, mean, stderr ----------
    lookup = {}

    for x_lab in x_labels:
        if x_lab not in x_to_index:
            continue
        base_x = x_to_index[x_lab]

        for i_h, hue_val in enumerate(hue_order if hue_order else [None]):
            x_pos = base_x + dodge_distances[i_h]

            if hue:
                row = panel_df[(panel_df[x] == x_lab) & (panel_df[hue] == hue_val)]
            else:
                row = panel_df[(panel_df[x] == x_lab)]

            if row.empty:
                lookup[(x_lab, hue_val)] = None
            else:
                lookup[(x_lab, hue_val)] = {
                    "x": x_pos,
                    "mean": float(row[y].iloc[0]),
                    "stderr": float(row[yerr].iloc[0]),
                }

    # ---------- Loop over x groups, collect all annotations first ----------
    all_annotations = []  # list of (x_draw, base_y, symbol)

    for x_lab in x_labels:
        significant_items = []

        for (m1, m2), base_symbol in zip(test_pairs, symbols):
            if m1 not in hue_order or m2 not in hue_order:
                continue

            entry1 = lookup.get((x_lab, m1))
            entry2 = lookup.get((x_lab, m2))

            if not entry1 or not entry2:
                continue

            m1_mean, m1_se = entry1["mean"], entry1["stderr"]
            m2_mean, m2_se = entry2["mean"], entry2["stderr"]

            sig, p = is_significant(m1_mean, m1_se, m2_mean, m2_se)

            if not sig:
                continue

            symbol = symbol_from_p(p, base_symbol)
            pair_center = 0.5 * (entry1["x"] + entry2["x"])
            base_y = max(m1_mean + m1_se, m2_mean + m2_se)
            significant_items.append((pair_center, base_y, symbol))

        if not significant_items:
            continue

        # Horizontal stacking within an x group
        n = len(significant_items)
        spacing = horiz_spacing_frac * bar_width
        offsets = np.linspace(-(n - 1) / 2 * spacing, (n - 1) / 2 * spacing, n)

        for (pair_center, base_y, symbol), dx in zip(significant_items, offsets):
            all_annotations.append((pair_center + dx, base_y, symbol))

    if not all_annotations:
        return

    # ---------- Draw annotations, updating y-limits incrementally ----------
    # Re-read current limits each time so expanding the axis is reflected
    # in the padding calculation for subsequent symbols.
    for x_draw, base_y, symbol in all_annotations:
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min if y_max > y_min else 1.0
        vpad = vertical_pad_frac * y_range

        y_draw = base_y + vpad

        ax.text(
            x_draw,
            y_draw,
            symbol,
            ha="center",
            va="bottom",
            color="grey",
            fontweight="bold",
        )

        # Expand axis to accommodate the new annotation
        text_height_estimate = 0.04 * y_range  # rough estimate for font height
        needed_ymax = y_draw + text_height_estimate
        if needed_ymax > y_max:
            ax.set_ylim(y_min, needed_ymax)


def add_significance_annotations(
    plot_obj,
    data,
    x,
    y,
    yerr,
    hue,
    hue_order,
    x_labels,
    test_pairs,
    symbols=None,
    dodge_width=0.8,
    alpha=0.05,
):
    """
    Works for single Axes or seaborn FacetGrid.
    """

    if symbols is None:
        symbols = ["*"] * len(test_pairs)

    import matplotlib.pyplot as plt

    # ---------- Single axis ----------
    if isinstance(plot_obj, plt.Axes):
        _annotate_significance_panel(
            plot_obj,
            data,
            x,
            y,
            yerr,
            hue,
            hue_order,
            x_labels,
            test_pairs,
            symbols,
            dodge_width=dodge_width,
            alpha=alpha,
        )
        return

    # ---------- FacetGrid ----------
    if hasattr(plot_obj, "axes_dict"):
        # seaborn stores these as _col_var / _row_var, not col_var / row_var
        row_var = getattr(plot_obj, "_row_var", None) or getattr(
            plot_obj, "row_var", None
        )
        col_var = getattr(plot_obj, "_col_var", None) or getattr(
            plot_obj, "col_var", None
        )

        for key, ax in plot_obj.axes_dict.items():
            panel_df = data.copy()

            # Determine facet values and filter to this panel only
            if isinstance(key, tuple):
                if row_var and col_var:
                    row_val, col_val = key
                    panel_df = panel_df[
                        (panel_df[row_var] == row_val) & (panel_df[col_var] == col_val)
                    ]
                elif col_var:
                    panel_df = panel_df[panel_df[col_var] == key[0]]
                elif row_var:
                    panel_df = panel_df[panel_df[row_var] == key[0]]
            else:
                # key is a scalar (single facet variable)
                if col_var:
                    panel_df = panel_df[panel_df[col_var] == key]
                elif row_var:
                    panel_df = panel_df[panel_df[row_var] == key]

            if panel_df.empty:
                continue

            _annotate_significance_panel(
                ax,
                panel_df,
                x,
                y,
                yerr,
                hue,
                hue_order,
                x_labels,
                test_pairs,
                symbols,
                dodge_width=dodge_width,
                alpha=alpha,
            )
