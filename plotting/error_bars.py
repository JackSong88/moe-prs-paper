import numpy as np
import seaborn as sns


def add_error_bars(
    plot_obj, data, x, y, yerr=None, hue=None, hue_order=None, error_kw=None, order=None
):
    """
    Add error bars to a seaborn catplot (FacetGrid) or barplot (Axes) with hue grouping.

    Parameters:
    -----------
    plot_obj : sns.FacetGrid or matplotlib.axes.Axes
        The FacetGrid object returned by sns.catplot or the Axes object returned by sns.barplot.
    data : DataFrame
        The dataframe containing the data
    x : str
        Column name for x-axis grouping
    y : str
        Column name for y-axis values
    yerr : str, optional
        Column name for error values (defaults to f'{y}_err')
    hue : str, optional
        Column name for hue grouping
    hue_order : list, optional
        Order of hue categories
    error_kw : dict, optional
        Additional keyword arguments for errorbar formatting
    order : list, optional
        Order of x-axis categories (should match the order used in catplot/barplot)
    """

    # Default error bar styling
    default_error_kw = {"ls": "", "color": "black", "capsize": 0, "capthick": 0}
    if error_kw:
        default_error_kw.update(error_kw)

    if yerr is None:
        yerr = f"{y}_err"

    is_catplot = isinstance(plot_obj, sns.FacetGrid)

    if is_catplot:
        axes = plot_obj.axes_dict
        # Extract col and row information for FacetGrid
        col_name = plot_obj.col_names[0] if plot_obj.col_names else None
        row_name = plot_obj.row_names[0] if plot_obj.row_names else None
        # Get hue info from FacetGrid if available
        if hue is None and plot_obj.hue_vars:
            hue = plot_obj.hue_vars[0]
        if hue_order is None and plot_obj.hue_names:
            hue_order = plot_obj.hue_names
    else:
        axes = {None: plot_obj}  # Treat single Axes like a dict for iteration
        col_name = None
        row_name = None
        # For a single barplot, we need to infer hue and hue_order
        if hue is None:  # Try to infer hue if not explicitly provided
            # This is a bit tricky for bare Axes. A common way is to check the legend or how bars are grouped.
            # However, direct inspection of the seaborn barplot internal structure is more reliable if available.
            # A simpler, more robust approach is to *require* `hue` to be passed for barplots if it's used.
            # But let's try to be clever if `hue` is omitted but present in the data for grouping.
            # If a barplot was created with hue, its legend handles often provide the hue order.
            # This is a heuristic.
            if plot_obj.legend_:
                legend_labels = [
                    text.get_text() for text in plot_obj.legend_.get_texts()
                ]
                # If the legend labels match distinct values in any data column, that could be our hue.
                for col in data.columns:
                    if set(legend_labels) == set(
                        data[col].astype(str).drop_duplicates().tolist()
                    ):
                        hue = col
                        hue_order = legend_labels
                        break
        elif (
            hue_order is None and hue in data.columns
        ):  # If hue is provided but order isn't
            # Try to infer hue order from the order of bars, or default to data order
            # This is complex as it depends on Seaborn's internal bar ordering.
            # A safer default is to use the unique values from the data, which
            # seaborn often sorts alphabetically by default if no order is specified.
            hue_order = data[hue].drop_duplicates().tolist()
            # If there's a legend, prioritize its order
            if plot_obj.legend_:
                legend_labels = [
                    text.get_text() for text in plot_obj.legend_.get_texts()
                ]
                if set(legend_labels) == set(hue_order):
                    hue_order = legend_labels

    # Determine master x-axis category order
    x_labels_master = None
    if order is not None:
        x_labels_master = order
    elif is_catplot:
        for temp_ax in plot_obj.axes.flat:
            temp_labels = [label.get_text() for label in temp_ax.get_xticklabels()]
            if temp_labels and not all(label == "" for label in temp_labels):
                x_labels_master = temp_labels
                break
    else:  # For a single barplot
        x_labels_master = [label.get_text() for label in plot_obj.get_xticklabels()]
        if (
            not x_labels_master
        ):  # Fallback if labels are not yet set (e.g., for empty plot)
            x_labels_master = data[x].drop_duplicates().tolist()

    if x_labels_master is None:
        x_labels_master = data[x].drop_duplicates().tolist()

    for val, ax in axes.items():
        facet_data = data.copy()
        current_col_val = None
        current_row_val = None

        if is_catplot:
            if isinstance(val, tuple):
                current_row_val, current_col_val = val
            elif col_name is not None:
                current_col_val = val
            else:
                current_row_val = val

            if col_name is not None and current_col_val is not None:
                facet_data = facet_data[facet_data[col_name] == current_col_val]
            if row_name is not None and current_row_val is not None:
                facet_data = facet_data[facet_data[row_name] == current_row_val]

        if facet_data.empty:
            continue

        x_labels = x_labels_master
        x_positions = ax.get_xticks()

        if hue and hue_order:
            num_hues = len(hue_order)

            # Re-calculating dodge distances based on standard seaborn barplot dodging
            # This assumes seaborn's default dodging logic, which is generally consistent.
            dodge_width = 0.8  # Standard total width for a group of dodged bars
            bar_width = dodge_width / num_hues
            dodge_distances = np.linspace(
                -dodge_width / 2 + bar_width / 2,
                dodge_width / 2 - bar_width / 2,
                num_hues,
            )

            lw = 2 if len(x_labels) < 5 else 0.75
            default_error_kw["lw"] = lw

            for i, hue_val in enumerate(hue_order):
                # Ensure string comparison for consistency
                hue_data = facet_data[facet_data[hue] == str(hue_val)]

                if hue_data.empty:
                    continue

                x_to_pos = {cat: pos for pos, cat in enumerate(x_labels)}

                plot_data_for_hue = []
                for x_cat in x_labels:
                    cat_data = hue_data[hue_data[x] == x_cat]
                    if not cat_data.empty:
                        plot_data_for_hue.append(
                            {
                                "x_pos": x_to_pos[x_cat] + dodge_distances[i],
                                "y": cat_data[y].iloc[0],
                                "yerr": cat_data[yerr].iloc[0],
                            }
                        )

                if plot_data_for_hue:
                    x_pos = [d["x_pos"] for d in plot_data_for_hue]
                    y_vals = [d["y"] for d in plot_data_for_hue]
                    y_errs = [d["yerr"] for d in plot_data_for_hue]

                    ax.errorbar(x_pos, y_vals, yerr=y_errs, **default_error_kw)
        else:
            lw = 2 if len(x_labels) < 5 else 0.75
            default_error_kw["lw"] = lw

            plot_data_no_hue = []
            for x_cat in x_labels:
                cat_data = facet_data[facet_data[x] == x_cat]
                if not cat_data.empty:
                    plot_data_no_hue.append(
                        {"y": cat_data[y].iloc[0], "yerr": cat_data[yerr].iloc[0]}
                    )

            if plot_data_no_hue:
                y_vals = [d["y"] for d in plot_data_no_hue]
                y_errs = [d["yerr"] for d in plot_data_no_hue]
                ax.errorbar(x_positions, y_vals, yerr=y_errs, **default_error_kw)


def add_error_bars_to_catplot(
    grid,
    data,
    x,
    y,
    yerr=None,
    hue=None,
    hue_order=None,
    col=None,
    row=None,
    error_kw=None,
    order=None,
):
    """
    Add error bars to a seaborn catplot with facets and hue grouping.

    Parameters:
    -----------
    grid : sns.FacetGrid
        The FacetGrid object returned by sns.catplot
    data : DataFrame
        The dataframe containing the data
    x : str
        Column name for x-axis grouping
    y : str
        Column name for y-axis values
    yerr : str, optional
        Column name for error values (defaults to f'{y}_err')
    hue : str, optional
        Column name for hue grouping
    hue_order : list, optional
        Order of hue categories
    col : str, optional
        Column name for column faceting
    row : str, optional
        Column name for row faceting
    error_kw : dict, optional
        Additional keyword arguments for errorbar formatting
    order : list, optional
        Order of x-axis categories (should match the order used in catplot)
    """

    # Default error bar styling
    default_error_kw = {"ls": "", "color": "black", "capsize": 0, "capthick": 0}
    if error_kw:
        default_error_kw.update(error_kw)

    if yerr is None:
        yerr = f"{y}_err"

    # Get x-axis category order once - try multiple approaches
    x_labels_master = None
    if order is not None:
        x_labels_master = order
    else:
        # Try to get labels from any subplot that has them visible
        for temp_ax in grid.axes.flat:
            temp_labels = [label.get_text() for label in temp_ax.get_xticklabels()]
            if temp_labels and not all(label == "" for label in temp_labels):
                x_labels_master = temp_labels
                break

        # Fallback: get unique categories from data
        if x_labels_master is None:
            x_labels_master = data[x].drop_duplicates().tolist()

    row_val = None
    col_val = None

    # Iterate through each subplot in the grid
    for val, ax in grid.axes_dict.items():
        # Filter data for this specific facet
        facet_data = data.copy()

        if isinstance(val, tuple):
            row_val, col_val = val
        elif col is not None:
            col_val = val
        else:
            row_val = val

        if col is not None and col_val is not None:
            facet_data = facet_data[facet_data[col] == col_val]
        if row is not None and row_val is not None:
            facet_data = facet_data[facet_data[row] == row_val]

        if facet_data.empty:
            continue

        # Use the master x_labels order for all subplots
        x_labels = x_labels_master
        x_positions = range(len(x_labels))

        if hue and hue_order:
            # Calculate dodge distances for multiple hues
            num_hues = len(hue_order)
            dodge_width = 0.8  # Total width for all bars
            bar_width = dodge_width / num_hues
            dodge_distances = np.linspace(
                -dodge_width / 2 + bar_width / 2,
                dodge_width / 2 - bar_width / 2,
                num_hues,
            )

            # Set line width based on number of groups
            lw = 2 if len(x_labels) < 5 else 0.75
            default_error_kw["lw"] = lw

            # Add error bars for each hue
            for i, hue_val in enumerate(hue_order):
                hue_data = facet_data[facet_data[hue] == hue_val]

                if hue_data.empty:
                    continue

                # Create mapping from x categories to positions
                x_to_pos = {cat: pos for pos, cat in enumerate(x_labels)}

                # Get data in the correct order
                plot_data = []
                for x_cat in x_labels:
                    cat_data = hue_data[hue_data[x] == x_cat]
                    if not cat_data.empty:
                        plot_data.append(
                            {
                                "x_pos": x_to_pos[x_cat] + dodge_distances[i],
                                "y": cat_data[y].iloc[0],
                                "yerr": cat_data[yerr].iloc[0],
                            }
                        )

                if plot_data:
                    x_pos = [d["x_pos"] for d in plot_data]
                    y_vals = [d["y"] for d in plot_data]
                    y_errs = [d["yerr"] for d in plot_data]

                    ax.errorbar(x_pos, y_vals, yerr=y_errs, **default_error_kw)
        else:
            # No hue grouping - simpler case
            lw = 2 if len(x_labels) < 5 else 0.75
            default_error_kw["lw"] = lw

            # Get data in the correct order
            plot_data = []
            for x_cat in x_labels:
                cat_data = facet_data[facet_data[x] == x_cat]
                if not cat_data.empty:
                    plot_data.append(
                        {"y": cat_data[y].iloc[0], "yerr": cat_data[yerr].iloc[0]}
                    )

            if plot_data:
                y_vals = [d["y"] for d in plot_data]
                y_errs = [d["yerr"] for d in plot_data]
                ax.errorbar(x_positions, y_vals, yerr=y_errs, **default_error_kw)
