import numpy
import pandas

from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from jormi.ww_plots import manage_plots


def _add_top_axis_resolution(ax, *, domain_length: float = 1.0) -> None:
    """
    Add a top x-axis labeling resolution N corresponding to the bottom-axis dx.

    Assumes dx = L / N, so N = L / dx.
    """
    top_ax = ax.twiny()
    top_ax.set_xscale("log")

    # Keep the same x-limits so ticks align vertically
    x0, x1 = ax.get_xlim()
    top_ax.set_xlim(x0, x1)

    # Choose some "nice" N tick labels (powers of 4), and only keep those inside the current x-range.
    # Convert N -> dx via dx = L/N for positioning.
    candidate_N = numpy.array([4**2, 4**3, 4**4, 4**5, 4**6], dtype=float)
    candidate_dx = domain_length / candidate_N

    mask = (candidate_dx <= max(x0, x1)) & (candidate_dx >= min(x0, x1))
    tick_dx = candidate_dx[mask]
    tick_N = candidate_N[mask].astype(int)

    top_ax.set_xticks(tick_dx)
    top_ax.set_xticklabels([f"{n}$^3$" for n in tick_N])

    top_ax.set_xlabel(r"resolution")

    # --- prevent minor ticks (and any bottom ticks) on the top axis ---
    top_ax.minorticks_off()
    top_ax.xaxis.set_minor_locator(NullLocator())
    top_ax.tick_params(axis="x", which="minor", top=False, bottom=False)
    top_ax.tick_params(axis="x", which="major", top=True, bottom=False, direction="in")


def main():
    df = pandas.read_csv("wave_convergence.csv")

    # --- types / clean ---
    df["dx"] = pandas.to_numeric(df["dx"], errors="coerce")
    df["error"] = pandas.to_numeric(df["error"], errors="coerce")
    df["interp_order"] = pandas.to_numeric(df["interp_order"], errors="coerce")
    df = df.dropna(subset=["dx", "error", "test", "emf_scheme", "ave_scheme", "interp_order"])
    df = df[(df["dx"] > 0) & (df["error"] > 0)]

    # --- scale by dx^-2 (i.e. plot error / dx^2 so 2nd-order becomes flat) ---
    df["scaled_error"] = df["error"] / (df["dx"] ** 2)

    # --- encodings ---
    emf_scheme_order = ["Q26", "B25", "FS18"]
    ave_scheme_order = ["LD04", "B25"]
    interp_order_order = [2, 3, 5]

    color_map = {
        "Q26": "cornflowerblue",
        "B25": "orangered",
        "FS18": "forestgreen",
    }
    marker_map = {
        "LD04": "D",
        "B25": "o",
    }
    marker_size_map = {
        "LD04": 10,
        "B25": 5,
    }
    fill_map = {
        "LD04": 0,
        "B25": 1,
    }
    skip_map = {
        "Q26": False,
        "B25": False,
        "FS18": False,
    }

    linestyle_map = {
        2: ":",
        3: "--",
        5: "-",
    }

    # Deterministic ordering
    df["emf_scheme"] = pandas.Categorical(df["emf_scheme"], categories=emf_scheme_order, ordered=True)
    df["ave_scheme"] = pandas.Categorical(df["ave_scheme"], categories=ave_scheme_order, ordered=True)
    df["interp_order"] = pandas.Categorical(df["interp_order"], categories=interp_order_order, ordered=True)
    df = df.sort_values(["test", "emf_scheme", "ave_scheme", "interp_order", "dx"])

    tests = list(df["test"].unique())

    fig, axs = manage_plots.create_figure(num_rows=1, num_cols=2, share_x=True, share_y=True)
    axes = [axs[0, 0], axs[0, 1]]

    for ax, test in zip(axes, tests):
        dft = df[df["test"] == test]

        for (emf_scheme, ave_scheme, p), g in dft.groupby(
            ["emf_scheme", "ave_scheme", "interp_order"],
            sort=False,
        ):
            emf_scheme = str(emf_scheme)
            ave_scheme = str(ave_scheme)
            p = int(p)

            if skip_map[emf_scheme]:
                continue

            ax.plot(
                g["dx"],
                g["scaled_error"],
                color=color_map[emf_scheme],
                marker=marker_map[ave_scheme],
                markersize=marker_size_map[ave_scheme],
                markerfacecolor=color_map[emf_scheme] if fill_map[ave_scheme] else "none",
                markeredgewidth=1.5,
                linestyle=linestyle_map[p],
                linewidth=1.5,
            )

        ax.set_title(test)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim([1e-1, 0.8e-4])
        ax.set_ylim([0.5e-6, 1e-3])

        # Add top axis with resolution labels (no minor ticks)
        _add_top_axis_resolution(ax, domain_length=1.0)

    fig.supxlabel(r"$\Delta x$")
    fig.supylabel(r"error scaled by $(\Delta x)^{2}$")

    # --- Split legends: colour (emf_scheme), marker (ave_scheme), linestyle (interp_order) ---
    legend_emf_scheme = [
        Line2D([0], [0], color=color_map[s], lw=2.2, label=s)
        for s in emf_scheme_order
        if s in color_map
    ]
    legend_ave_scheme = [
        Line2D([0], [0], color="black", marker=marker_map[s], linestyle="None", markersize=7, label=s)
        for s in ave_scheme_order
        if s in marker_map
    ]
    legend_order = [
        Line2D([0], [0], color="black", linestyle=linestyle_map[p], lw=2.2, label=f"p={p}")
        for p in interp_order_order
        if p in linestyle_map
    ]

    # ax0 = axes[0]
    # leg1 = ax0.legend(handles=legend_emf_scheme, title="emf_scheme", frameon=False, loc="upper right")
    # ax0.add_artist(leg1)
    # leg2 = ax0.legend(handles=legend_ave_scheme, title="ave_scheme", frameon=False, loc="lower left")
    # ax0.add_artist(leg2)
    # ax0.legend(handles=legend_order, title="interp", frameon=False, loc="lower right")

    manage_plots.save_figure(fig, "wave_convergence-2.pdf")


if __name__ == "__main__":
    main()
