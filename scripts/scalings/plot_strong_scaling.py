import pandas

from jormi.ww_plots import manage_plots


def main():
    df = pandas.read_csv("strong_scaling.csv")
    df["gpu_count"] = pandas.to_numeric(df["gpu_count"], errors="coerce")
    df["us_per_zone_update"] = pandas.to_numeric(df["us_per_zone_update"], errors="coerce")
    df = df.dropna(subset=["gpu_count", "us_per_zone_update", "scheme1", "scheme2"])
    color_map = {
        "Q26": "cornflowerblue",
        "B25": "orangered",
        "FS18": "forestgreen",
    }
    marker_map = {
        "LD04": "o",
        "B25": "D",
    }
    linestyle_map = {
        "LD04": "-",
        "B25": "--",
    }
    marker_size_map = {
        "LD04": 60,
        "B25": 120,
    }
    df["updates_per_s_per_gpu"] = 1e6 / (df["us_per_zone_update"] * df["gpu_count"])
    fig, ax = manage_plots.create_figure()
    scheme1_order = ["Q26", "B25", "FS18"]
    scheme2_order = ["LD04", "B25"]
    df["scheme1"] = pandas.Categorical(df["scheme1"], categories=scheme1_order, ordered=True)
    df["scheme2"] = pandas.Categorical(df["scheme2"], categories=scheme2_order, ordered=True)
    df = df.sort_values(["scheme1", "scheme2", "gpu_count"])
    for (scheme1, scheme2), group in df.groupby(["scheme1", "scheme2"], sort=False):
        scheme1 = str(scheme1)
        scheme2 = str(scheme2)
        ax.scatter(
            group["gpu_count"],
            group["updates_per_s_per_gpu"],
            c="none",
            s=marker_size_map[scheme2],
            marker=marker_map[scheme2],
            edgecolors=color_map[scheme1],
            linewidths=1.5,
            label=f"{scheme1} + {scheme2}",
        )
        first_value = group.sort_values("gpu_count")["updates_per_s_per_gpu"].iloc[0]
        ax.axhline(
            first_value,
            linestyle=linestyle_map[scheme2],
            linewidth=1.2,
            color=color_map[scheme1],
            alpha=0.35,
        )
    ax.set_xlabel(r"GPU count")
    ax.set_ylabel(r"Z. updates $/$ s. $/$ GPU")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(2**1, 2**(9.5))
    ax.set_ylim(8e6, 6e7)
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(
    #     by_label.values(),
    #     by_label.keys(),
    #     loc="lower left",
    #     frameon=False,
    #     fontsize=16,
    #     ncol=1,
    #     labelspacing=0.375,
    #     columnspacing=1.2,
    #     handletextpad=0.4,
    #     borderpad=0.2,
    # )
    manage_plots.save_figure(fig, "strong_scaling.pdf")


if __name__ == "__main__":
    main()
