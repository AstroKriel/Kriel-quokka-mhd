import pandas

from jormi.ww_plots import manage_plots


def main():
    df = pandas.read_csv("weak_scaling.csv")  # rename as needed

    # types / clean
    df["num_gpus"] = pandas.to_numeric(df["num_gpus"], errors="coerce")
    df["us_per_zone_update"] = pandas.to_numeric(df["us_per_zone_update"], errors="coerce")
    df = df.dropna(subset=["num_gpus", "us_per_zone_update", "compute_scheme", "averaging_scheme"])

    # encodings
    color_map = {
        "Q26": "cornflowerblue",
        "B25b": "orangered",
        "FS18": "forestgreen",
    }
    marker_map = {
        "LD04": "o",
        "B25a": "D",
    }
    marker_size_map = {
        "LD04": 60,
        "B25a": 120,
    }

    # derived metric (us is microseconds)
    df["updates_per_s_per_gpu"] = 1e6 / (df["us_per_zone_update"] * df["num_gpus"])

    # deterministic plotting order
    compute_order = ["Q26", "B25b", "FS18"]
    avg_order = ["LD04", "B25a"]
    df["compute_scheme"] = pandas.Categorical(df["compute_scheme"], categories=compute_order, ordered=True)
    df["averaging_scheme"] = pandas.Categorical(df["averaging_scheme"], categories=avg_order, ordered=True)
    df = df.sort_values(["compute_scheme", "averaging_scheme", "num_gpus"])

    fig, ax = manage_plots.create_figure()

    for (compute, avg), g in df.groupby(["compute_scheme", "averaging_scheme"], sort=False):
        compute = str(compute)
        avg = str(avg)

        ax.scatter(
            g["num_gpus"],
            g["updates_per_s_per_gpu"],
            c="none",
            marker=marker_map[avg],
            s=marker_size_map[avg],
            edgecolors=color_map[compute],
            linewidths=1.5,
            label=f"{compute} + {avg}",
        )

        # # perfect scaling reference: horizontal line at the 1-GPU baseline for that combo (if present),
        # # otherwise use the smallest available GPU count for that combo.
        # if (g["num_gpus"] == 1).any():
        #     idx0 = g.loc[g["num_gpus"] == 1, "num_gpus"].index[0]
        #     y0 = float(g.at[idx0, "updates_per_s_per_gpu"])
        # else:
        #     idx0 = g["num_gpus"].idxmin()
        #     y0 = float(g.at[idx0, "updates_per_s_per_gpu"])

        # ax.axhline(y0, linestyle="--", linewidth=1.2, color=color_map[compute], alpha=0.35)

    ax.set_xlabel("GPU count")
    ax.set_ylabel("Zone updates per second per GPU")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(2**(-1), 2**(9.5))
    ax.set_ylim(8e6, 6e7)

    # # de-dup legend
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(
    #     by_label.values(),
    #     by_label.keys(),
    #     frameon=False,
    #     fontsize=14,
    #     ncol=2,
    #     labelspacing=0.5,
    #     columnspacing=1.2,
    #     handletextpad=0.4,
    #     borderpad=0.2,
    # )

    manage_plots.save_figure(fig, "weak_scaling.pdf")


if __name__ == "__main__":
    main()
