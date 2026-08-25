# Figure migration: expected defects

Every script was written for figures drawn at a fixed size and then rescaled by LaTeX on
inclusion. Now that a figure is drawn at the width the paper prints it at, the same defects
surface in each one. Work through these without being asked.

## 1. Legend text sits too high

`legend_size` (10.34 pt) reads too large beside panel annotations at 8.92 pt. Legends name
curves the same way annotations do, so put them on the same level:

```python
default_text_sizes = style_figure.TextSizeParams()
style_figure.set_figure_params(
    figure_params=style_figure.FigureParams(
        text_size_params=style_figure.TextSizeParams(
            legend_level=default_text_sizes.annotation_level,
        ),
    ),
)
```

Tie it to `annotation_level` rather than writing the number, so the two move together.

## 2. Margins clip labels

The defaults (left 34, right 6, bottom 28, top 6 pt) hold a left y label and a bottom x
label, and nothing else. Anything further out is silently cut: a right-hand y axis, a
twinned top axis, a shared `supylabel`. Measure rather than guess, and leave each side the
~6 pt the others get.

## 3. Marks are too heavy

Sizes were chosen for a figure that was later shrunk. Against the house values:

| quantity | house | typically found |
|---|---|---|
| `marker_size` | 5.0 | 6 to 9.5 |
| `line_width` | 0.9 | 1.2 to 1.5 |
| `marker_edge_width` | 0.6 | 1.0 to 1.5 |

Where two mark sizes encode something, such as a small marker nesting inside a large one,
keep their ratio and anchor the larger on the house value; do not flatten them.

## 4. Legend swatches disagree with the data

Drop `marker_size=` and `line_width=` from `add_custom_legend` so a swatch inherits what
the data is drawn with, rather than both being set to numbers that happen to match.

Two exceptions, both seen in the shock-tube figures:

- Where the data is drawn deliberately small so that hundreds of points do not merge into
  a band, a swatch stays at a readable size instead of matching it.
- `add_custom_legend` fills a marker with the entry's colour and draws its edge in the
  theme's, so it cannot draw an unfilled marker with a coloured edge. A legend that needs
  that is built by hand, with a comment saying so.

## 5. Text sizes named by hand

Any `text_size=`, `fontsize=` or `labelsize=` should come out, so the calibrated levels
apply and `largest_size` moves the whole paper at once.

Search for the spaced form too. `text_size = 20` survived the first sweep because the
pattern required no space around the `=`, and it left two annotations at more than twice
the size of everything around them. The check that catches it is to measure the rendered
text sizes rather than to grep: every artist should land on one of the calibrated levels.

## 6. Labels placed outside the figure

A share of the figure width has to lie inside it. `figure.supylabel(..., x=-0.05)` puts the
label off the canvas whatever the margin holds, and no margin change will rescue it.

## 7. Panel gaps too wide

The default 10 pt assumes tick labels sit between panels. Where an axis is shared, only the
frames are in the gap, so roughly 5 pt is enough.

## 8. Aspect and width follow the paper

`figure_aspect = num_columns * panel_aspect_ratio / num_rows`, so solve rather than guess.
Width comes from how `main.tex` places the figure: `figure` is one column, `figure*` spans
the page, each scaled by its `\linewidth` fraction.

## Finally

Copy the figure into `paper/figures/results/` and recompile, without being asked.
