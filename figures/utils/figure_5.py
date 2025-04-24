from figures.utils.load_data import load_color
from figures.utils.modified_circos import Circos


def create_radar_chart(df, figure_size, r_max, font_size, line_width, marker_size):
    method_colors = load_color()

    def label_kws_handler(col_name):
        return dict(size=font_size, color="black", family="Arial")

    def grid_label_formatter(label):
        return f"{label:.1f}"

    def line_kws_handler(col_name):
        return dict(lw=line_width)

    circos = Circos.radar_chart(
        df,
        vmax=1.0,
        r_lim=(0, r_max),
        fill=False,
        circular=True,
        marker_size=marker_size,
        bg_color="#FFFFFFFF",
        grid_interval_ratio=0.2,
        cmap=method_colors,
        line_kws_handler=line_kws_handler,
        show_grid_label=True,
        label_kws_handler=label_kws_handler,
        grid_label_formatter=grid_label_formatter,
        grid_label_kws=dict(size=font_size, color="black", family="Arial"),
    )

    fig = circos.plotfig(dpi=300, figsize=(figure_size, figure_size))
    return fig
