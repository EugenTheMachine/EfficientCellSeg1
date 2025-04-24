import plotly.io as pio
import streamlit as st
from figures.utils.load_data import load_extended_figure_3_4_data
from data.utils import read_image_to_numpy, read_mask_to_numpy
from project_root import PROJECT_ROOT
from visualize_cell import plot_image_with_mask_and_boxes, separate_masks

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    image_ids_for_all_datasets = {
        "cellseg_blood": [7],
    }
    df = load_extended_figure_3_4_data()
    used_times = df.mean().to_dict()
    used_times = {
        "box_nms": used_times["box_nms"],
        "mask_nms": used_times["mask_nms"],
        "opt_mask_nms": used_times["opt_mask_nms"],
    }
    ap50s = {
        "box_nms": 0.64044946,
        "mask_nms": 0.9111111,
        "opt_mask_nms": 0.9111111,
    }
    used_times = list(used_times.values())
    ap50s = list(ap50s.values())

    image = read_image_to_numpy(
        PROJECT_ROOT / "figures/images/extended_figure_3_4/image.png"
    )
    true_mask = read_mask_to_numpy(
        PROJECT_ROOT / "figures/images/extended_figure_3_4/true_mask.png"
    )
    pred_mask_opt_mask_nms = read_mask_to_numpy(
        PROJECT_ROOT / "figures/images/extended_figure_3_4/pred_mask_opt_mask_nms.png"
    )
    pred_mask_mask_nms = read_mask_to_numpy(
        PROJECT_ROOT / "figures/images/extended_figure_3_4/pred_mask_mask_nms.png"
    )
    pred_mask_box_nms = read_mask_to_numpy(
        PROJECT_ROOT / "figures/images/extended_figure_3_4/pred_mask_box_nms.png"
    )

    pred_masks = [pred_mask_box_nms, pred_mask_mask_nms, pred_mask_opt_mask_nms]
    contour_width = 2
    contour_dash = "solid"
    pred_hit, pred_miss, true_hit, true_miss = separate_masks(
        pred_mask_box_nms, pred_mask_mask_nms
    )

    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            h = image.shape[0]
            w = image.shape[1]
            fig = plot_image_with_mask_and_boxes(
                image=image,
                mask=pred_masks[i],
                contour_width=contour_width,
                contour_dash=contour_dash,
            )
            anno_font_size = int(45 * h / 500)
            anno_rect_width = int(w)
            anno_rect_height = int(80 * h / 500)
            latex_size = "LARGE"
            anno_color = "#FFFFFF"
            fig.add_shape(
                dict(
                    type="rect",
                    x0=0,
                    y0=0,
                    x1=anno_rect_width,
                    y1=anno_rect_height,
                    fillcolor="rgba(0, 0, 0, 0.5)",
                    line=dict(color="rgba(0, 0, 0, 0.5)", width=0, dash="solid"),
                )
            )
            fig.add_annotation(
                x=0,
                y=anno_rect_height / 2,
                xref="x",
                yref="y",
                xanchor="left",
                yanchor="middle",
                text=f"  {ap50s[i]:.2f}",
                showarrow=False,
                font=dict(
                    family="Arial",
                    color=anno_color,
                ),
            )
            fig.update_layout(
                font=dict(
                    family="Arial",
                    size=anno_font_size,
                    color=anno_color,
                ),
            )
            fig.add_annotation(
                x=anno_rect_width - 10,
                y=anno_rect_height / 2,
                xref="x",
                yref="y",
                xanchor="right",
                yanchor="middle",
                showarrow=False,
                text=f"{used_times[i]:.2f} s ",
                font=dict(
                    family="Arial",
                    size=anno_font_size,
                    color=anno_color,
                ),
            )

            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                autosize=False,
                width=w,
                height=h,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
            )
            st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=h, width=w)
            if st.download_button(
                label=f"extended_figure_3_4_{i+1}.pdf",
                data=pio.to_image(fig, format="pdf"),
                file_name=f"extended_figure_3_4_{i+1}.pdf",
                mime="application/pdf",
            ):
                pass
