import plotly.io as pio
import streamlit as st

from data.utils import calcualte_overlap_box, masks_to_bboxes
from figures.utils.load_data import load_extended_figure_2_data
from visualize_cell import plot_image_with_mask_and_boxes

if __name__ == "__main__":
    image_ids_for_all_datasets = {
        "cellseg_blood": [7],
    }

    box_width = 3
    contour_width = 0
    box_dash = "solid"
    contour_dash = "solid"

    image, true_mask, contour_color, box_color, overlap_box_color = load_extended_figure_2_data()
    image_size = image.shape[:2]
    selected_mask = true_mask.copy()

    boxs = masks_to_bboxes(selected_mask)
    fig = plot_image_with_mask_and_boxes(
        image,
        selected_mask,
        boxs,
        contour_color,
        box_color,
        box_width,
        contour_width,
        box_dash,
        contour_dash,
        image_size,
    )
    box_overlap = calcualte_overlap_box(boxs[0], boxs[1])
    x_min, y_min, x_max, y_max = box_overlap
    fig.add_shape(
        type="rect",
        x0=x_min,
        y0=y_min,
        x1=x_max,
        y1=y_max,
        line=dict(color=overlap_box_color, width=box_width, dash=box_dash),
        fillcolor="rgba(0,0,0,0)",
    )
    fig.show()
    st.plotly_chart(fig, use_container_width=False, theme=None)
    if st.download_button(
        label="Download PDF",
        data=pio.to_image(fig, format="pdf"),
        file_name="extended_figure_3_1.pdf",
        mime="application/pdf",
    ):
        pass
