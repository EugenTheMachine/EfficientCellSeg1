import plotly.io as pio
import streamlit as st

from data.utils import masks_to_bboxes
from figures.utils.load_data import load_extended_figure_2_data
from visualize_cell import plot_image_with_mask_and_boxes

if __name__ == "__main__":
    image_ids_for_all_datasets = {
        "cellseg_blood": [7],
    }

    box_width = 0
    contour_width = 3
    box_dash = "solid"
    contour_dash = "solid"

    image, true_mask, contour_color, box_color, overlap_box_color = load_extended_figure_2_data()
    image_size = image.shape[:2]
    selected_mask = true_mask.copy()
    selected_mask[selected_mask > 2] = 0

    boxs = masks_to_bboxes(true_mask)
    fig = plot_image_with_mask_and_boxes(
        image=image,
        mask=selected_mask,
        boxs=boxs,
        contour_color=contour_color,
        contour_width=contour_width,
        contour_dash=contour_dash,
        fig_size=image_size,
        box_color=box_color,
        box_width=box_width,
        box_dash=box_dash,
    )
    fig.show()
    st.plotly_chart(fig, use_container_width=False, theme=None)
    if st.download_button(
        label="Download PDF",
        data=pio.to_image(fig, format="pdf"),
        file_name="extended_figure_3_3.pdf",
        mime="application/pdf",
    ):
        pass
