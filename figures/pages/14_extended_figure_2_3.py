import shutil

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from data.utils import resize_image, resize_mask
from figures.utils.load_image import load_gt_images, load_pred_masks
from visualize_cell import plot_image_with_mask
from metrics import average_precision
from project_root import PROJECT_ROOT
from visualize_cell import find_contours_with_padding

st.set_page_config(layout="wide")
st.title("Image and Mask Display")


def crop_region(image, box):
    y1, y2, x1, x2 = box
    return image[x1:x2, y1:y2]


def plot_image_with_annotation(image, pred_mask, true_mask, ap50, cell_num, contour_width, image_size):
    fig = plot_image_with_mask(
        image,
        mask=true_mask,
        contour_dash="solid",
        contour_width=contour_width,
        fig_size=(image_size, image_size),
        box=None,
        contour_color="#EFEF00",
    )
    if pred_mask is not None:
        contours = find_contours_with_padding(pred_mask)
        x, y = [], []
        for contour in contours:
            y = np.concatenate([y, contour[:, 0], [None]])
            x = np.concatenate([x, contour[:, 1], [None]])
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color="#78DCE8", width=contour_width, dash="solid"),
                hoverinfo="skip",
            )
        )
    anno_font_size = int(45 * image_size / 500)
    anno_color = "#FFFFFF"

    fig.update_layout(
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
        width=image_size,
        height=image_size,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


def configure_ui():
    cols = st.columns(3)
    with cols[0]:
        image_size = st.selectbox("Image Size", [100, 250, 500], index=2)
    with cols[1]:
        contour_width = st.slider("Contour Width", min_value=0, max_value=10, value=2, step=1)
    with cols[2]:
        save_images = st.checkbox("Save Images", value=False)
    return image_size, contour_width, save_images


if __name__ == "__main__":
    train_ids_all = {
        "cellseg_blood": 138,
        "deepbacs_rod_brightfield": 6,
        "deepbacs_rod_fluorescence": 70,
        "dsb2018_stardist": 435,
        "cellpose_specialized": 60
    }
    test_ids_all = {
        "cellseg_blood": [4],
        "deepbacs_rod_brightfield": [11],
        "deepbacs_rod_fluorescence": [7],
        "dsb2018_stardist": [31],
        "cellpose_specialized": [0],
    }
    boxes = {
        "cellseg_blood": [[200, 420, 3, 223], [320, 480, 230, 390]],
        "deepbacs_rod_brightfield": [[70, 170, 360, 460], [290, 410, 210, 330]],
        "deepbacs_rod_fluorescence": [[3, 103, 280, 380], [260, 410, 170, 320]],
        "dsb2018_stardist": [[160, 280, 160, 280], [220, 420, 290, 490]],
        "cellpose_specialized": [[0, 80, 360, 440], [0, 80, 110, 190]],
    }
    rotations_for_all_datasets = {
        "cellseg_blood": [0, 0, 0, 0, 0, 0],
        "deepbacs_rod_brightfield": [0, 0, 0, 0, 0, 0],
        "deepbacs_rod_fluorescence": [0, 0, 0, 0, 0, 0],
        "dsb2018_stardist": [0, 0, 0, 0, 0, 0],
        "cellpose_specialized": [0, 0, 0, 0, 0, 0],
    }
    datasets = list(train_ids_all.keys())

    images_all = {}
    pred_masks_all = {}
    true_masks_all = {}
    ap50_all = {}
    num_cells_all = {}
    for dataset in datasets:
        train_id = train_ids_all[dataset]
        test_ids = test_ids_all[dataset]
        images, true_masks, num_cells = load_gt_images(dataset, select_ids=test_ids, train_or_test="test")
        for i, true_mask in enumerate(true_masks):
            if dataset in ["cellseg_blood", "deepbacs_rod_brightfield", "deepbacs_rod_fluorescence"]:
                true_masks[i] = resize_mask(true_mask, (512, 512))
                images[i] = resize_image(images[i], (512, 512))
        pred_masks = load_pred_masks(dataset, train_id=train_id, test_ids=test_ids)
        ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=[0.5])
        ap50_all[dataset] = {image_id: ap[i][0] for i, image_id in enumerate(test_ids)}
        pred_masks_all[dataset] = pred_masks
        images_all[dataset] = images
        num_cells_all[dataset] = num_cells
        true_masks_all[dataset] = true_masks

    image_size, contour_width, save_images = configure_ui()
    save_dir = PROJECT_ROOT / "saved_images/extended_figure_2"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    for row_idx in range(len(datasets)):
        dataset = datasets[row_idx]
        st.write(f"**Dataset: {dataset}**")
        images = images_all[dataset]
        pred_masks = pred_masks_all[dataset]
        true_masks = true_masks_all[dataset]
        num_cells = num_cells_all[dataset]
        ap_50s = ap50_all[dataset]
        image_ids = test_ids_all[dataset]
        dataset_boxes = boxes[dataset]

        cols = st.columns(2)

        for image_idx in range(len(image_ids)):
            image = images[image_idx]
            pred_mask = pred_masks[image_idx]
            true_mask = true_masks[image_idx]
            true_mask = resize_mask(true_mask, (image_size, image_size))
            pred_mask = resize_mask(pred_mask, (image_size, image_size))
            image = resize_image(image, (image_size, image_size))
            image_id = image_ids[image_idx]
            ap50 = ap_50s[image_id]
            num_cell = num_cells[image_idx]

            for box_idx, box in enumerate(dataset_boxes):
                cropped_image = crop_region(image, box)
                cropped_pred_mask = crop_region(pred_mask, box)
                cropped_true_mask = crop_region(true_mask, box)

                resize_shape = (image_size, image_size)
                cropped_image = resize_image(cropped_image, resize_shape)
                cropped_pred_mask = resize_mask(cropped_pred_mask, resize_shape)
                cropped_true_mask = resize_mask(cropped_true_mask, resize_shape)

                fig = plot_image_with_annotation(
                    cropped_image,
                    cropped_pred_mask,
                    cropped_true_mask,
                    ap50,
                    num_cell,
                    contour_width,
                    image_size,
                )

                if save_images:
                    image_file = save_dir / f"dataset_{dataset}_image_{image_id}_box_{box_idx}.pdf"
                    pio.write_image(fig, image_file)

                with cols[box_idx]:
                    st.components.v1.html(fig.to_html(include_mathjax="cdn"), height=image_size, width=image_size)
