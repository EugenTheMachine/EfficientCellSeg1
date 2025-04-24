import streamlit as st

from data.utils import read_image_to_numpy, read_mask_to_numpy, remap_mask_color
from gui.pages.utils.web_utils import (
    PRED_MASK_DIR,
    SUPPORT_EXTENSION,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
    delete_file,
    initialize_session_state,
    list_files,
)
from visualize_cell import InstanceVisualizer, VisItem, VisStyle, match_masks


def file_upload_section():
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            test_image_files = st.file_uploader(
                "Upload Test Images",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="visualize_test_image_files",
            )

            if test_image_files:
                TEST_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                for uploaded_file in test_image_files:
                    with open(TEST_IMAGE_DIR / uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("Images uploaded successfully.")

            existing_test_images = list_files(TEST_IMAGE_DIR)
            with st.expander(f"Existing {len(existing_test_images)} Files:", expanded=True):
                if existing_test_images:
                    for img in existing_test_images:
                        col_img, col_del = st.columns([3, 1])
                        with col_img:
                            st.text(img)
                        with col_del:
                            delete_button = st.button(
                                "Delete",
                                key=f"delete_visualize_test_image_{img}",
                                on_click=lambda img=img: delete_file(TEST_IMAGE_DIR, img),
                                help=f"Delete {img}",
                            )
                else:
                    st.info("No images uploaded yet.")

        with col2:
            test_mask_files = st.file_uploader(
                "Upload True Masks",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="visualize_test_mask_files",
            )

            if test_mask_files:
                TEST_MASK_DIR.mkdir(parents=True, exist_ok=True)
                for uploaded_file in test_mask_files:
                    with open(TEST_MASK_DIR / uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("True masks uploaded successfully.")

            existing_test_masks = list_files(TEST_MASK_DIR)
            with st.expander(f"Existing {len(existing_test_masks)} Files:", expanded=True):
                if existing_test_masks:
                    for msk in existing_test_masks:
                        col_msk, col_del = st.columns([3, 1])
                        with col_msk:
                            st.text(msk)
                        with col_del:
                            delete_button = st.button(
                                "Delete",
                                key=f"delete_visualize_test_mask_{msk}",
                                on_click=lambda msk=msk: delete_file(TEST_MASK_DIR, msk),
                                help=f"Delete {msk}",
                            )
                else:
                    st.info("No true masks uploaded yet.")

        with col3:
            test_image_files = st.file_uploader(
                "Upload Prediction Images",
                accept_multiple_files=True,
                type=SUPPORT_EXTENSION,
                key="visualize_predict_image_files",
            )

            if test_image_files:
                PRED_MASK_DIR.mkdir(parents=True, exist_ok=True)
                for uploaded_file in test_image_files:
                    with open(PRED_MASK_DIR / uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.success("Prediction images uploaded successfully.")

            existing_predict_masks = list_files(PRED_MASK_DIR)
            with st.expander(f"Existing {len(existing_predict_masks)} Files:", expanded=True):
                if existing_predict_masks:
                    for img in existing_predict_masks:
                        col_img, col_del = st.columns([3, 1])
                        with col_img:
                            st.text(img)
                        with col_del:
                            delete_button = st.button(  # noqa: F841
                                "Delete",
                                key=f"delete_visualize_predict_image_{img}",
                                on_click=lambda img=img: delete_file(PRED_MASK_DIR, img),
                                help=f"Delete {img}",
                            )
                else:
                    st.info("No prediction images uploaded yet.")


def visualization_settings():
    with st.expander("Advance Settings", expanded=False):
        col1, col2, col3, col4, col5 = st.columns(5, vertical_alignment="bottom")

        with col1:
            st.session_state.vis_sync_axes = st.checkbox("Sync Axes", st.session_state.vis_sync_axes)
        with col2:
            st.session_state.vis_true_mask_color = st.color_picker(
                "Ground Truth Color", st.session_state.vis_true_mask_color
            )
        with col3:
            st.session_state.vis_pred_mask_color = st.color_picker(
                "Prediction Color", st.session_state.vis_pred_mask_color
            )
        with col4:
            st.session_state.vis_true_mask_line_width = st.slider(
                "True Line Width", 1, 5, st.session_state.vis_true_mask_line_width
            )
        with col5:
            st.session_state.vis_pred_mask_line_width = st.slider(
                "Predict Line Width", 1, 5, st.session_state.vis_pred_mask_line_width
            )


def build_visualization_items(image, true_mask=None, pred_mask=None):
    if true_mask is not None:
        true_mask = remap_mask_color(true_mask)
    if pred_mask is not None:
        pred_mask = remap_mask_color(pred_mask)

    items = [VisItem(image=image, style=VisStyle(display_mode="image", title="Image"))]

    if true_mask is not None and pred_mask is not None:
        matched_pred_mask = match_masks(true_mask, pred_mask)
        items.extend(
            [
                VisItem(
                    image=image,
                    true_mask=true_mask,
                    style=VisStyle(
                        display_mode="contour",
                        title="Ground Truth",
                        true_mask_color=st.session_state.vis_true_mask_color,
                        true_mask_line_width=st.session_state.vis_true_mask_line_width,
                    ),
                ),
                VisItem(
                    image=image,
                    pred_mask=matched_pred_mask,
                    style=VisStyle(
                        display_mode="contour",
                        title="Prediction",
                        pred_mask_color=st.session_state.vis_pred_mask_color,
                        pred_mask_line_width=st.session_state.vis_pred_mask_line_width,
                    ),
                ),
                VisItem(
                    image=image,
                    true_mask=true_mask,
                    pred_mask=matched_pred_mask,
                    style=VisStyle(
                        display_mode="error_map",
                        title="Error Map",
                        error_false_positive_color=st.session_state.vis_error_false_positive_color,
                        error_false_negative_color=st.session_state.vis_error_false_negative_color,
                    ),
                ),
                VisItem(
                    image=image, true_mask=true_mask, style=VisStyle(display_mode="mask", title="Ground Truth Mask")
                ),
                VisItem(
                    image=image,
                    pred_mask=matched_pred_mask,
                    style=VisStyle(display_mode="mask", title="Prediction Mask"),
                ),
            ]
        )
    elif true_mask is not None:
        items.extend(
            [
                VisItem(
                    image=image,
                    true_mask=true_mask,
                    style=VisStyle(
                        display_mode="contour",
                        title="Ground Truth",
                        true_mask_color=st.session_state.vis_true_mask_color,
                        true_mask_line_width=st.session_state.vis_true_mask_line_width,
                    ),
                ),
                VisItem(
                    image=image, true_mask=true_mask, style=VisStyle(display_mode="mask", title="Ground Truth Mask")
                ),
            ]
        )
    elif pred_mask is not None:
        items.extend(
            [
                VisItem(
                    image=image,
                    pred_mask=pred_mask,
                    style=VisStyle(
                        display_mode="contour",
                        title="Prediction",
                        pred_mask_color=st.session_state.vis_pred_mask_color,
                        pred_mask_line_width=st.session_state.vis_pred_mask_line_width,
                    ),
                ),
                VisItem(image=image, pred_mask=pred_mask, style=VisStyle(display_mode="mask", title="Prediction Mask")),
            ]
        )

    return items


if __name__ == "__main__":
    st.set_page_config(page_title="Instance Segmentation Visualization", layout="wide")
    st.title("Instance Segmentation Visualization")

    initialize_session_state()

    file_upload_section()

    visualization_settings()

    col1, col2, col3 = st.columns(3)
    with col1:
        images = list_files(TEST_IMAGE_DIR)
        selected_image = st.selectbox("Select Image", ["None"] + images)

    with col2:
        masks = list_files(TEST_MASK_DIR)
        selected_mask = st.selectbox("Select Ground Truth", ["None"] + masks)

    with col3:
        preds = list_files(PRED_MASK_DIR)
        selected_pred = st.selectbox("Select Prediction", ["None"] + preds)

    if st.button("Generate Visualization", use_container_width=True, type="primary"):
        if selected_image != "None":
            try:
                image = read_image_to_numpy(TEST_IMAGE_DIR / selected_image)
                true_mask = None if selected_mask == "None" else read_mask_to_numpy(TEST_MASK_DIR / selected_mask)
                pred_mask = None if selected_pred == "None" else read_mask_to_numpy(PRED_MASK_DIR / selected_pred)

                if true_mask is not None and pred_mask is not None:
                    n_cols = 3
                elif true_mask is not None or pred_mask is not None:
                    n_cols = 3
                else:
                    n_cols = 1

                vis = InstanceVisualizer(
                    n_cols=n_cols,
                    sync_axes=st.session_state.vis_sync_axes,
                    subplot_size=(300, 300)
                )

                items = build_visualization_items(image, true_mask, pred_mask)

                fig = vis.plot(items)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
        else:
            st.warning("Please select an image to visualize.")
