import streamlit as st

st.set_page_config(page_title="CellSeg1", page_icon="🔬", layout="wide")

st.markdown("# 🔬 Welcome to CellSeg1!")

st.markdown('''
## Key Features
- Requires only one training image with a few dozen cell annotations
- Works with diverse cell types and imaging modalities
- Achieves comparable performance to models trained on hundreds of images
- Demonstrates superior cross-dataset generalization
- User-friendly GUI for training, testing, and visualization

#### GPU Memory and Batch Size Settings
- 8GB VRAM: Try `batch_size=1` with `gradient_accumulation=32`
- 12GB VRAM: Try `batch_size=2` with `gradient_accumulation=16`
- 24GB VRAM: Try `batch_size=4` with `gradient_accumulation=8`
- For detailed analysis of memory usage and training efficiency with different batch sizes, please refer to Extended Figure 5 in our paper.

#### Training and Prediction Sessions
- Browser can be closed during training/prediction - progress will be preserved
- Reopening the browser should restore the current training/prediction state

#### Troubleshooting
If you encounter persistent issues that cannot be resolved by restarting the program, try deleting these files from `/path_to_your_code_folder/cellseg1/streamlit_storage/`:
- training_state.json
- training_progress.json
- training_stop_flag
- prediction_state.json
- prediction_progress.json
- prediction_stop_flag

Then restart the program.
''')