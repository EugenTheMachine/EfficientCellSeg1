import copy
import io

import pandas as pd
import streamlit as st

from figures.utils.figure_5 import create_radar_chart
from figures.utils.load_data import load_figure_5_1_data

st.set_page_config(layout="wide")
st.title("Figure_5_1")


def main():
    radar_data = load_figure_5_1_data()

    df = pd.DataFrame(
        data=radar_data,
    ).T
    figure_size = 5
    r_max = 100
    line_width = 3
    marker_size = 0
    font_size = 15

    fig = create_radar_chart(
        df,
        figure_size,
        r_max,
        font_size,
        line_width,
        marker_size,
    )
    fig_copy = copy.deepcopy(fig)
    st.pyplot(fig, use_container_width=False)
    buffer = io.BytesIO()
    fig_copy.savefig(buffer, format="pdf")
    buffer.seek(0)

    if st.download_button(
        label="Download PDF",
        data=buffer,
        file_name="figure_5_1.pdf",
        mime="application/pdf",
    ):
        pass


if __name__ == "__main__":
    main()
