"""
Per-channel (R/G/B) color histogram explainability view for a result artwork.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


class HistogramVisualizer:
    """Computes and renders the RGB color histogram of an artwork image."""

    BINS = 25
    INTENSITY_RANGE = (0, 256)

    def compute_histogram(self, image):
        """Computes the color histogram for a BGR image and returns statistics."""
        hist_b = cv2.calcHist([image], [0], None, [self.BINS], list(self.INTENSITY_RANGE)).flatten()
        hist_g = cv2.calcHist([image], [1], None, [self.BINS], list(self.INTENSITY_RANGE)).flatten()
        hist_r = cv2.calcHist([image], [2], None, [self.BINS], list(self.INTENSITY_RANGE)).flatten()

        pixels = image.reshape(-1, 3)
        count = len(pixels)
        r_mean = np.mean(pixels[:, 2])
        g_mean = np.mean(pixels[:, 1])
        b_mean = np.mean(pixels[:, 0])
        r_std = np.std(pixels[:, 2])
        g_std = np.std(pixels[:, 1])
        b_std = np.std(pixels[:, 0])

        return hist_r, hist_g, hist_b, {
            'count': count,
            'r_mean': r_mean,
            'g_mean': g_mean,
            'b_mean': b_mean,
            'r_std': r_std,
            'g_std': g_std,
            'b_std': b_std,
        }

    def draw_histogram(self, hist_r, hist_g, hist_b, stats):
        """Draws the color histogram with the reference image layout."""
        fig = plt.figure(figsize=(5, 7))
        fig.patch.set_facecolor('white')

        channels = [hist_r, hist_g, hist_b]
        colors = ['red', 'green', 'blue']

        for idx, (hist, color) in enumerate(zip(channels, colors)):
            ax = fig.add_subplot(3, 1, idx + 1)
            ax.set_facecolor('white')

            bin_width = self.INTENSITY_RANGE[1] / self.BINS
            bin_edges = np.arange(0, self.INTENSITY_RANGE[1] + bin_width, bin_width)
            bin_centers = bin_edges[:-1] + bin_width / 2

            ax.bar(bin_centers, hist, width=bin_width * 0.9, color='black', edgecolor='black')
            ax.set_xlim(0, 255)
            ax.set_ylim(0, np.max(hist) * 1.1)
            ax.set_xlabel('Intensity')
            ax.set_ylabel('Frequency')
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            cmap = LinearSegmentedColormap.from_list(f'{color}_gradient', ['black', color])

            ax_gradient = ax.twinx()
            ax_gradient.imshow(gradient, aspect='auto', extent=[0, 255, 0, 1], cmap=cmap, alpha=0.7)
            ax_gradient.set_yticks([])
            ax_gradient.set_xlim(0, 255)

        stats_text = (
            f"Count: {int(stats['count']):,}\n"
            f"rMean: {stats['r_mean']:.2f}\t\trStdDev: {stats['r_std']:.2f}\n"
            f"gMean: {stats['g_mean']:.2f}\t\tgStdDev: {stats['g_std']:.2f}\n"
            f"bMean: {stats['b_mean']:.2f}\t\tbStdDev: {stats['b_std']:.2f}"
        )
        fig.text(0.1, 0.02, stats_text, fontsize=9, verticalalignment='bottom', fontfamily='monospace')

        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        return fig

    def get_histogram_figure(self, image: Image.Image) -> plt.Figure:
        """Compute the RGB color histogram figure for an already-loaded artwork image."""
        cv_img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        hist_r, hist_g, hist_b, stats = self.compute_histogram(cv_img)
        return self.draw_histogram(hist_r, hist_g, hist_b, stats)

    # ------------------------------------------------------------------
    # Streamlit dialog
    # ------------------------------------------------------------------
    def show_dialog(self, title: str, image: Image.Image):
        """Compute the color histogram and open a Streamlit dialog to display it."""

        @st.dialog(f"Color Histogram — {title}", width="large")
        def _render_dialog():
            with st.spinner("Computing color histogram…"):
                try:
                    fig = self.get_histogram_figure(image)
                except Exception as e:
                    st.error(f"Could not compute color histogram: {e}")
                    return

            st.pyplot(fig)
            plt.close(fig)

        _render_dialog()
