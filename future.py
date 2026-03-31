import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from typing import Callable


# ============================
#   MAIN PIXEL SORT FUNCTION
# ============================
def pixel_sorting(
    image: Image.Image,
    calculate_value: Callable[[np.ndarray], np.ndarray],
    apply_condition: Callable[[np.ndarray], np.ndarray],
    rotation: int = 0,
) -> Image.Image:

    def sort_interval(interval: np.ndarray, interval_index: int) -> np.ndarray:
        return np.argsort(interval) + interval_index

    def process_row(row: int, row_values: np.ndarray):
        interval_indices = np.flatnonzero(edges[row])
        split_values = np.split(row_values, interval_indices)

        # Sort only intervals after the first run
        for idx, interval in enumerate(split_values[1:]):
            split_values[idx + 1] = sort_interval(interval, interval_indices[idx])

        # First block: leave unchanged
        split_values[0] = np.arange(split_values[0].size)

        merged_order = np.concatenate(split_values).astype("uint32")

        # Apply reordering to each channel
        for c in range(rotated_pixels.shape[-1]):
            rotated_pixels[row, :, c] = rotated_pixels[row, merged_order, c]

    # Load pixels
    pixel_array = np.array(image)

    # Rotate if needed
    rotated_pixels = np.rot90(pixel_array, rotation)

    # Calculate per-pixel metric (luminance, hue, saturation, etc.)
    pixel_values = calculate_value(rotated_pixels)

    # Detect edges (condition mask → convolve)
    edges = np.apply_along_axis(
        lambda row: np.convolve(row, [-1, 1], "same"),
        0,
        apply_condition(pixel_values),
    )

    # Sort each row
    for row, (row_values, _) in enumerate(zip(pixel_values, edges)):
        process_row(row, row_values)

    # Rotate result back
    return Image.fromarray(np.rot90(rotated_pixels, -rotation))


# ============================
#   CONDITION FUNCTION
# ============================
def apply_condition(lum, threshold=0.5):
    return lum > threshold


# ============================
#   SORT MODES
# ============================
def hue(pixels):
    r, g, b = np.split(pixels, 3, 2)
    return np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)[:, :, 0]


def sat(pixels):
    r, g, b = np.split(pixels, 3, 2)
    maximum = np.maximum(r, np.maximum(g, b))
    minimum = np.minimum(r, np.minimum(g, b))

    # FIX: avoid divide-by-zero warning, match notebook behavior
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = (maximum - minimum) / maximum
        sat[np.isnan(sat)] = 0

    return sat[:, :, 0]


def laplace(pixels):
    lum = np.average(pixels, 2) / 255.0
    return np.abs(
        convolve2d(
            lum,
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
            mode="same",
        )
    )


# ============================
#   SCRIPT ENTRY POINT
# ============================
if __name__ == "__main__":

    img_path = r"C:\Users\Welcome-Pc\Downloads\WhatsApp Image 2026-03-31 at 3.18.57 PM.jpeg"
    img = Image.open(img_path)

    # 1. LUMINANCE SORT
    out1 = pixel_sorting(
        img,
        lambda p: np.average(p, axis=2) / 255,
        lambda lum: apply_condition(lum, threshold=0.25) - 90,
    )
 

    # 2. HUE SORT
    out2 = pixel_sorting(img, hue, lambda lum: apply_condition(lum, 0.25), -90)
 

    # 3. SATURATION SORT
    out3 = pixel_sorting(img, sat, lambda lum: apply_condition(lum, 0.25), -90)
    

    # 4. LAPLACE SORT
    out4 = pixel_sorting(img, laplace, lambda lum: apply_condition(lum, 0.25), -90)
   

    # 5. CHAIN MULTIPLE EFFECTS
    img_copy = img.copy()
    for effect in [sat, hue, laplace]:
        img_copy = pixel_sorting(
            img_copy,
            effect,
            lambda lum: apply_condition(lum, 0.25),
            -90,
        )
    img_copy.save("sorted_combined.png")

    print("Done. Saved image.")