import numpy as np
from PIL import Image, ImageEnhance
from scipy.signal import convolve2d
from typing import Callable


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

        for idx, interval in enumerate(split_values[1:]):
            split_values[idx + 1] = sort_interval(interval, interval_indices[idx])

        split_values[0] = np.arange(split_values[0].size)
        merged_order = np.concatenate(split_values).astype("uint32")

        for c in range(rotated_pixels.shape[-1]):
            rotated_pixels[row, :, c] = rotated_pixels[row, merged_order, c]

    pixel_array = np.array(image)
    rotated_pixels = np.rot90(pixel_array, rotation)
    pixel_values = calculate_value(rotated_pixels)
    edges = np.apply_along_axis(
        lambda row: np.convolve(row, [-1, 1], "same"),
        0,
        apply_condition(pixel_values),
    )

    for row, row_values in enumerate(pixel_values):
        process_row(row, row_values)

    return Image.fromarray(np.rot90(rotated_pixels, -rotation))


def apply_condition(lum, threshold=0.5):
    return lum > threshold


def hue(pixels):
    r, g, b = np.split(pixels, 3, 2)
    return np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)[:, :, 0]


def sat(pixels):
    r, g, b = np.split(pixels, 3, 2)
    maximum = np.maximum(r, np.maximum(g, b))
    minimum = np.minimum(r, np.minimum(g, b))
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


def neon_boost(image: Image.Image, factor=2.0) -> Image.Image:
    """Enhance vibrance and contrast for a futuristic neon effect."""
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(factor)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    return image


if __name__ == "__main__":
    img_path = r"C:\Users\Welcome-Pc\Downloads\WhatsApp Image 2026-03-31 at 3.18.57 PM.jpeg"
    img = Image.open(img_path).convert("RGB")

    # Chain futuristic pixel sorts
    effects = [sat, hue, laplace]
    img_copy = img.copy()
    for effect in effects:
        img_copy = pixel_sorting(
            img_copy,
            effect,
            lambda lum: apply_condition(lum, 0.25),
            -90,
        )

    # Neon / cyberpunk enhancement
    img_copy = neon_boost(img_copy, factor=2.5)

    # Optional: add subtle noise/glitch streaks
    arr = np.array(img_copy)
    glitch = (np.random.rand(*arr.shape) * 20).astype(np.uint8)
    arr = np.clip(arr + glitch, 0, 255)
    img_copy = Image.fromarray(arr)

    img_copy.save("sorted_futuristic.png")
    print("Done. Futuristic image saved as sorted_futuristic.png")