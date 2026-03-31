import cv2
import numpy as np

# ============================================================
# Load image
# ============================================================
img = cv2.imread(r"C:\Users\Welcome-Pc\Downloads\WhatsApp Image 2026-03-31 at 3.18.57 PM.jpeg")
h, w, c = img.shape


# ============================================================
# 1. SLIGHT GAUSSIAN BLUR (soft old-lens look)
# ============================================================
blurred = cv2.GaussianBlur(img, (5, 5), 0)


# ============================================================
# 2. YELLOW TINT (old-photo warm tone)
# ============================================================
yellow = blurred.astype(np.float32)
yellow[:, :, 2] *= 1.05   # red up
yellow[:, :, 1] *= 1.10   # green up
yellow[:, :, 0] *= 0.95   # blue down
yellow = np.clip(yellow, 0, 255).astype(np.uint8)


# ============================================================
# 3. COLOR QUANTIZATION (K-means)
# ============================================================
Z = yellow.reshape((-1, 3)).astype(np.float32)
K = 6  # fewer colors = more retro
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
_, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
quantized = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)


# ============================================================
# 4. SOFT OUTLINES (Laplacian fade)
# ============================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
edges = cv2.GaussianBlur(edges, (3, 3), 0)
edges = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)[1]
edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edges = cv2.GaussianBlur(edges, (7, 7), 0)  # faded edges


# ============================================================
# 5. VIGNETTE (dark corners)
# ============================================================
kernel_x = cv2.getGaussianKernel(w, 400)
kernel_y = cv2.getGaussianKernel(h, 400)
vignette = kernel_y @ kernel_x.T
vignette = cv2.normalize(vignette, None, 0.5, 1.0, cv2.NORM_MINMAX)
vignette = np.dstack([vignette] * 3)

vintage = (quantized * vignette).astype(np.uint8)


# ============================================================
# 6. FINAL CARTOON COMBINATION
# ============================================================
cartoon = cv2.bitwise_and(vintage, edges)


# ============================================================
# 7. DATAMOSH STRIP FUNCTION
# ============================================================
def datamosh_strip(img, width=70, intensity=40):
    h, w, c = img.shape

    strip = img[:, w-width:w].copy()

    # Row tearing
    for y in range(h):
        shift = np.random.randint(-intensity, intensity)
        strip[y] = np.roll(strip[y], shift, axis=0)

    # Channel offsets
    for ch in range(3):
        offset = np.random.randint(-10, 10)
        strip[:, :, ch] = np.roll(strip[:, :, ch], offset, axis=0)

    # Block corruption
    for _ in range(60):
        y = np.random.randint(0, h-20)
        x = np.random.randint(0, width-20)
        block = strip[y:y+20, x:x+20]
        np.random.shuffle(block.reshape(-1, 3))

    return strip


# ============================================================
# 8. APPLY DATAMOSH STRIP
# ============================================================
strip = datamosh_strip(cartoon, width=70, intensity=50)
final = cartoon.copy()
final[:, -70:] = strip


# ============================================================
# 9. RESIZE ALL OUTPUT WINDOWS TO FIXED SIZE
# ============================================================

WINDOW_W = 900
WINDOW_H = 700

def show_fixed(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, WINDOW_W, WINDOW_H)
    cv2.imshow(name, img)

show_fixed("Original", img)
show_fixed("Vintage Cartoon", cartoon)
show_fixed("Datamosh Strip Applied", final)

cv2.waitKey(0)
