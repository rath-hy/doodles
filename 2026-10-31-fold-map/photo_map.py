"""Fold maps from photos: fold a printed ChArUco board, photograph a view, and recover where every
visible point of the view came from on the sheet.

A ChArUco board is OpenCV's camera calibration target: a checkerboard with a small ArUco marker in
each white square. Each marker is a black-and-white code with its own number, so a marker found in
a photo says which part of the sheet that patch of paper is, and its four corners say how the
patch is turned and placed. Folding moves paper in rigid pieces, so:

  1. detect the markers in a photo of a folded view,
  2. group them into rigid pieces (markers on one piece agree on one transform, found by RANSAC),
  3. assign every pixel of paper in the photo to the piece whose transform best predicts the
     board's pattern there (local normalized cross-correlation).

The result is, for each photo pixel, the point of the printed sheet it shows: the same kind of map
fold_map.py computes from a crease pattern, but measured from real paper, for any stage of any
model, folded flat or not.

simulate_photo() folds the board virtually with a FoldedModel, adds blur, noise, uneven light and
shadows along the layer edges, and returns the true map alongside, to test the whole pipeline.

Coordinates: sheet points are in millimeters on the printed side as you look at it lying flat,
x to the right and y down from the top-left corner. Needs opencv-python 4.7 or later.
"""
from dataclasses import dataclass

import numpy as np
import cv2

from fold_map import apply
from illusion import SHEET_MM


# ---------------------------------------------------------------------------------------------
# The board


@dataclass
class Board:
    squares: int            # squares per side
    sheet_mm: float         # side of the printed square
    image: np.ndarray       # (R, R) uint8, the printed side, 0 black, 255 white
    corners: dict           # marker id -> (4, 2) corners on the sheet in mm
    dictionary: object
    charuco: object

    @property
    def px_mm(self):        # board image pixels per mm
        return self.image.shape[0] / self.sheet_mm

    @property
    def square_mm(self):
        return self.sheet_mm / self.squares


def make_board(squares=44, sheet_mm=SHEET_MM, px_per_square=40, marker_ratio=0.75):
    """A ChArUco board filling a square sheet. Half the squares hold a marker, and OpenCV's
    4x4 dictionary has 1,000 markers, so squares can go up to 44 per side: on 200 mm (the square
    the story notebooks print), 4.5 mm squares with 3.4 mm markers. More squares is better, because
    a marker cut by a crease or covered by another layer can't be read. (Folding scales with the
    paper, so what matters is squares per sheet, not their size in mm.)"""
    if squares * squares // 2 > 1000:
        raise ValueError("at most 44 squares per side: the dictionary has 1,000 markers")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    square = sheet_mm / squares
    charuco = cv2.aruco.CharucoBoard((squares, squares), square, marker_ratio * square, dictionary)
    size = squares * px_per_square
    image = charuco.generateImage((size, size), marginSize=0, borderBits=1)
    corners = {int(i): np.asarray(p, float)[:, :2] for i, p in zip(np.ravel(charuco.getIds()), charuco.getObjPoints())}
    return Board(squares, sheet_mm, image, corners, dictionary, charuco)


def save_board_pdf(board, path, page_mm=(210, 297), top_mm=10):
    """An A4 PDF with the board at actual size, centered across the page, a thin cut line around
    it, and a 100 mm scale bar to check the printer didn't rescale it. Print at 100% (not "fit to
    page")."""
    from PIL import Image, ImageDraw
    dpi = board.px_mm * 25.4
    W, H = (round(m * board.px_mm) for m in page_mm)
    page = Image.new("L", (W, H), 255)
    R = board.image.shape[0]
    x0, y0 = (W - R) // 2, round(top_mm * board.px_mm)
    page.paste(Image.fromarray(board.image), (x0, y0))
    draw = ImageDraw.Draw(page)
    draw.rectangle([x0 - 1, y0 - 1, x0 + R, y0 + R], outline=160)
    bar_y = y0 + R + round(10 * board.px_mm)
    draw.line([x0, bar_y, x0 + round(100 * board.px_mm), bar_y], fill=0, width=3)
    for x in (x0, x0 + round(100 * board.px_mm)):
        draw.line([x, bar_y - 12, x, bar_y + 12], fill=0, width=3)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=round(3 * board.px_mm))        # about 3 mm tall
    except TypeError:                                                   # Pillow before 10.1
        font = None
    draw.text((x0, bar_y + 18), f"this bar is 100 mm: print at 100%, not fit to page\n"
                                f"ChArUco board, {board.squares} x {board.squares} squares of "
                                f"{board.square_mm:.2f} mm, 4x4_1000 markers. Cut along the gray line.",
              fill=0, font=font, spacing=round(board.px_mm))
    page.save(path, resolution=dpi)


def sample_board(board, xy):
    """Board brightness in [0, 1] at sheet points xy (..., 2) in mm, bilinear; NaN off the sheet."""
    img = board.image.astype(np.float32) / 255
    s = board.px_mm
    mx = (xy[..., 0] * s - 0.5).astype(np.float32)
    my = (xy[..., 1] * s - 0.5).astype(np.float32)
    out = cv2.remap(img, np.nan_to_num(mx, nan=-10), np.nan_to_num(my, nan=-10), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    out[~np.isfinite(mx) | (xy[..., 0] < 0) | (xy[..., 1] < 0) | (xy[..., 0] > board.sheet_mm)
        | (xy[..., 1] > board.sheet_mm)] = np.nan
    return out


# ---------------------------------------------------------------------------------------------
# Simulated photos


BACKGROUND = (0.25, 0.45, 0.8)     # a plain colored backdrop: paper is black and white, so it separates easily


def simulate_photo(model, view, board, px_mm=15, printed_side=2, blur=1.0, noise=0.02, light=0.3,
                   shadow=0.5, background=BACKGROUND, seed=0):
    """A photo of the folded board, looking straight at one view ("front" or "back"), at px_mm
    pixels per mm. Adds blur (Gaussian sigma in pixels), noise, a light falloff across the frame
    (light: how much darker the dimmest corner is), and dark lines where one layer's edge sits on
    another (shadow: how dark). Returns (photo (H, W, 3) in [0, 1], truth), where truth holds
    sheet_mm (H, W, 2), the printed-sheet point each pixel shows (NaN off paper or on the
    unprinted side), and face (H, W), the crease-pattern face shown (-1 off paper)."""
    rng = np.random.default_rng(seed)
    lo, hi = model.folded.min(0), model.folded.max(0)
    mm = board.sheet_mm / (model.sheet.max(0) - model.sheet.min(0)).max()    # model units -> mm
    res = int(round((hi - lo).max() * 1.06 * mm * px_mm))
    views, _, _ = model.views(res)
    xy, side, face = views[view]
    slo, shi = model.sheet.min(0), model.sheet.max(0)
    t = (xy - slo) / (shi - slo)                         # 0..1 across the sheet, raw (side 1) orientation
    u = (1 - t[..., 0]) if printed_side == 2 else t[..., 0]   # side 2 is the sheet turned over
    sheet_mm = np.stack([u, t[..., 1]], -1) * board.sheet_mm
    printed = side == printed_side
    sheet_mm[~printed] = np.nan

    gray = np.where(printed, np.nan_to_num(sample_board(board, sheet_mm), nan=1.0), 1.0)
    paper = side > 0
    # Shadows along layer edges: wherever the visible face changes between neighbors.
    edge = np.zeros_like(paper)
    for axis in (0, 1):
        d = np.diff(face, axis=axis) != 0
        if axis == 0:
            edge[1:] |= d; edge[:-1] |= d
        else:
            edge[:, 1:] |= d; edge[:, :-1] |= d
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & paper
    gray = np.where(edge, gray * (1 - shadow), gray)

    rgb = np.where(paper[..., None], gray[..., None] * np.array([1.0, 0.98, 0.95]), np.array(background))
    H, W = paper.shape
    yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
    a = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(a) * xx + np.sin(a) * yy)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    rgb = rgb * (1 - light * ramp)[..., None]
    if blur:
        rgb = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), blur)
    rgb = np.clip(rgb + rng.normal(0, noise, rgb.shape), 0, 1).astype(np.float32)
    return rgb, dict(sheet_mm=sheet_mm, face=np.where(paper, face, -1))


# ---------------------------------------------------------------------------------------------
# Recovering the map from a photo


def paper_mask(photo, min_saturation=0.3, min_value=0.15):
    """Paper pixels: the board is black and white and the backdrop is colored, so the backdrop is
    whatever is clearly colored (black ink has no reliable hue, so dark pixels count as paper)."""
    hsv = cv2.cvtColor((np.clip(photo, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    backdrop = (hsv[..., 1] > min_saturation * 255) & (hsv[..., 2] > min_value * 255)
    backdrop = cv2.morphologyEx(backdrop.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return ~backdrop.astype(bool)


def detect(photo, board):
    """Markers in a photo: {id: (4, 2) corners in photo pixels}."""
    gray = (np.clip(photo.mean(-1) if photo.ndim == 3 else photo, 0, 1) * 255).astype(np.uint8)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(board.dictionary, params).detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): np.asarray(c, float).reshape(4, 2) for i, c in zip(np.ravel(ids), corners)}


def _fit(src, dst, kind):
    if kind == "homography":
        M, _ = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32), 0)
        return M
    # Rotation, scale and shift, by least squares: x' = a x - b y + tx, y' = b x + a y + ty.
    x, y = src[:, 0], src[:, 1]
    one, zero = np.ones_like(x), np.zeros_like(x)
    A = np.concatenate([np.stack([x, -y, one, zero], 1), np.stack([y, x, zero, one], 1)])
    a, b, tx, ty = np.linalg.lstsq(A, np.concatenate([dst[:, 0], dst[:, 1]]), rcond=None)[0]
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]])


def _project(M, P):
    q = np.c_[P, np.ones(len(P))] @ M.T
    return q[:, :2] / q[:, 2:]


def fit_pieces(markers, board, kind="similarity", tol=0.15):
    """Group detected markers into rigid pieces of paper. Repeatedly takes the transform (sheet mm
    -> photo pixels) of one marker that the most other markers agree with, to within tol marker
    sides, refits it to all of them, and sets them aside. kind: "similarity" (photo taken straight
    on) or "homography" (at an angle). Returns a list of (3x3 matrix, [marker ids]), largest first."""
    side_px = {i: np.linalg.norm(c[1] - c[0]) for i, c in markers.items()}
    # Skip markers whose own corners don't fit the transform (misread, or bent across a crease).
    left = {i: c for i, c in markers.items() if i in board.corners and
            np.abs(_project(_fit(board.corners[i], c, kind), board.corners[i]) - c).max() < tol * side_px[i]}
    pieces = []
    while left:
        ids = list(left)
        best = None
        for i in ids:
            M = _fit(board.corners[i], left[i], kind)
            agree = [j for j in ids
                     if np.abs(_project(M, board.corners[j]) - left[j]).max() < tol * side_px[j]]
            if best is None or len(agree) > len(best[1]):
                best = (M, agree)
        M, agree = best
        for _ in range(3):              # refit to everything that agrees, then look again
            M = _fit(np.concatenate([board.corners[j] for j in agree]), np.concatenate([left[j] for j in agree]), kind)
            agree = [j for j in ids
                     if np.abs(_project(M, board.corners[j]) - left[j]).max() < tol * side_px[j]] or agree
        pieces.append((M, agree))
        for j in agree:
            del left[j]
    return pieces


def _local_ncc(a, b, valid, win):
    # Normalized cross-correlation of a and b over a win x win window around each pixel, using only
    # pixels where valid.
    k = (win, win)
    w = cv2.boxFilter(valid.astype(np.float32), -1, k, normalize=False)
    a = np.where(valid, a, 0).astype(np.float32)
    b = np.where(valid, b, 0).astype(np.float32)
    s = lambda x: cv2.boxFilter(x, -1, k, normalize=False)
    n = np.maximum(w, 1)
    ma, mb = s(a) / n, s(b) / n
    cov = s(a * b) / n - ma * mb
    va = s(a * a) / n - ma * ma
    vb = s(b * b) / n - mb * mb
    ncc = cov / np.sqrt(np.maximum(va * vb, 1e-8))
    return np.where(w > 0.5 * win * win, ncc, -1)


def dense_map(photo, board, pieces, mask=None, window_mm=None, min_ncc=0.5):
    """For every paper pixel of the photo, the sheet point it shows: each piece's transform predicts
    the board's pattern across the photo, and each pixel goes to the piece whose prediction matches
    best nearby (normalized cross-correlation over a window about one checker square wide). Returns
    (sheet_mm (H, W, 2) with NaN where no piece matches, label (H, W) piece index or -1)."""
    gray = photo.mean(-1) if photo.ndim == 3 else photo
    H, W = gray.shape
    mask = paper_mask(photo) if mask is None else mask
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    best = np.full((H, W), -np.inf, np.float32)
    label = np.full((H, W), -1)
    sheet_mm = np.full((H, W, 2), np.nan)
    for k, (M, ids) in enumerate(pieces):
        px_per_mm = np.sqrt(abs(np.linalg.det(M[:2, :2])))
        win = int(round((window_mm or board.square_mm) * px_per_mm)) | 1
        xy = _project(np.linalg.inv(M), pts).reshape(H, W, 2)
        pred = sample_board(board, xy)
        valid = mask & np.isfinite(pred)
        pred = cv2.GaussianBlur(np.nan_to_num(pred, nan=0.5).astype(np.float32), (0, 0), 1.0)
        ncc = _local_ncc(gray, pred, valid, win)
        take = valid & (ncc > best) & (ncc > min_ncc)
        best[take] = ncc[take]
        label[take] = k
        sheet_mm[take] = xy[take]
    return sheet_mm, label


def recover(photo, board, kind="similarity"):
    """The whole pipeline: markers, pieces, and the dense map. Returns a dict with markers,
    pieces, sheet_mm and label."""
    markers = detect(photo, board)
    pieces = fit_pieces(markers, board, kind)
    sheet_mm, label = dense_map(photo, board, pieces)
    return dict(markers=markers, pieces=pieces, sheet_mm=sheet_mm, label=label)


def evaluate(result, truth, tol_mm=0.5):
    """Compare a recovered map to the true one, over the paper that shows the printed side:
    the share of it recovered to within tol_mm, wrong by over 2 tol_mm, and not recovered."""
    shown = np.isfinite(truth["sheet_mm"][..., 0])
    got = np.isfinite(result["sheet_mm"][..., 0])
    err = np.linalg.norm(result["sheet_mm"] - truth["sheet_mm"], axis=-1)
    both = shown & got
    return dict(markers=len(result["markers"]), pieces=len(result["pieces"]),
                accurate=float((both & (err < tol_mm)).sum() / shown.sum()),
                wrong=float((both & (err >= 2 * tol_mm)).sum() / shown.sum()),
                missing=float((shown & ~got).sum() / shown.sum()),
                median_error_mm=float(np.median(err[both])) if both.any() else np.nan,
                error_mm=np.where(both, err, np.nan))


# ---------------------------------------------------------------------------------------------
# Pictures


def piece_colors(n):
    import matplotlib
    return matplotlib.colormaps["tab20"](np.arange(n) % 20)[:, :3]


def on_sheet(result, board, res=600):
    """Where the recovered view comes from on the printed sheet: the board faded, with each piece's
    paper tinted in its color. This is the visibility map you'd otherwise shade by hand."""
    img = cv2.resize(board.image, (res, res), interpolation=cv2.INTER_AREA).astype(float) / 255
    out = np.repeat((0.75 + 0.25 * img)[..., None], 3, -1)
    lab = result["label"]
    xy = result["sheet_mm"][lab >= 0] / board.sheet_mm * res
    rc = np.clip(xy[:, ::-1].astype(int), 0, res - 1)
    colors = piece_colors(len(result["pieces"]))[lab[lab >= 0]]
    tint = np.zeros((res, res, 3)); count = np.zeros((res, res))
    np.add.at(tint, (rc[:, 0], rc[:, 1]), colors)
    np.add.at(count, (rc[:, 0], rc[:, 1]), 1)
    seen = count > 0
    out[seen] = 0.35 * out[seen] * img[seen][:, None] + 0.65 * tint[seen] / count[seen][:, None]
    return out


def plot_recovery(photo, result, board, truth=None):
    """The photo with detected markers, the pieces found, where they come from on the sheet, and
    (with the truth from simulate_photo) the error."""
    import matplotlib.pyplot as plt
    n = 4 if truth is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5))
    axes[0].imshow(np.clip(photo, 0, 1))
    for c in result["markers"].values():
        axes[0].add_patch(plt.Polygon(c, fill=False, ec="#f33", lw=0.8))
    axes[0].set_title(f"photo: {len(result['markers'])} markers found")
    lab = result["label"]
    colors = piece_colors(len(result["pieces"]))
    pieces = np.ones(lab.shape + (3,))
    pieces[lab >= 0] = colors[lab[lab >= 0]]
    axes[1].imshow(pieces)
    axes[1].set_title(f"{len(result['pieces'])} rigid pieces")
    axes[2].imshow(on_sheet(result, board))
    axes[2].set_title("where they are on the printed sheet")
    if truth is not None:
        ev = evaluate(result, truth)
        im = axes[3].imshow(ev["error_mm"], vmin=0, vmax=2, cmap="magma")
        axes[3].imshow(np.where(np.isfinite(truth["sheet_mm"][..., 0]) & ~np.isfinite(result["sheet_mm"][..., 0]),
                                1.0, np.nan), cmap="cool", vmin=0, vmax=1)
        axes[3].set_title(f"error (mm), missing in cyan\n{ev['accurate']:.1%} within 0.5 mm, "
                          f"{ev['wrong']:.1%} off by over 1 mm")
        fig.colorbar(im, ax=axes[3], fraction=0.035)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
