# Fold map

**Started:** 2026-10-31

## What sparked it

A possible CS180 project: printing images on paper so they show up on a folded origami model, or hiding images in places only visible when it's unfolded. Before any of that, I need to know which parts of the sheet are visible once it's folded, and where each visible point ends up.

## How it works

[`fold_map.py`](fold_map.py) takes a crease pattern and a folded state from [Flat-Folder](https://github.com/origamimagiro/flat-folder) (Jason Ku's solver for flat-folded states):

```
python fold_map.py 004_traditional_Crane_cp.fold 004_traditional_Crane_state.fold
```

1. **Positions:** each face of the crease pattern is a rigid piece. Comparing its corners on the sheet and in the folded state gives the map between the two, and tells you whether the piece was flipped over.
2. **Stacking:** the folded state's `faceOrders` say which of two overlapping faces is on top. In the FOLD spec, `[f, g, s]` is relative to **g's** normal, so the sign flips for faces that were turned over.
3. **Visibility:** a point of the sheet is visible from the front if its face is the bottommost layer where it lands (Flat-Folder's view looks from −z) and from the back if it's the topmost.

Outputs:

- `visibility_sheet.png`: both sides of the sheet, colored hidden / visible from the front / visible from the back.
- `folded_views.png`: the folded model from both sides, textured by where each visible point sits on the sheet.
- `fold_map.npz`: for each pixel of each folded view, the sheet point it shows and which side, for splicing images later.

Conventions follow Flat-Folder's display: side 1 is the side you see in its crease pattern view, the front is its folded view, and the back is its folded view with Flip checked.

## Crane results

The traditional crane (Flat-Folder example `004`; 72 faces and 5 valid folded states; this uses the state that was exported):

- **Side 1 is entirely hidden.** Fold from one-sided paper with the color on side 2 and no white shows, which matches real cranes.
- **Side 2:** 11.8% of it is visible from the front, 11.8% from the back, and 76.4% is hidden.
- **Visible areas don't follow the creases.** Some boundaries cut across faces, wherever another layer's edge covers part of a face.

## Checking it against a real crane

I folded a crane from notebook paper following [Rachel Katz's diagrams](https://origamiwithrachelkatz.oriland.com/folding/crane.php), stopping before opening the wings so it stayed flat. I shaded everything visible on one face with straight pencil lines and on the other with circles, then unfolded it. The shading matches the predicted regions in `visibility_sheet.png` (side 2). Photos are in [`crane_pics/`](crane_pics/), with location metadata removed.

## Which way is up

[`directions.py`](directions.py) draws a letter F on each visible region of side 2, oriented the way a picture must be printed there to appear upright on the folded crane (`directions.png`). For the crane, all 18 visible regions only need rotating (by multiples of 22.5°), never mirroring, since you see the same side of the paper you print on.

## Decal: print a picture that appears on the folded crane

```
python decal.py                                   # test pictures: a big F and B
python decal.py --front cat.jpg --back dog.jpg    # your own pictures
```

Each picture is center-cropped to the folded silhouette (about 1.3:1, landscape, for the crane) and mapped onto the sheet point by point: every visible point gets the color of the picture at the place it lands when folded. Only visible regions are printed, plus a 1% bleed (`--bleed`) so imperfect folds don't show white slivers. Tiny letters A–D in hidden areas mark the corners, and the preview shows where each lands (crane: A tail, C head, B and D wings). Writes `decal_side2.png` (8 × 8 inches at 300 dpi; also `decal_side1.png` for models that show both sides) and `decal_preview.png` (the model simulated from the printed sheets).

## Notebook

[`fold_map.ipynb`](fold_map.ipynb) walks through the whole workflow for any flat-foldable model from Flat-Folder: where each face lands, the layer stacking, the visibility map, which way is up, and the decal, with uploads for your own model and pictures on Colab. It imports the code from this folder (cloning the repo on Colab), and ends with a list of building blocks for further experiments.

## Illusions: one picture flat, another folded

[`illusions.ipynb`](illusions.ipynb) and [`illusion.py`](illusion.py) make a sheet that shows one picture when flat and other pictures on the folded crane's front and back. `FoldViews` turns the fold map into pixel maps between the printed side and the folded views (98–99% one-to-one at 256 × 256, since folding is rigid). On top of that:

- **Composite (CPU):** the flat picture in hidden areas and the folded pictures in visible ones. The baseline; the flat picture ends up shattered.
- **Generated (GPU):** Visual Anagrams (Geng, Park, Owens, CVPR 2024) adapted to folding, using DeepFloyd IF. At every denoising step the model sees the sheet flat and folded, each with its own prompt, and the predictions are mapped back and blended per pixel. The folded views are drawn on a plain backdrop held at the current noise level.

The geometry and the sampler's bookkeeping were tested on CPU with a stand-in model, and the notebook runs on a free Colab GPU.

### Trial 1

A city street at night (oil painting) flat, a red fox on the front and a white owl on the back ([`prints/`](prints/): the generated sheet, the A4 print file with two copies, and the virtual fold). Folded on the computer, both animals appear. Printed on a home printer and folded by hand ([`crane_pics/trial_1/`](crane_pics/trial_1/), including a timelapse), the fox comes through but the owl is barely visible: the print is darker and muddier than the screen, white crease lines cut across the picture, and the folds aren't exact. Bolder, simpler, higher-contrast pictures should survive better.

## Folding story: a picture at every stage

[`story.ipynb`](story.ipynb) and [`story.py`](story.py) make one sheet that shows a different picture flat, as a preliminary base, as a bird base, and as the finished crane (for example a field of leaves, a bud, a half-open flower, a flower in full bloom).

- **The stages.** [`make_stages.py`](make_stages.py) writes crease patterns for the two bases on the crane's sheet, with creases that match the crane's, and solves them with Flat-Folder's own solver, run outside the browser by [`ff_solve.mjs`](ff_solve.mjs). That script reproduces the crane state exported from the website exactly. The results are in `stages/`. Both bases leave the BD diagonal unfolded; folding it doubles each base over.
- **Digits (CPU).** `python story.py` places 0, 1, 2, 3 so they compete for as few pixels as possible. It shows the core difficulty: each face of the preliminary base is an uncut quarter of the sheet, so its picture is readable flat. The bird base shows 2 pieces of paper, the crane about 10.
- **Generated (GPU).** The DeepFloyd sampler in `illusion.py` now takes any list of folded views, from any models. The notebook runs it with the flat sheet plus the three stages, each with its own prompt, so later pictures can hide in the texture of earlier ones.
- **First GPU run.** With the digits 1, 2, 3 (pencil drawings on a "geometric mess"), the 1 vanished into the flat scene, and the 2 and 3 blurred into each other. The reason: consecutive stages show a lot of the same paper, because folding keeps much of the outside on the outside. With the sides used, 53% of the crane's visible paper also showed on the bird base. The notebook now picks sides by how much of each stage's own paper is shared (`least_shared_sides`; for the crane, the finished crane shares none and the two bases share 41% and 50%), and the default story is a transformation whose stages can share strokes: a bud opening into a flower.

## Next

- Splice an image onto the visible regions using `fold_map.npz`.
- Measure the fold map from photos of a folded sheet printed with a coded pattern, and compare it to the computed map.
