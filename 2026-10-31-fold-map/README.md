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

## Next

- Splice an image onto the visible regions using `fold_map.npz`.
- Measure the fold map from photos of a folded sheet printed with a coded pattern, and compare it to the computed map.
