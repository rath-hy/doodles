# Schrödinger's box

**Started:** 2026-11-05

## What sparked it

While working on origami illusions in the fold map project, I wondered if two textures could overlap to form a picture when light shines through them. That turned into an origami box, printed only on the inside with noise. Squeeze it flat one way and hold it to a light, and the walls show a dead cat. Squeeze it the other way, and they show a live one. You can't do both at once. 

## How it works

[`schrodingers_box.ipynb`](schrodingers_box.ipynb) uses visual cryptography (Naor and Shamir, 1994). Each picture pixel is a 2 x 2 block of printed cells, half clear, in a random arrangement. One wall of a pair gets the same arrangement as the other where the picture is bright, and the opposite where it's dark. Light passes only where both are clear, so the picture glows on black, and each wall alone is the same noise whatever the picture is.

The model is the [hinged square box](https://origamiok.com/origami-hinged-square-box/), whose side is `sqrt(2) / 6` of the sheet's. On a 200 mm square, the most A4 allows, that's about 47 mm. So each wall gets 32 x 32 picture pixels, or 64 x 64 cells of 0.74 mm. A slip of a fraction `f` of a cell cuts contrast to about `1 - 7f/6`, so slip matters a lot at this size. The floor and lid carry ordinary pictures, because closing the lid leaves too big a gap.

## Printing

[`prints/test_sheet_A.pdf`](prints/test_sheet_A.pdf) and [`prints/test_sheet_B.pdf`](prints/test_sheet_B.pdf) are A4 test sheets with both cats at 1.4, 1.0 and 0.74 mm cells. Print both at 100%, lay A on B with both printed sides up, line up the crosses, and hold them to a light. [`prints/faces/`](prints/faces/) has the four walls and the floor at actual size, to place on the folded sheet.

So far this is a simulation. The sheets haven't been printed, and the box hasn't been folded.
