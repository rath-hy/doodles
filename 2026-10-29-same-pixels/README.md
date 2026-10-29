# Same pixels, new picture

**Started:** 2026-10-29

## What sparked it

Learning about color histograms in CS180 made me wonder: could an image generator take a photo and a text prompt, and produce a new picture whose color histogram is *identical* to the photo's? Because the output is the same size, an identical histogram means it's a rearrangement of the photo's pixels.

## How it works

[`same_pixels.ipynb`](same_pixels.ipynb) runs Stable Diffusion 3.5 Medium, a rectified-flow model, with a custom sampling loop:

1. Partway through sampling, take the model's prediction of the final image and pull its colors onto the source's color distribution using sliced optimal transport.
2. Put the corrected image back on the same noise trajectory and keep sampling.
3. At the end, run an exact snap that assigns every output pixel a distinct source pixel. Pixels whose exact color is available keep it, and the rest are paired along a 3D Hilbert curve through color space.

## Running it

Open the notebook in Google Colab (File → Upload notebook) with a GPU runtime. You'll need to accept the SD3.5 Medium license on Hugging Face.

## Status

- **Model loop:** not yet run on a GPU.
- **`sliced_ot` projection and exact snap:** tested locally on CPU with synthetic 512×512 images. The histogram comes out identical. Projecting first shrinks the snap's average color change from 77/255 to 6.5/255.

It needs a GPU, so it can't run as a static page on puthyrathy.com/doodles. A gallery of results could go there later.
