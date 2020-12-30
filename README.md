# fourier circles
 Circles that spin to draw things

 These python scripts allow you to create an animation that shows spinning circles tracing out a path.

## Usage

1. Clone the repository

    `git clone https://github.com/pynathanthomas/fourier_circles.git`

2. Install dependencies
    `pip install -r ~/fourier_circles/requirements.txt`

3. Find or create an image that has clear contours

4. Run `contour_from_image.py`, replacing the appropriate variables.

5. Run `main.py`, replacing the appropriate variables.

This will create multiple videos, based on the `MEMORY_THRESHOLD` constant in `main.py`. To combine them, run `concat_vid.py`.

