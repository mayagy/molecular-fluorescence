import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

JPEGS_DIR = Path(__file__).resolve().parent / "data" / "jpegs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BEAM_LENGTH_CM = 10

FILENAME_PATTERN = re.compile(r"^(R_6G|R_B|F)_(\d+)\.jpg$")

DYE_CHANNEL = {
    "F": 1,      # green
    "R_B": 0,    # red
    "R_6G": 1,   # green
}


def parse_filename(name):
    m = FILENAME_PATTERN.match(name)
    if not m:
        return None
    dye_key = m.group(1)
    conc = float("0." + m.group(2))
    return dye_key, conc


def process_image(path):
    parsed = parse_filename(path.name)
    if parsed is None:
        return
    dye_key, conc = parsed
    ch = DYE_CHANNEL[dye_key]
    stem = path.stem

    img = plt.imread(path).astype(np.float64) / 255.0
    channel = img[:, :, ch]

    beam_row = int(np.argmax(channel.mean(axis=1)))
    half_band = 20
    band = slice(max(0, beam_row - half_band),
                 min(channel.shape[0], beam_row + half_band))
    profile = channel[band, :].mean(axis=0)
    x = np.linspace(0, BEAM_LENGTH_CM, len(profile))
    log_profile = np.log(np.clip(profile, a_min=1e-10, a_max=None))

    # Heatmap of the selected channel
    fig, ax = plt.subplots()
    ax.imshow(channel, cmap="hot", aspect="auto")
    ax.axhline(beam_row, color="cyan", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")
    ax.set_title(f"{stem} — {'Green' if ch == 1 else 'Red'} Channel")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{stem}_heatmap.png", dpi=150)
    plt.close(fig)

    # ln(Intensity) vs position
    fig, ax = plt.subplots()
    ax.plot(x, log_profile)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("ln(Intensity)")
    ax.set_title(f"{stem} — ln(Intensity) vs Position")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{stem}_log.png", dpi=150)
    plt.close(fig)

    print(f"{path.name}: ch={'G' if ch == 1 else 'R'}, beam_row={beam_row}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    for path in sorted(JPEGS_DIR.glob("*.jpg")):
        process_image(path)


if __name__ == "__main__":
    main()
