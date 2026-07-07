import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

JPEGS_DIR = Path(__file__).resolve().parent / "data" / "jpegs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BEAM_LENGTH_CM = 10

FILENAME_PATTERN = re.compile(r"^(R_6G|R_B|F)_(\d+)\.jpg$")

DYE_NAMES = {
    "F": "Fluorescein",
    "R_B": "Rhodamine B",
    "R_6G": "Rhodamine 6G",
}

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


def linear(x, a, b):
    return a * x + b


def process_image(path):
    parsed = parse_filename(path.name)
    if parsed is None:
        return
    dye_key, conc = parsed
    ch = DYE_CHANNEL[dye_key]
    stem = path.stem
    title_prefix = f"{DYE_NAMES[dye_key]} {conc} mM"

    img = plt.imread(path).astype(np.float64) / 255.0
    channel = img[:, :, ch]

    beam_row = int(np.argmax(channel.mean(axis=1)))
    half_band = 20
    band_slice = slice(max(0, beam_row - half_band),
                       min(channel.shape[0], beam_row + half_band))
    band = channel[band_slice, :]

    profile = band.mean(axis=0)
    profile_std = band.std(axis=0)

    x = np.linspace(0, BEAM_LENGTH_CM, len(profile))

    profile_clipped = np.clip(profile, a_min=1e-10, a_max=None)
    log_profile = np.log(profile_clipped)
    # Minimum error: 1 count in 8-bit image (1/255)
    profile_std = np.clip(profile_std, a_min=1.0 / 255, a_max=None)
    log_err = profile_std / profile_clipped

    out_dir = RESULTS_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap of the selected channel
    fig, ax = plt.subplots()
    ax.imshow(channel, cmap="hot", aspect="auto")
    ax.axhline(beam_row, color="cyan", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")
    ax.set_title(f"{title_prefix} — {'Green' if ch == 1 else 'Red'} Channel")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap.png", dpi=150)
    plt.close(fig)

    # Linear fit to ln(Intensity) vs position
    valid = (log_err > 0) & np.isfinite(log_err) & np.isfinite(log_profile)
    x_fit = x[valid]
    y_fit = log_profile[valid]
    yerr_fit = log_err[valid]

    popt, pcov = curve_fit(linear, x_fit, y_fit, sigma=yerr_fit,
                           absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))
    slope, intercept = popt
    slope_err, intercept_err = perr

    residuals = y_fit - linear(x_fit, *popt)
    chi2 = np.sum((residuals / yerr_fit) ** 2)
    ndf = len(x_fit) - 2
    chi2_red = chi2 / ndf

    # Combined fit + residuals plot
    fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(8, 6),
                                         gridspec_kw={"height_ratios": [3, 1]},
                                         sharex=True)
    ax_fit.errorbar(x_fit, y_fit, yerr=yerr_fit, fmt=",", color="#5599cc",
                    alpha=0.8, linewidth=0.3, label="Data", zorder=1)
    ax_fit.plot(x_fit, linear(x_fit, *popt), "r-", linewidth=1.5,
                label=f"Fit: slope = {slope:.4f} ± {slope_err:.4f}", zorder=2)
    ax_fit.set_ylabel("ln(Intensity)")
    ax_fit.set_title(f"{title_prefix} — Linear Fit")
    ax_fit.legend()
    ax_fit.grid(True, which="both")

    ax_res.errorbar(x_fit, residuals, yerr=yerr_fit, fmt=",", color="#5599cc",
                    alpha=0.8, linewidth=0.3)
    ax_res.axhline(0, color="r", linewidth=1)
    ax_res.set_xlabel("x [cm]")
    ax_res.set_ylabel("Residuals")
    ax_res.grid(True, which="both")

    fig.tight_layout()
    fig.savefig(out_dir / "fit.png", dpi=150)
    plt.close(fig)

    # Fit results text file
    with open(out_dir / "fit.txt", "w") as f:
        f.write(f"Sample: {title_prefix}\n")
        f.write(f"Slope:     {slope:.6f} +/- {slope_err:.6f} cm^-1\n")
        f.write(f"Intercept: {intercept:.6f} +/- {intercept_err:.6f}\n")
        f.write(f"Chi2:      {chi2:.2f}\n")
        f.write(f"NDF:       {ndf}\n")
        f.write(f"Chi2/NDF:  {chi2_red:.4f}\n")

    print(f"{path.name}: slope={slope:.4f}±{slope_err:.4f}, "
          f"chi2/ndf={chi2_red:.2f}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    for path in sorted(JPEGS_DIR.glob("*.jpg")):
        process_image(path)


if __name__ == "__main__":
    main()
