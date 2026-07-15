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

# ============================================================
# Store attenuation coefficients for summary plot
# ============================================================

summary = {
    "Fluorescein": {"c": [], "alpha": [], "err": []},
    "Rhodamine B": {"c": [], "alpha": [], "err": []},
    "Rhodamine 6G": {"c": [], "alpha": [], "err": []},
}
# Maximum concentration included in the alpha(c) linear fit.
# Fluorescein at 0.10 mM is excluded because it clearly departs
# from the approximately linear Beer--Lambert regime.
ALPHA_FIT_MAX_CONC_MILLIMOLAR = {
    "Fluorescein": 0.050,
    "Rhodamine B": 0.100,
    "Rhodamine 6G": 0.100,
}
def parse_filename(name):
    m = FILENAME_PATTERN.match(name)
    if not m:
        return None
    dye_key = m.group(1)
    conc_str = m.group(2)
    conc = float(conc_str[0] + "." + conc_str[1:])
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
                label=f"Fit", zorder=2)
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
    # --------------------------------------------------------
    # Save fit results for summary plot
    # --------------------------------------------------------

    summary[DYE_NAMES[dye_key]]["c"].append(conc)
    summary[DYE_NAMES[dye_key]]["alpha"].append(-slope)
    summary[DYE_NAMES[dye_key]]["err"].append(slope_err)
    print(f"{path.name}: slope={slope:.4f}±{slope_err:.4f}, "
          f"chi2/ndf={chi2_red:.2f}")

def fit_alpha_vs_concentration():
    """
    Fit alpha(c) = m*c + b for each dye.

    Concentration c is supplied in mM, so:
        m has units cm^-1 mM^-1

    Since 1 mM = 10^-3 mol/L, the conventional molar attenuation
    coefficient is:
        epsilon = 1000*m  [L mol^-1 cm^-1]
    """

    fit_results = {}

    fig, ax = plt.subplots(figsize=(7, 5))

    for dye, values in summary.items():
        c = np.asarray(values["c"], dtype=float)
        alpha = np.asarray(values["alpha"], dtype=float)
        alpha_err = np.asarray(values["err"], dtype=float)

        # Sort measurements by concentration
        order = np.argsort(c)
        c = c[order]
        alpha = alpha[order]
        alpha_err = alpha_err[order]

        # Retain only the selected approximately linear regime
        max_conc = ALPHA_FIT_MAX_CONC_MILLIMOLAR[dye]
        fit_mask = c <= max_conc

        c_fit = c[fit_mask]
        alpha_fit = alpha[fit_mask]
        err_fit = alpha_err[fit_mask]

        if len(c_fit) < 3:
            raise ValueError(
                f"Not enough concentration points to fit {dye}: "
                f"{len(c_fit)} points available."
            )

        # Weighted linear fit: alpha = slope*c + intercept
        popt, pcov = curve_fit(
            linear,
            c_fit,
            alpha_fit,
            sigma=err_fit,
            absolute_sigma=True,
        )

        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))

        fitted_alpha = linear(c_fit, *popt)
        residuals = alpha_fit - fitted_alpha

        chi2 = np.sum((residuals / err_fit) ** 2)
        ndf = len(c_fit) - 2
        chi2_red = chi2 / ndf if ndf > 0 else np.nan

        # Convert from cm^-1 mM^-1 to L mol^-1 cm^-1
        epsilon = 1000.0 * slope
        epsilon_err_formal = 1000.0 * slope_err

        # The image-derived errors are very small and the reduced chi-square
        # may be much larger than one. This scaled error reflects the observed
        # scatter more realistically.
        uncertainty_scale = np.sqrt(chi2_red) if chi2_red > 1 else 1.0
        epsilon_err_scaled = epsilon_err_formal * uncertainty_scale

        fit_results[dye] = {
            "slope_cm-1_mM-1": slope,
            "slope_err_cm-1_mM-1": slope_err,
            "intercept_cm-1": intercept,
            "intercept_err_cm-1": intercept_err,
            "epsilon_L_mol-1_cm-1": epsilon,
            "epsilon_err_formal": epsilon_err_formal,
            "epsilon_err_scaled": epsilon_err_scaled,
            "chi2": chi2,
            "ndf": ndf,
            "chi2_red": chi2_red,
            "max_concentration_mM": max_conc,
        }

        # Plot all measured points
        ax.errorbar(
            c,
            alpha,
            yerr=alpha_err,
            fmt="o",
            capsize=3,
            markersize=5,
            label=f"{dye} data",
        )

        # Plot the fit only across the fitted concentration interval
        c_line = np.linspace(0.0, max_conc, 250)
        ax.plot(
            c_line,
            linear(c_line, *popt),
            linewidth=1.5,
            label=(
                rf"{dye} fit: "
                rf"$\varepsilon_{{\rm eff}}={epsilon:.0f}$ "
                rf"L mol$^{{-1}}$ cm$^{{-1}}$"
            ),
        )

    ax.set_xlabel("Concentration [mM]")
    ax.set_ylabel(r"Attenuation coefficient $\alpha$ [cm$^{-1}$]")
    ax.set_title("Linear Fits of Optical Attenuation vs Concentration")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "alpha_linear_fits.png", dpi=200)
    plt.close(fig)

    # Save detailed numerical fit results
    output_path = RESULTS_DIR / "molar_attenuation_fits.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "Fit model: alpha(c) = slope*c + intercept\n"
            "Concentration is expressed in mM.\n"
            "epsilon_eff = 1000*slope in L mol^-1 cm^-1.\n\n"
        )

        for dye, result in fit_results.items():
            f.write(f"=== {dye} ===\n")
            f.write(
                "Fitted concentration range: "
                f"c <= {result['max_concentration_mM']:.3f} mM\n"
            )
            f.write(
                "Slope:     "
                f"{result['slope_cm-1_mM-1']:.6f} +/- "
                f"{result['slope_err_cm-1_mM-1']:.6f} "
                "cm^-1 mM^-1\n"
            )
            f.write(
                "Intercept: "
                f"{result['intercept_cm-1']:.6f} +/- "
                f"{result['intercept_err_cm-1']:.6f} cm^-1\n"
            )
            f.write(
                "epsilon_eff: "
                f"{result['epsilon_L_mol-1_cm-1']:.2f} +/- "
                f"{result['epsilon_err_formal']:.2f} "
                "L mol^-1 cm^-1 (formal fit uncertainty)\n"
            )
            f.write(
                "epsilon_eff: "
                f"{result['epsilon_L_mol-1_cm-1']:.2f} +/- "
                f"{result['epsilon_err_scaled']:.2f} "
                "L mol^-1 cm^-1 "
                "(uncertainty scaled by sqrt(chi2/NDF))\n"
            )
            f.write(f"Chi2:      {result['chi2']:.3f}\n")
            f.write(f"NDF:       {result['ndf']}\n")
            f.write(f"Chi2/NDF:  {result['chi2_red']:.3f}\n\n")

    return fit_results
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    for path in sorted(JPEGS_DIR.glob("*.jpg")):
        process_image(path)

    # ============================================================
    # Summary plot: alpha vs concentration
    # ============================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    for dye in summary:
        c = np.array(summary[dye]["c"])
        alpha = np.array(summary[dye]["alpha"])
        err = np.array(summary[dye]["err"])

        order = np.argsort(c)

        ax.errorbar(
            c[order],
            alpha[order],
            yerr=err[order],
            marker="o",
            markersize=2,
            capsize=3,
            linewidth=0.5,
            label=dye,
        )

    ax.set_xlabel("Concentration [mM]")
    ax.set_ylabel(r"Attenuation coefficient $\alpha$ [cm$^{-1}$]")
    ax.set_title("Optical Attenuation vs Concentration")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "all_alpha.png", dpi=200)
    plt.close(fig)

    # ============================================================
    # Save attenuation summary table
    # ============================================================

    with open(
        RESULTS_DIR / "attenuation_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write("Dye\tConcentration(mM)\talpha(cm^-1)\terror\n")

        for dye in summary:
            for c, a, e in zip(
                summary[dye]["c"],
                summary[dye]["alpha"],
                summary[dye]["err"],
            ):
                f.write(
                    f"{dye}\t{c:.4f}\t{a:.5f}\t{e:.5f}\n"
                )

    # ============================================================
    # Fit alpha(c) and extract epsilon_eff
    # ============================================================

    molar_fit_results = fit_alpha_vs_concentration()

    print("\nEffective molar attenuation coefficients:")

    for dye, result in molar_fit_results.items():
        print(
            f"{dye}: epsilon_eff = "
            f"{result['epsilon_L_mol-1_cm-1']:.2f} +/- "
            f"{result['epsilon_err_scaled']:.2f} "
            "L mol^-1 cm^-1"
        )


if __name__ == "__main__":
    main()