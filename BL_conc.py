import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

DATA_DIR = Path(__file__).resolve().parent / "data" / "csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "partA"
DATA_REGION = (470, 700)  # nm — cut off incident light (450 nm) and very noisy tail (>700 nm)
NOISE_REGION = (650, 700)  # nm — tail where signal is ~0

DYE_INFO = {
    "F": "Fluorescein",
    "R_B": "Rhodamine B",
    "R_6G": "Rhodamine 6G",
}

FILENAME_PATTERN = re.compile(
    r"^(R_6G|R_B|F)_(\d+)\.csv$"
)


def parse_filename(name: str):
    m = FILENAME_PATTERN.match(name)
    if not m:
        return None
    dye_key = m.group(1)
    conc_str = m.group(2)
    conc = float(conc_str[0] + "." + conc_str[1:])
    return dye_key, conc


def load_and_process(path: Path):
    df = pd.read_csv(path)
    wavelength = df.iloc[:, 0].values
    intensity = df.iloc[:, 1].values

    mask = (wavelength >= DATA_REGION[0]) & (wavelength <= DATA_REGION[1])
    wavelength = wavelength[mask]
    intensity = intensity[mask]

    integral = trapezoid(intensity, wavelength)

    noise_mask = (wavelength >= NOISE_REGION[0]) & (wavelength <= NOISE_REGION[1])
    sigma_noise = intensity[noise_mask].std()
    dlam = np.median(np.diff(wavelength))
    n_pts = len(wavelength)
    sigma_integral = sigma_noise * dlam * np.sqrt(n_pts)

    return wavelength, intensity, integral, sigma_integral


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dye_data = {key: [] for key in DYE_INFO}

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        parsed = parse_filename(csv_path.name)
        if parsed is None:
            continue
        dye_key, conc = parsed
        wavelength, intensity, integral, sigma = load_and_process(csv_path)
        dye_data[dye_key].append((conc, wavelength, intensity, integral, sigma))

    for dye_key, entries in dye_data.items():
        entries.sort(key=lambda e: e[0])
        dye_name = DYE_INFO[dye_key]

        # --- Plot 1: intensity spectra ---
        fig, ax = plt.subplots()
        for conc, wl, intensity, _, _ in entries:
            ax.plot(wl, intensity, label=f"{conc} mM")
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity [AU]")
        ax.set_title(f"{dye_name} — Emission Spectra")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{dye_key}_spectra.png", dpi=150)
        plt.close(fig)

        # --- Plot 2: integral vs concentration (per dye) ---
        fig, ax = plt.subplots()
        concentrations = [e[0] for e in entries]
        integrals = [e[3] for e in entries]
        sigmas = [e[4] for e in entries]
        ax.errorbar(concentrations, integrals, yerr=sigmas, fmt="o-", capsize=3, markersize=2, linewidth=0.5)
        ax.set_xlabel("Concentration [mM]")
        ax.set_ylabel("Integrated Intensity")
        ax.set_title(f"{dye_name} — Integral vs Concentration")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{dye_key}_integral.png", dpi=150)
        plt.close(fig)

    # --- Combined integral plot: all 3 dyes ---
    fig, ax = plt.subplots()
    for dye_key, entries in dye_data.items():
        entries.sort(key=lambda e: e[0])
        concentrations = [e[0] for e in entries]
        integrals = [e[3] for e in entries]
        sigmas = [e[4] for e in entries]
        ax.errorbar(concentrations, integrals, yerr=sigmas, fmt="o-", markersize=2, linewidth=0.5,
                    capsize=3, label=DYE_INFO[dye_key])
    ax.set_xlabel("Concentration [mM]")
    ax.set_ylabel("Integrated Intensity")
    ax.set_title("Integral vs Concentration — All Dyes")
    ax.legend(loc="upper right")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "all_integrals.png", dpi=150)
    plt.close(fig)

    # --- Linear fits on 3 lowest concentrations ---
    def linear(x, a, b):
        return a * x + b

    combined_fit_fig, combined_fit_ax = plt.subplots()
    fit_results = []

    for dye_key, entries in dye_data.items():
        entries.sort(key=lambda e: e[0])
        dye_name = DYE_INFO[dye_key]

        low3 = entries[:3]
        c = np.array([e[0] for e in low3])
        s = np.array([e[3] for e in low3])
        s_err = np.array([e[4] for e in low3])

        popt, pcov = curve_fit(linear, c, s, sigma=s_err, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        slope, intercept = popt
        slope_err, intercept_err = perr

        residuals = s - linear(c, *popt)
        chi2 = np.sum((residuals / s_err) ** 2)
        ndf = len(c) - 2
        chi2_red = chi2 / ndf if ndf > 0 else float("nan")

        fit_results.append({
            "dye": dye_name,
            "slope": slope, "slope_err": slope_err,
            "intercept": intercept, "intercept_err": intercept_err,
            "chi2": chi2, "ndf": ndf, "chi2_red": chi2_red,
            "concentrations": c, "integrals": s, "errors": s_err,
        })

        # Per-dye fit + residuals
        fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(8, 6),
                                             gridspec_kw={"height_ratios": [3, 1]},
                                             sharex=True)
        ax_fit.errorbar(c, s, yerr=s_err, fmt="o", markersize=2, capsize=3)
        c_line = np.linspace(0, c[-1] * 1.1, 100)
        ax_fit.plot(c_line, linear(c_line, *popt), "r-")
        ax_fit.set_ylabel("Integrated Intensity")
        ax_fit.set_title(f"{dye_name} — S(c) Linear Fit")
        ax_fit.grid(True)

        ax_res.errorbar(c, residuals, yerr=s_err, fmt="o", markersize=2, capsize=3)
        ax_res.axhline(0, color="r", linewidth=1)
        ax_res.set_xlabel("Concentration [mM]")
        ax_res.set_ylabel("Residuals")
        ax_res.grid(True)

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{dye_key}_Sc_fit.png", dpi=150)
        plt.close(fig)

        # Add to combined plot
        color = f"C{list(DYE_INFO).index(dye_key)}"
        combined_fit_ax.errorbar(c, s, yerr=s_err, fmt="o", markersize=2, linewidth=0.5,
                                capsize=3, color=color)
        combined_fit_ax.plot(c_line, linear(c_line, *popt), "-", color=color, linewidth=0.5,
                            label=f"{dye_name}")

    combined_fit_ax.set_xlabel("Concentration [mM]")
    combined_fit_ax.set_ylabel("Integrated Intensity")
    combined_fit_ax.set_title("S(c) Linear Fits — All Dyes")
    combined_fit_ax.legend()
    combined_fit_ax.grid(True)
    combined_fit_fig.tight_layout()
    combined_fit_fig.savefig(RESULTS_DIR / "all_Sc_fits.png", dpi=150)
    plt.close(combined_fit_fig)

    # --- Write fit results to text file ---
    with open(RESULTS_DIR / "Sc_fit_results.txt", "w") as f:
        for r in fit_results:
            f.write(f"=== {r['dye']} ===\n")
            f.write(f"Slope:     {r['slope']:.4f} +/- {r['slope_err']:.4f}\n")
            f.write(f"Intercept: {r['intercept']:.4f} +/- {r['intercept_err']:.4f}\n")
            f.write(f"Chi2:      {r['chi2']:.2f}\n")
            f.write(f"NDF:       {r['ndf']}\n")
            f.write(f"Chi2/NDF:  {r['chi2_red']:.4f}\n\n")


if __name__ == "__main__":
    main()
