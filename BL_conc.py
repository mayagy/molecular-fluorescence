import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

DATA_DIR = Path(__file__).resolve().parent / "data" / "csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
WAVELENGTH_CUT = 470  # nm

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
    conc = float("0." + m.group(2))  # e.g. "0005" -> 0.0005 mM
    return dye_key, conc


def load_and_process(path: Path):
    df = pd.read_csv(path)
    wavelength = df.iloc[:, 0].values
    intensity = df.iloc[:, 1].values

    mask = wavelength >= WAVELENGTH_CUT
    wavelength = wavelength[mask]
    intensity = intensity[mask]

    log_intensity = np.log10(np.clip(intensity, a_min=1e-10, a_max=None))
    integral = trapezoid(log_intensity, wavelength)

    return wavelength, intensity, integral


def main():
    dye_data: dict[str, list[tuple[float, np.ndarray, np.ndarray, float]]] = {
        key: [] for key in DYE_INFO
    }

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        parsed = parse_filename(csv_path.name)
        if parsed is None:
            continue
        dye_key, conc = parsed
        wavelength, intensity, integral = load_and_process(csv_path)
        dye_data[dye_key].append((conc, wavelength, intensity, integral))

    for dye_key, entries in dye_data.items():
        entries.sort(key=lambda e: e[0])
        dye_name = DYE_INFO[dye_key]
        concentrations = [e[0] for e in entries]
        integrals = [e[3] for e in entries]

        # --- Plot 1: intensity spectra ---
        fig, ax = plt.subplots()
        for conc, wl, intensity, _ in entries:
            ax.plot(wl, intensity, label=f"{conc} mM")
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity [AU]")
        ax.set_title(f"{dye_name} — Emission Spectra")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR.parent / f"{dye_key}_spectra.png", dpi=150)

        # --- Plot 2: integral vs concentration ---
        fig, ax = plt.subplots()
        ax.plot(concentrations, integrals, "o-")
        ax.set_xlabel("Concentration [mM]")
        ax.set_ylabel("Integrated log₁₀(Intensity)")
        ax.set_title(f"{dye_name} — Integral vs Concentration")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR.parent / f"{dye_key}_integral.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
