from pathlib import Path

from commodity_factors.data import (load_raw_data, 
                                    prepare_panel, 
                                    validate_panel)


def main():
    raw_path = Path("data/raw/wti_2021_2025.parquet")
    output_path = Path("data/processed/wti_panel_2021_2025.parquet")

    raw = load_raw_data(raw_path)
    panel = prepare_panel(raw, n=6)
    summary = validate_panel(panel, n=6)

    print("Panel summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()