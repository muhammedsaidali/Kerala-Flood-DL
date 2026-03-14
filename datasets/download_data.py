"""
download_data.py — Kerala Flood Dataset Downloader
Downloads and prepares:
  1. FloodNet dataset (UAV imagery, flood/no-flood)
  2. Sentinel-1 SAR tiles via Copernicus Open Access Hub
  3. ISRO Bhuvan WMS raster tiles for Kerala
  4. IMD historical rainfall CSVs
  5. Synthetic fallback (no API keys needed)
"""

import os
import io
import sys
import time
import json
import zipfile
import hashlib
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
KOLLAM_LAT, KOLLAM_LON = 8.8932, 76.6141
KERALA_BBOX = (74.8, 8.0, 77.4, 12.8)  # min_lon, min_lat, max_lon, max_lat

# Public datasets
FLOODNET_KAGGLE = "https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation"
SENTINEL_HUB = "https://scihub.copernicus.eu/dhus/odata/v1"
BHUVAN_WMS = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/gwc/service/wms"

# Kerala IMD district rainfall stations
KERALA_STATIONS = {
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kollam":             (8.8932, 76.6141),
    "Pathanamthitta":     (9.2648, 76.7870),
    "Alappuzha":          (9.4981, 76.3388),
    "Kottayam":           (9.5916, 76.5222),
    "Idukki":             (9.9189, 76.9749),
    "Ernakulam":          (9.9816, 76.2999),
    "Thrissur":           (10.5276, 76.2144),
    "Palakkad":           (10.7867, 76.6548),
    "Malappuram":         (11.0730, 76.0740),
    "Kozhikode":          (11.2588, 75.7804),
    "Wayanad":            (11.6854, 76.1320),
    "Kannur":             (11.8745, 75.3704),
    "Kasaragod":          (12.4996, 74.9869),
}


# ─────────────────────────────────────────────────────────
# 1. BHUVAN WMS TILE FETCHER
# ─────────────────────────────────────────────────────────
def fetch_bhuvan_tile(
    lat: float,
    lon: float,
    zoom: int = 12,
    layer: str = "india_eo_stack",
    size: int = 256,
    save_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Fetch satellite tile from ISRO Bhuvan WMS service.
    Returns (H, W, 3) uint8 RGB array or None on failure.
    """
    # Calculate WMS bounding box from tile coordinates
    delta = 0.05 / (2 ** (zoom - 10))
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "BBOX": bbox,
        "WIDTH": size,
        "HEIGHT": size,
        "SRS": "EPSG:4326",
        "FORMAT": "image/png",
        "TRANSPARENT": "true",
    }

    try:
        r = requests.get(BHUVAN_WMS, params=params, timeout=10)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        arr = np.array(img)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)
        return arr
    except Exception as e:
        print(f"[Bhuvan] Tile fetch failed ({lat:.3f},{lon:.3f}): {e}")
        return None


def download_bhuvan_grid(
    output_dir: str,
    n_tiles: int = 50,
    zoom: int = 12,
) -> List[str]:
    """Download a grid of Bhuvan tiles covering Kerala."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    min_lon, min_lat, max_lon, max_lat = KERALA_BBOX

    lats = np.linspace(min_lat + 0.1, max_lat - 0.1, int(np.sqrt(n_tiles)) + 1)
    lons = np.linspace(min_lon + 0.1, max_lon - 0.1, int(np.sqrt(n_tiles)) + 1)

    tasks = [(lat, lon) for lat in lats for lon in lons][:n_tiles]
    saved = []

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {}
        for i, (lat, lon) in enumerate(tasks):
            save_path = f"{output_dir}/bhuvan_{i:04d}_{lat:.3f}_{lon:.3f}.png"
            futures[ex.submit(fetch_bhuvan_tile, lat, lon, zoom, save_path=save_path)] = save_path

        for fut in as_completed(futures):
            path = futures[fut]
            arr = fut.result()
            if arr is not None:
                saved.append(path)
                sys.stdout.write(f"\r[Bhuvan] Downloaded {len(saved)}/{len(tasks)} tiles")
                sys.stdout.flush()

    print(f"\n[Bhuvan] Saved {len(saved)} tiles to {output_dir}")
    return saved


# ─────────────────────────────────────────────────────────
# 2. IMD RAINFALL DATA (via Open-Meteo free API)
# ─────────────────────────────────────────────────────────
def download_imd_rainfall(
    output_dir: str,
    start_date: str = "2018-01-01",
    end_date: str = "2023-12-31",
) -> str:
    """
    Download historical rainfall for all Kerala stations using Open-Meteo.
    Saves to {output_dir}/kerala_rainfall.csv
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_records = []

    print("[IMD] Downloading historical rainfall for Kerala stations …")
    for station_name, (lat, lon) in KERALA_STATIONS.items():
        try:
            url = (
                f"https://archive-api.open-meteo.com/v1/archive?"
                f"latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&daily=precipitation_sum,wind_speed_10m_max,temperature_2m_max,"
                f"temperature_2m_min,relative_humidity_2m_max,surface_pressure_mean"
                f"&timezone=Asia%2FKolkata"
            )
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            d = r.json()

            daily = d.get("daily", {})
            dates = daily.get("time", [])
            for i, dt in enumerate(dates):
                all_records.append({
                    "date": dt,
                    "station": station_name,
                    "lat": lat,
                    "lon": lon,
                    "rainfall_mm": daily.get("precipitation_sum", [0])[i] or 0.0,
                    "wind_speed_kmh": daily.get("wind_speed_10m_max", [0])[i] or 0.0,
                    "temp_c": (
                        ((daily.get("temperature_2m_max", [0])[i] or 0) +
                         (daily.get("temperature_2m_min", [0])[i] or 0)) / 2
                    ),
                    "humidity_pct": daily.get("relative_humidity_2m_max", [0])[i] or 0.0,
                    "pressure_hpa": daily.get("surface_pressure_mean", [1013])[i] or 1013.0,
                })

            time.sleep(0.5)  # rate limit
            print(f"  ✓ {station_name}: {len(dates)} days")

        except Exception as e:
            print(f"  ✗ {station_name}: {e}")

    df = pd.DataFrame(all_records)
    out_path = f"{output_dir}/kerala_rainfall.csv"
    df.to_csv(out_path, index=False)
    print(f"[IMD] Saved {len(df)} records → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────
# 3. FLOODNET DATASET (Kaggle public)
# ─────────────────────────────────────────────────────────
def prepare_floodnet_dataset(
    raw_dir: str,
    output_dir: str,
    damage_threshold: float = 0.3,
) -> Tuple[str, str]:
    """
    Prepare FloodNet segmentation dataset for classification.
    Converts binary flood masks to 4-class damage labels.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"[FloodNet] Dataset not found at {raw_dir}")
        print(f"[FloodNet] Download from: {FLOODNET_KAGGLE}")
        print("[FloodNet] Or run: kaggle datasets download faizalkarim/flood-area-segmentation")
        return generate_placeholder_csv(output_dir)

    # Find image/mask pairs
    image_dir = raw_path / "Image"
    mask_dir = raw_path / "Mask"

    if not image_dir.exists():
        image_dir = raw_path
        mask_dir = raw_path

    rows = []
    for img_path in sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = mask_dir / (img_path.stem + "_mask" + img_path.suffix)

        if mask_path.exists():
            try:
                mask = np.array(Image.open(mask_path).convert("L"))
                flood_ratio = (mask > 127).mean()

                # Map flood coverage to damage class
                if flood_ratio < 0.05:
                    label = 0
                elif flood_ratio < 0.25:
                    label = 1
                elif flood_ratio < 0.60:
                    label = 2
                else:
                    label = 3

                rows.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "flood_ratio": round(float(flood_ratio), 4),
                    "label": label,
                    "tweet_text": _generate_tweet(label),
                    **_generate_weather(label),
                })
            except Exception as e:
                print(f"  Skip {img_path.name}: {e}")

    if not rows:
        return generate_placeholder_csv(output_dir)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(0.8 * len(df))
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    train_path = f"{output_dir}/train.csv"
    val_path = f"{output_dir}/val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"[FloodNet] {len(df)} samples prepared (train={len(train_df)}, val={len(val_df)})")
    return train_path, val_path


def generate_placeholder_csv(output_dir: str, n: int = 500) -> Tuple[str, str]:
    """Generate placeholder CSVs for testing without real data."""
    from model.preprocess import generate_synthetic_dataset
    return generate_synthetic_dataset(output_dir, n_samples=n)


def _generate_tweet(label: int) -> str:
    tweets = {
        0: "No flood reported in this area, situation normal #Kerala",
        1: "Minor flooding in low-lying areas of Kerala, stay alert",
        2: "Major flooding reported, rescue teams deployed #KeralaFloods",
        3: "Catastrophic flood! Entire villages submerged, NDRF activated #KeralaEmergency",
    }
    return tweets.get(label, "Flood situation monitoring in Kerala")


def _generate_weather(label: int) -> dict:
    import random
    bases = {
        0: (2, 2.0, 70, 28, 1010),
        1: (25, 3.5, 80, 27, 1006),
        2: (80, 6.0, 88, 26, 1002),
        3: (180, 12.0, 95, 25, 995),
    }
    rain, river, hum, temp, pres = bases[label]
    return {
        "rainfall_mm": round(rain + random.gauss(0, rain * 0.2), 1),
        "river_level_m": round(river + random.gauss(0, river * 0.1), 2),
        "wind_speed_kmh": round(random.uniform(10, 30 + label * 15), 1),
        "humidity_pct": round(hum + random.gauss(0, 5), 1),
        "temp_c": round(temp + random.gauss(0, 1), 1),
        "pressure_hpa": round(pres + random.gauss(0, 3), 1),
        "duration_hrs": round(random.uniform(1, 6 + label * 10), 1),
        "upstream_rainfall_mm": round(rain * random.uniform(0.4, 1.3), 1),
        "soil_moisture": round(min(0.3 + label * 0.17 + random.gauss(0, 0.05), 0.98), 2),
        "previous_flood_days": random.randint(0, label * 4),
        "lat": round(KOLLAM_LAT + random.uniform(-2, 2), 4),
        "lon": round(KOLLAM_LON + random.uniform(-1, 1), 4),
    }


# ─────────────────────────────────────────────────────────
# 4. SENTINEL-1 SAR (via Copernicus Scihub – needs account)
# ─────────────────────────────────────────────────────────
def download_sentinel1(
    output_dir: str,
    username: str,
    password: str,
    start_date: str = "2018-07-01",
    end_date: str = "2018-09-30",
    max_products: int = 10,
) -> List[str]:
    """
    Download Sentinel-1 GRD SAR products for Kerala.
    Requires Copernicus Scihub account: https://scihub.copernicus.eu/
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    footprint = (
        f"POLYGON(({KERALA_BBOX[0]} {KERALA_BBOX[1]},"
        f"{KERALA_BBOX[2]} {KERALA_BBOX[1]},"
        f"{KERALA_BBOX[2]} {KERALA_BBOX[3]},"
        f"{KERALA_BBOX[0]} {KERALA_BBOX[3]},"
        f"{KERALA_BBOX[0]} {KERALA_BBOX[1]}))"
    )

    query = (
        f"https://scihub.copernicus.eu/dhus/search?format=json&rows={max_products}"
        f"&q=platformname:Sentinel-1 AND producttype:GRD"
        f" AND footprint:\"Intersects({footprint})\""
        f" AND beginposition:[{start_date}T00:00:00.000Z TO {end_date}T23:59:59.999Z]"
    )

    try:
        r = requests.get(query, auth=(username, password), timeout=30)
        r.raise_for_status()
        data = r.json()
        products = data.get("feed", {}).get("entry", [])
        print(f"[Sentinel-1] Found {len(products)} products")

        saved = []
        for p in products[:max_products]:
            uuid = p["id"]
            title = p["title"]
            url = f"{SENTINEL_HUB}/Products('{uuid}')/$value"
            out_path = f"{output_dir}/{title}.zip"

            print(f"  Downloading {title} …")
            try:
                r2 = requests.get(url, auth=(username, password), stream=True, timeout=120)
                r2.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r2.iter_content(chunk_size=8192):
                        f.write(chunk)
                saved.append(out_path)
                print(f"  ✓ Saved {out_path}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")

        return saved

    except Exception as e:
        print(f"[Sentinel-1] Error: {e}")
        print("[Sentinel-1] Register at https://scihub.copernicus.eu/ for SAR data")
        return []


# ─────────────────────────────────────────────────────────
# 5. VERIFY DATASET INTEGRITY
# ─────────────────────────────────────────────────────────
def verify_dataset(data_dir: str) -> Dict:
    """Check dataset files and return statistics."""
    import glob
    data_dir = Path(data_dir)
    stats = {"status": "ok", "issues": []}

    for split in ["train", "val"]:
        csv_path = data_dir / f"{split}.csv"
        if not csv_path.exists():
            stats["issues"].append(f"Missing {split}.csv")
            stats["status"] = "error"
            continue

        df = pd.read_csv(csv_path)
        stats[split] = {
            "rows": len(df),
            "label_distribution": df["label"].value_counts().to_dict(),
            "columns": list(df.columns),
        }

        # Check label range
        if not df["label"].between(0, 3).all():
            stats["issues"].append(f"{split}.csv has invalid labels (must be 0-3)")

    return stats


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Kerala Flood Dataset Downloader")
    p.add_argument("--output_dir",    default="data")
    p.add_argument("--download_rain", action="store_true", help="Download IMD rainfall")
    p.add_argument("--download_bhuvan", action="store_true", help="Download Bhuvan tiles")
    p.add_argument("--floodnet_dir",  default="", help="Path to FloodNet raw dataset")
    p.add_argument("--sentinel_user", default="", help="Copernicus Scihub username")
    p.add_argument("--sentinel_pass", default="", help="Copernicus Scihub password")
    p.add_argument("--synthetic",     action="store_true", default=True)
    p.add_argument("--n_synthetic",   type=int, default=2000)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 60)
    print("  Kerala Flood Dataset Preparation")
    print("═" * 60)

    # Synthetic data (always works, no API keys needed)
    if args.synthetic:
        sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
        from preprocess import generate_synthetic_dataset
        train_csv, val_csv = generate_synthetic_dataset(
            str(output_dir / "synthetic"), args.n_synthetic
        )
        print(f"\n[Synthetic] ✓ train: {train_csv}")
        print(f"[Synthetic] ✓ val:   {val_csv}")

    # IMD rainfall
    if args.download_rain:
        rain_csv = download_imd_rainfall(str(output_dir / "imd"))
        print(f"[IMD] ✓ Rainfall CSV: {rain_csv}")

    # Bhuvan satellite tiles
    if args.download_bhuvan:
        tile_dir = str(output_dir / "bhuvan_tiles")
        tiles = download_bhuvan_grid(tile_dir, n_tiles=20)
        print(f"[Bhuvan] ✓ {len(tiles)} tiles saved")

    # FloodNet preparation
    if args.floodnet_dir:
        train_csv, val_csv = prepare_floodnet_dataset(
            args.floodnet_dir, str(output_dir / "floodnet")
        )
        print(f"[FloodNet] ✓ train: {train_csv}")
        print(f"[FloodNet] ✓ val:   {val_csv}")

    # Sentinel-1
    if args.sentinel_user and args.sentinel_pass:
        sar_files = download_sentinel1(
            str(output_dir / "sentinel1"),
            args.sentinel_user, args.sentinel_pass,
        )
        print(f"[Sentinel-1] ✓ {len(sar_files)} SAR products downloaded")

    # Verify
    verify_dir = str(output_dir / "synthetic") if args.synthetic else str(output_dir / "floodnet")
    if Path(verify_dir).exists():
        stats = verify_dataset(verify_dir)
        print(f"\n[Verify] Dataset status: {stats['status']}")
        if stats.get("train"):
            print(f"  Train: {stats['train']['rows']} rows | "
                  f"Labels: {stats['train']['label_distribution']}")
        if stats.get("val"):
            print(f"  Val:   {stats['val']['rows']} rows | "
                  f"Labels: {stats['val']['label_distribution']}")

    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()