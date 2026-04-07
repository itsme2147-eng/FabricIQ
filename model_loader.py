"""
model_loader.py
===============
Downloads your trained .pt model files from Google Drive using gdown.
Models stay on Google Drive — no GitHub file size issues.

Usage in Streamlit:
    from model_loader import download_models, load_all_models
    paths = download_models(GDRIVE_IDS)
    models = load_all_models(paths)
"""

import os
import numpy as np
import streamlit as st

# ── Local cache directory (Streamlit Cloud temp storage) ─────────────────────
MODEL_CACHE = "/tmp/fabriciq_models"


def _ensure_dir():
    os.makedirs(MODEL_CACHE, exist_ok=True)


def download_single(file_id: str, filename: str, force: bool = False) -> str:
    """
    Download one file from Google Drive via gdown.
    Returns local path. Skips if already cached unless force=True.
    """
    _ensure_dir()
    local_path = os.path.join(MODEL_CACHE, filename)

    if os.path.exists(local_path) and not force:
        size_mb = os.path.getsize(local_path) / 1e6
        return local_path  # already cached

    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown not installed. Add 'gdown' to requirements.txt"
        )

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, local_path, quiet=False)

    if not os.path.exists(local_path) or os.path.getsize(local_path) < 100:
        raise RuntimeError(
            f"Download failed or file too small: {filename}\n"
            f"Make sure the file is shared as 'Anyone with the link can view'."
        )
    return local_path


def download_models(gdrive_ids: dict, progress_callback=None) -> dict:
    """
    Download all models listed in gdrive_ids dict.
    
    gdrive_ids format:
        {
          'fae_best.pt':                    'YOUR_FILE_ID_HERE',
          'fae_mu.npy':                     'YOUR_FILE_ID_HERE',
          'fae_sig.npy':                    'YOUR_FILE_ID_HERE',
          'mobilenet_v3_small_recon.pt':    'YOUR_FILE_ID_HERE',
          'mobilenet_v3_small_selfsup.pt':  'YOUR_FILE_ID_HERE',
          'efficientnet_b0_recon.pt':       'YOUR_FILE_ID_HERE',
          'efficientnet_b0_selfsup.pt':     'YOUR_FILE_ID_HERE',
        }
    
    Returns dict: filename -> local_path (only successfully downloaded files).
    """
    _ensure_dir()
    downloaded = {}
    errors = {}

    for filename, file_id in gdrive_ids.items():
        if not file_id or file_id.startswith('YOUR_'):
            continue  # skip unconfigured entries
        try:
            path = download_single(file_id, filename)
            downloaded[filename] = path
            if progress_callback:
                progress_callback(filename, True, None)
        except Exception as e:
            errors[filename] = str(e)
            if progress_callback:
                progress_callback(filename, False, str(e))

    return downloaded, errors


def get_cached_models() -> list:
    """Return list of already-cached model filenames."""
    _ensure_dir()
    return [f for f in os.listdir(MODEL_CACHE)
            if f.endswith(('.pt', '.npy'))]


def clear_cache():
    """Delete all cached model files."""
    import shutil
    if os.path.exists(MODEL_CACHE):
        shutil.rmtree(MODEL_CACHE)
    _ensure_dir()


def load_all_models(downloaded_paths: dict) -> dict:
    """
    Load all downloaded model weights into memory.
    Returns dict suitable for score_image_deep().
    """
    from fabriciq_models import load_deep_models
    return load_deep_models(MODEL_CACHE)


def extract_gdrive_id(url_or_id: str) -> str:
    """
    Extract file ID from a Google Drive share URL or return as-is if already an ID.
    
    Handles formats:
      https://drive.google.com/file/d/FILE_ID/view?usp=sharing
      https://drive.google.com/uc?id=FILE_ID
      FILE_ID  (plain ID)
    """
    import re
    # Pattern: /d/FILE_ID/ or id=FILE_ID
    match = re.search(r'/d/([a-zA-Z0-9_-]{25,})', url_or_id)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]{25,})', url_or_id)
    if match:
        return match.group(1)
    # Assume it's already a raw ID
    if len(url_or_id) > 20 and '/' not in url_or_id:
        return url_or_id.strip()
    return url_or_id
