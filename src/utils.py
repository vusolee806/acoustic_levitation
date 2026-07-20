import os

def get_asset_path(filename: str) -> str:
    """Safely resolves paths to your MuJoCo model files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "assets", filename)
