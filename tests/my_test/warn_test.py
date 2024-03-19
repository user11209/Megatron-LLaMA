import warnings
from warnings import warn

def do_warn(e):
    warn(
        f"Failed to load image Python extension: '{e}'"
        f"If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. "
        f"Otherwise, there might be something wrong with your environment. "
        f"Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?"
    )

def do_warn_hello(e):
  warn(f"hello, and {e}")

do_warn("shit")
do_warn_hello("shit")
warnings.filterwarnings("ignore", message="hello")
warnings.filterwarnings("ignore", message="Failed to load")
do_warn("shit")
do_warn("shit")