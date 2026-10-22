import os
import sys

# Ensure root directory is accessible by Python runtime
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app import app

# Vercel Serverless Function entrypoint
app = app
