#!/usr/bin/env python
"""
PDF ingestion script — canonical CLI entry point for document indexing.

This script provides a simplified interface for indexing PDF documents
into Qdrant. For advanced options (custom model, threshold, search),
use the underlying module directly:

    python -m database.semantic_chunker index ./archives/ --threshold 0.8
    python -m database.semantic_chunker search "correção de solo"

Usage:
    python scripts/ingest.py ./archives/
    python scripts/ingest.py ./archives/document.pdf
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.semantic_chunker import main as chunker_main


def main() -> None:
    """Entry point for the ingest script."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <path>")
        print("  <path> can be a directory of PDFs or a single PDF file")
        print()
        print("For advanced options, use the underlying module:")
        print("  python -m database.semantic_chunker index <path> [--model MODEL] [--threshold N]")
        print("  python -m database.semantic_chunker search <query> [--top-k N]")
        sys.exit(1)

    # Forward to semantic_chunker with 'index' command
    sys.argv = [sys.argv[0], "index"] + sys.argv[1:]
    chunker_main()


if __name__ == "__main__":
    main()
