"""
Migrate existing JSON strategy folders to SQLite.

Usage:
    python scripts/migrate_to_sqlite.py                    # migrate all folders
    python scripts/migrate_to_sqlite.py --folder strategies_2026-02-27  # specific folder
    python scripts/migrate_to_sqlite.py --dry-run          # show what would be migrated
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.factory.strategy_factory import StrategyFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_folder(folder: Path, dry_run: bool = False) -> int:
    """Migrate a single JSON strategy folder to SQLite.
    Returns number of strategies migrated."""

    index_file = folder / 'index.json'
    db_file = folder / 'strategies.db'

    if not index_file.exists():
        logger.warning(f"  No index.json in {folder.name}, skipping")
        return 0

    if db_file.exists():
        logger.info(f"  {folder.name}: strategies.db already exists, skipping")
        return 0

    # Load index
    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)

    strategies = index.get('strategies', {})
    if not strategies:
        logger.info(f"  {folder.name}: empty index, skipping")
        return 0

    if dry_run:
        logger.info(f"  [DRY RUN] {folder.name}: would migrate {len(strategies)} strategies")
        return len(strategies)

    # Create factory (this initializes the DB)
    factory = StrategyFactory(storage_dir=str(folder))

    # Batch load all JSON files and insert
    batch = []
    errors = 0
    for sid, info in strategies.items():
        json_file = folder / f"{sid}.json"
        if not json_file.exists():
            errors += 1
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception):
            errors += 1
            continue

        config = {
            'id': data['strategy_id'],
            'asset': data['asset'],
            'strategy_type': data['strategy_type'],
            'strategy_name': data['strategy_name'],
            'params': data.get('params', {}),
            'related_asset': data.get('related_asset'),
        }
        perf = data.get('performance', {})
        batch.append((config, perf))

    if batch:
        count = factory.save_strategies_batch(batch)
        # Restore is_active flags
        for sid, info in strategies.items():
            if info.get('is_active', False):
                factory.set_active(sid, True)
        factory.close()
        logger.info(f"  {folder.name}: migrated {count} strategies ({errors} errors)")
        return count

    factory.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description='Migrate JSON strategies to SQLite')
    parser.add_argument('--folder', default=None, help='Specific folder name to migrate')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated')
    parser.add_argument('--delete-json', action='store_true',
                        help='Delete JSON files after successful migration')
    args = parser.parse_args()

    factory_base = Path(__file__).parent.parent / 'src' / 'factory'

    if args.folder:
        folders = [factory_base / args.folder]
    else:
        folders = sorted([
            d for d in factory_base.iterdir()
            if d.is_dir() and d.name.startswith('strategies_')
        ])

    if not folders:
        print("No strategy folders found.")
        return

    print(f"Found {len(folders)} strategy folder(s) to process.\n")

    total_migrated = 0
    for folder in folders:
        count = migrate_folder(folder, dry_run=args.dry_run)
        total_migrated += count

        # Optionally delete JSON files (also when DB already existed)
        if not args.dry_run and args.delete_json and (folder / 'strategies.db').exists():
            deleted = 0
            for f in folder.glob('*.json'):
                f.unlink()
                deleted += 1
            if deleted:
                logger.info(f"  Deleted {deleted} JSON files from {folder.name}")

    print(f"\nDone! Total migrated: {total_migrated} strategies across {len(folders)} folders.")


if __name__ == '__main__':
    main()
