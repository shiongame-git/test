"""GA timing entry with the same metadata and timing scope as API/GUI/compare."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from packing import (BLPriority, GAConfig, GAExecutionConfig, RunOptions, parse_box_sizes_bytes,
                     run_bl_ga, run_pp_ga, write_run_metadata_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--decoder', choices=('BL', 'PP'), default='BL')
    parser.add_argument('--priority', choices=('BottomFirst', 'BackFirst'), default='BottomFirst')
    parser.add_argument('--population', type=int, default=100)
    parser.add_argument('--generations', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be >= 1')
    raw = args.fixture.read_bytes()
    container, boxes = parse_box_sizes_bytes(raw, source=str(args.fixture))
    options = RunOptions(origin='benchmark', source=str(args.fixture), fixture_sha256=hashlib.sha256(raw).hexdigest())
    config = GAConfig(population_size=args.population, generation_count=args.generations)
    execution = GAExecutionConfig(args.workers, args.cache)
    runner = run_bl_ga if args.decoder == 'BL' else run_pp_ga
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(args.repeat):
        run = runner(container, boxes, priority=BLPriority(args.priority), config=config,
                     seed=args.seed, execution=execution, run_options=options)
        metadata = run.run_metadata
        path = args.output_dir / f"ga-{metadata['run_id']}.run.json"
        write_run_metadata_json(path, metadata)
        print(f"{path}: elapsed_s={metadata['statistics']['elapsed_seconds']:.6f}")


if __name__ == '__main__':
    main()
