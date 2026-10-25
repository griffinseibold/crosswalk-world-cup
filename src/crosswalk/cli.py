"""`crosswalk` command line: one entry point for every Job and the Deployment.

    crosswalk validate                     Milestone 0 GPU check
    crosswalk serve   --config inf.yaml    inference service (Deployment)
    crosswalk infer   --config inf.yaml    one offline pass over the footage
    crosswalk benchmark --config bench.yaml   Experiment A
    crosswalk train   --config train.yaml  training (resumes automatically)
    crosswalk evaluate --config train.yaml [--checkpoint s3://...]
    crosswalk make-synthetic out/          tiny dataset for smoke tests
    crosswalk hash-dataset --config train.yaml
    crosswalk sync s3://datasets/bdd100k=/scratch/data/bdd100k
    crosswalk stage tracking --config stages.yaml   a streaming stage (Deployment)

``--set a.b=value`` overrides any config key (parsed as YAML), so a Job can
reuse one ConfigMap'd file with per-run changes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # boto and MLflow are chatty at INFO.
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set batching.max_batch_size=4",
    )


def cmd_validate(args) -> int:
    from crosswalk import validate

    result = validate.run(
        device_name=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        model_name=args.model,
    )
    print(validate.report(result))
    if args.output:
        from crosswalk.storage import atomic_write_bytes

        atomic_write_bytes(args.output, validate.to_json(result).encode())
    return 0 if result["ok"] else 1


def cmd_serve(args, serve: bool) -> int:
    from crosswalk.config import InferenceConfig, load_config
    from crosswalk.inference import service

    config = load_config(InferenceConfig, args.config, args.overrides)
    if not serve:
        # Offline: every frame, as fast as possible, then exit.
        config = config.model_copy(
            update={"source": config.source.model_copy(update={"realtime": False, "loop": False})}
        )
    state = service.run(config, serve=serve)
    print(json.dumps({k: v for k, v in state.items() if k != "runtime"}, indent=2, default=str))
    return 0


def cmd_benchmark(args) -> int:
    from crosswalk.benchmark import harness
    from crosswalk.config import BenchmarkConfig, load_config

    config = load_config(BenchmarkConfig, args.config, args.overrides)
    result = harness.run_benchmark(config)
    problems = harness.validate_result(result)
    if problems:
        print("result failed schema validation: " + "; ".join(problems), file=sys.stderr)
        return 1
    path = harness.write_result(result, args.output_dir or config.output_dir)
    print(harness.markdown_table(result))
    print(f"\nwrote {path}")
    if args.s3_prefix:
        from crosswalk.storage import open_store

        open_store(args.s3_prefix).put_file(str(path), path.name)
        print(f"uploaded to {args.s3_prefix.rstrip('/')}/{path.name}")
    return 0


def cmd_train(args) -> int:
    from crosswalk.config import TrainingConfig, load_config
    from crosswalk.training.loop import train

    config = load_config(TrainingConfig, args.config, args.overrides)
    summary = train(config)
    print(json.dumps(summary.to_dict(), indent=2, default=str))
    return 0


def cmd_evaluate(args) -> int:
    from crosswalk.config import TrainingConfig, load_config
    from crosswalk.training.loop import evaluate_only

    config = load_config(TrainingConfig, args.config, args.overrides)
    print(json.dumps(evaluate_only(config, args.checkpoint), indent=2, default=str))
    return 0


def cmd_make_synthetic(args) -> int:
    from crosswalk import synthetic

    paths = synthetic.make_dataset(
        args.output, images=args.images, size=(args.width, args.height), seed=args.seed
    )
    if args.video:
        from crosswalk.inference.video import write_test_video

        paths["video"] = str(Path(args.output) / "clip.mp4")
        write_test_video(
            paths["video"], frames=args.video_frames, size=(args.width * 4, args.height * 4)
        )
    print(json.dumps(paths, indent=2))
    return 0


def cmd_hash_dataset(args) -> int:
    from crosswalk.config import TrainingConfig, load_config
    from crosswalk.training import datasets, hashing

    config = load_config(TrainingConfig, args.config, args.overrides)
    classes = list(config.model.classes or [])
    out = {}
    for split in ("train", "val", "eval"):
        ds_config = getattr(config.data, split)
        if ds_config is None:
            continue
        parsed = datasets.load(ds_config, classes, config.seed)
        out[split] = hashing.dataset_manifest(parsed, ds_config.images, config.data.hash_images)
    print(json.dumps(out, indent=2))
    return 0


def cmd_sync(args) -> int:
    from crosswalk.storage import sync_prefix

    for pair in args.pairs:
        source, _, dest = pair.partition("=")
        if not source.startswith("s3://") or not dest:
            raise SystemExit(f"expected s3://bucket/prefix=/local/dir, got {pair!r}")
        counts = sync_prefix(source, dest)
        print(f"{source} -> {dest}: {counts}")
    return 0


def cmd_stage(args) -> int:
    from crosswalk.stages import run_stage

    state = run_stage(args.name, args.config, args.overrides, port=args.port)
    print(json.dumps(state, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crosswalk", description=__doc__.split("\n\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Milestone 0: train a few detector steps on the GPU")
    p.add_argument("--device", default="gpu", help="gpu (default), rocm, cuda, auto or cpu")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument(
        "--model", default="fcos_resnet50_fpn", choices=["fcos_resnet50_fpn", "fcos_tiny"]
    )
    p.add_argument("--output", help="also write the full result as JSON here")
    p.set_defaults(func=cmd_validate)

    for name, serve in (("serve", True), ("infer", False)):
        p = sub.add_parser(
            name, help="inference service" if serve else "one offline inference pass"
        )
        _config_args(p)
        p.set_defaults(func=lambda a, serve=serve: cmd_serve(a, serve))

    p = sub.add_parser("benchmark", help="Experiment A: batching strategy benchmark")
    _config_args(p)
    p.add_argument("--output-dir", help="overrides output_dir in the config")
    p.add_argument("--s3-prefix", help="also upload the result file, e.g. s3://datasets/results/")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("train", help="train or resume a run")
    _config_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score a checkpoint and the baseline on the eval set")
    _config_args(p)
    p.add_argument("--checkpoint", help="local path or s3:// URI of a training checkpoint")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("make-synthetic", help="write a tiny synthetic dataset")
    p.add_argument("output")
    p.add_argument("--images", type=int, default=16)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", action="store_true", help="also write a short test clip")
    p.add_argument("--video-frames", type=int, default=60)
    p.set_defaults(func=cmd_make_synthetic)

    p = sub.add_parser("sync", help="download s3:// prefixes to local directories")
    p.add_argument("pairs", nargs="+", metavar="s3://bucket/prefix=/local/dir")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("stage", help="run a streaming stage (tracking, indicators, ranking)")
    p.add_argument("name", choices=["tracking", "indicators", "ranking"])
    _config_args(p)
    p.add_argument("--port", type=int, default=8080, help="probes, /metrics and stage routes")
    p.set_defaults(func=cmd_stage)

    p = sub.add_parser(
        "hash-dataset", help="print the dataset manifests a training run would record"
    )
    _config_args(p)
    p.set_defaults(func=cmd_hash_dataset)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
