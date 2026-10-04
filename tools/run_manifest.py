"""Build, write or compare reproducible run manifests (plan Stage 1).

    PYTHONPATH=. python tools/run_manifest.py build \
        --config configs/minecraft_skybot.yaml --seed 0 --out runlogs/manifest.json
    PYTHONPATH=. python tools/run_manifest.py compare a.json b.json

`build` mirrors run_minecraft.py's config resolution (--seed, --forever).
`--no-hash-checkpoints` skips hashing brain state (a multi-GB skill bank);
those hashes are then recorded as "unknown", never omitted.
`compare` prints differing dotted keys (timestamps ignored); exit 1 if any.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.foundation.runtime.manifest import (  # noqa: E402
    build_manifest, compare_manifests, load_manifest, write_manifest)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", default="configs/minecraft_skybot.yaml")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--forever", action="store_true")
    b.add_argument("--timesteps", default="absent")
    b.add_argument("--repo-root", default=".")
    b.add_argument("--run-root", default=None,
                   help="where relative brain-state paths live (default: repo root)")
    b.add_argument("--no-hash-checkpoints", action="store_true")
    b.add_argument("--out", default=None, help="write JSON here (else stdout)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        man = build_manifest(args.config, repo_root=args.repo_root,
                             seed=args.seed, forever=args.forever,
                             timesteps=args.timesteps, run_root=args.run_root,
                             hash_checkpoints=not args.no_hash_checkpoints)
        if args.out:
            write_manifest(man, args.out)
            print(f"manifest -> {args.out} (source "
                  f"{man['source']['tree_sha256'][:12]}, "
                  f"{man['source']['n_files']} files, git "
                  f"{str(man['source']['git_revision'])[:12]} dirty="
                  f"{man['source']['git_dirty']})")
        else:
            print(json.dumps(man, indent=2, sort_keys=True, default=str))
        return 0
    diff = compare_manifests(load_manifest(args.a), load_manifest(args.b))
    for k in diff:
        print(k)
    print(f"{len(diff)} differing key(s)")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
