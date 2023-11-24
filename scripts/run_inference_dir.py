import pyrallis
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class InferConfig:
    cfg_path: Path
    ckpt: Path
    output_dir: Path
    vid_dir: Path
    filenames: Optional[list[Path]]


@pyrallis.wrap()
def main(cfg: InferConfig):
    for i, vid_path in enumerate(cfg.vid_dir.rglob("*.mp4")):
        vid_output_dir = cfg.output_dir / vid_path.stem
        vid_output_dir.mkdir(exist_ok=True)
        subprocess.run(
            f"./scripts/inference.sh {cfg.cfg_path} {cfg.ckpt} {vid_path} {vid_output_dir}",
            shell=True,
        )


if __name__ == "__main__":
    main()
