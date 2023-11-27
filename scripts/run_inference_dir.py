import pyrallis
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class InferConfig:
    root_dir: Path
    cfg_path: Path
    ckpt: Path
    output_dir: Path
    vid_dir: Path
    filenames: list

    def __post_init__(self):
        self.cfg_path = self.root_dir / self.cfg_path
        self.ckpt = self.root_dir / self.ckpt


@pyrallis.wrap()
def main(cfg: InferConfig):
    vid_dir = cfg.vid_dir
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    files = vid_dir.rglob("*.mp4")
    if len(cfg.filenames) > 0:
        files = [vid_dir / name for name in cfg.filenames]

    for i, vid_path in enumerate(files):
        vid_output_dir = cfg.output_dir / vid_path.stem
        vid_output_dir.mkdir(exist_ok=True)
        subprocess.run(
            f"{cfg.root_dir}/scripts/inference.sh {cfg.cfg_path} {cfg.ckpt} {vid_path} {vid_output_dir}",
            shell=True,
        )


if __name__ == "__main__":
    main()
