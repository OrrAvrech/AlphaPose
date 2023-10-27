import subprocess
from pathlib import Path

cfg_path = "./configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml"
ckpt = "./pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth"
output_dir = Path("./examples/res")

vid_dir = Path("/proj/vondrick2/orr/Data/human-feedback/segments/video")
for i, vid_path in enumerate(vid_dir.rglob("*.mp4")):
    vid_output_dir = output_dir / vid_path.stem
    vid_output_dir.mkdir(exist_ok=True)
    subprocess.run(f"./scripts/inference.sh {cfg_path} {ckpt} {vid_path} {vid_output_dir}", shell=True)
