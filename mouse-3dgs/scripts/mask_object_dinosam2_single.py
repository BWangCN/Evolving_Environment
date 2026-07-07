"""DINO+SAM2 video masking for IMG_0323 (SeeDo technique, mouse-only, no VLM/inpaint).
GroundingDINO('mouse') box on a seed frame -> SAM vit_h precise mask -> SAM2 video predictor
add_new_mask + propagate_in_video across all frames -> temporally-consistent binary mouse masks.
Outputs object_mask_dinosam2/<stem>.png (mouse=1, bg=0) at the undistorted training resolution."""
import os, sys, glob, shutil, argparse
import numpy as np, torch
from PIL import Image
from torchvision.ops import box_convert

ROOT = "/home/ravi/Desktop/Ravi/SeeDo"
sys.path.append(ROOT); sys.path.append(os.path.join(ROOT, "GroundingDINO"))
from groundingdino.util.inference import load_image_from_array, predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.models import build_model
from groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor
from huggingface_hub import hf_hub_download

IMG_DIR = "/home/ravi/Desktop/Ravi/3dgs/data/IMG_0323/images"      # undistorted, 1078x1917
OUT_DIR = "/home/ravi/Desktop/Ravi/3dgs/data/IMG_0323/object_mask_dinosam2"
VID_DIR = "/home/ravi/Desktop/Ravi/3dgs/data/IMG_0323/_sam2_frames"  # 0-indexed jpgs for SAM2
PROMPT = "mouse"; BOX_T = 0.3; TEXT_T = 0.25
ap = argparse.ArgumentParser(); ap.add_argument("--seed_frame", type=int, default=0); args = ap.parse_args()
DEVICE = "cuda"

def load_dino():
    repo = "ShilongLiu/GroundingDINO"
    cfg = hf_hub_download(repo_id=repo, filename="GroundingDINO_SwinB.cfg.py")
    ckpt = hf_hub_download(repo_id=repo, filename="groundingdino_swinb_cogcoor.pth")
    a = SLConfig.fromfile(cfg); a.device = DEVICE
    m = build_model(a); m.load_state_dict(clean_state_dict(torch.load(ckpt, map_location="cpu")["model"]), strict=False)
    return m.eval().to(DEVICE)

frames = sorted(glob.glob(f"{IMG_DIR}/*.jpg"))
stems = [os.path.basename(f).split(".")[0] for f in frames]
N = len(frames); print(f"frames: {N}  seed_frame idx={args.seed_frame} ({stems[args.seed_frame]}.jpg)")

# 0-indexed frame dir for SAM2 (symlinks; idx i -> original stems[i])
if os.path.isdir(VID_DIR): shutil.rmtree(VID_DIR)
os.makedirs(VID_DIR)
for i, f in enumerate(frames): os.symlink(f, f"{VID_DIR}/{i:05d}.jpg")

# --- GroundingDINO box on seed frame ---
dino = load_dino()
seed_rgb = np.array(Image.open(frames[args.seed_frame]).convert("RGB"))
H, W = seed_rgb.shape[:2]
_, img_t = load_image_from_array(seed_rgb)
boxes, logits, phrases = predict(model=dino, image=img_t, caption=PROMPT,
                                 box_threshold=BOX_T, text_threshold=TEXT_T, device=DEVICE)
print(f"\nGroundingDINO('{PROMPT}') on seed frame -> {boxes.shape[0]} boxes")
for b, l, p in zip(boxes, logits, phrases):
    print(f"   box(cxcywh norm)={np.round(b.numpy(),3)} conf={float(l):.3f} phrase='{p}'")
if boxes.shape[0] == 0:
    print("NO BOX from DINO on seed frame — try a different --seed_frame"); sys.exit(1)
best = int(torch.argmax(logits))                       # highest-confidence mouse box
box = boxes[best]; conf = float(logits[best])
xyxy = (box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0)) * torch.tensor([W, H, W, H])).to(DEVICE)
print(f"selected box conf={conf:.3f}  xyxy={np.round(xyxy.cpu().numpy()[0],1)}")

# --- SAM vit_h: box -> precise mask on seed frame ---
sam = sam_model_registry["vit_h"](checkpoint=f"{ROOT}/sam_vit_h_4b8939.pth").to(DEVICE)
sp = SamPredictor(sam); sp.set_image(seed_rgb)
tb = sp.transform.apply_boxes_torch(xyxy, seed_rgb.shape[:2]).to(DEVICE)
m, _, _ = sp.predict_torch(point_coords=None, point_labels=None, boxes=tb, multimask_output=False)
seed_mask = m[0][0].cpu().numpy().astype(bool)
print(f"SAM seed mask: {int(seed_mask.sum())} px ({100*seed_mask.mean():.2f}% of frame)")
del dino, sam, sp; torch.cuda.empty_cache()

# --- SAM2 video propagate ---
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
predictor = build_sam2_video_predictor("sam2_hiera_l.yaml",
            f"{ROOT}/segment-anything-2/checkpoints/sam2_hiera_large.pt", device=DEVICE)
state = predictor.init_state(video_path=VID_DIR)
predictor.reset_state(state)
predictor.add_new_mask(inference_state=state, frame_idx=args.seed_frame, obj_id=1, mask=seed_mask)
seg = {}
# forward from seed
for fi, oids, logits_o in predictor.propagate_in_video(state):
    seg[fi] = (logits_o[0] > 0.0).cpu().numpy()[0]
# reverse (cover frames before seed if seed_frame>0)
if args.seed_frame > 0:
    for fi, oids, logits_o in predictor.propagate_in_video(state, reverse=True):
        seg[fi] = (logits_o[0] > 0.0).cpu().numpy()[0]
print(f"\npropagated to {len(seg)} / {N} frames")

# --- save binary masks at undistorted resolution, named to original stems ---
os.makedirs(OUT_DIR, exist_ok=True)
counts = []
for i in range(N):
    msk = seg.get(i, np.zeros((H, W), bool))
    if msk.shape != (H, W):
        msk = np.array(Image.fromarray(msk.astype(np.uint8)*255).resize((W, H))) > 127
    Image.fromarray((msk.astype(np.uint8)), "L").save(f"{OUT_DIR}/{stems[i]}.png")  # values {0,1}
    counts.append(int(msk.sum()))
counts = np.array(counts)
print(f"saved {N} masks -> {OUT_DIR}")
print(f"mouse-px: median={np.median(counts):.0f} min={counts.min()} max={counts.max()}  "
      f"keep_frac median={100*np.median(counts)/(H*W):.3f}%")
np.save("/tmp/dinosam2_counts.npy", dict(counts=counts, stems=stems, seed=args.seed_frame, conf=conf), allow_pickle=True)
