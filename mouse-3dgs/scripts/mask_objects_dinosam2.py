"""Multi-object DINO+SAM2 masking for IMG_0365 (mouse + apple). SeeDo technique extended to 2 objects:
GroundingDINO('mouse') + GroundingDINO('apple') boxes on a seed frame -> SAM vit_h precise masks ->
SAM2 video predictor with TWO obj_ids (1=mouse,2=apple) propagated in ONE pass. Output: integer-labeled
mask per frame (bg=0, mouse=1, apple=2) -> object_mask_dinosam2/. Overlaps resolved by SAM2 logit so
the two objects are mutually exclusive (separable). Raw overlap reported for the gate."""
import os, sys, glob, shutil
import numpy as np, torch
from PIL import Image
from groundingdino.util.inference import load_image_from_array, predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.models import build_model
from groundingdino.util import box_ops
from segment_anything import sam_model_registry, SamPredictor
from sam2.build_sam import build_sam2_video_predictor
from huggingface_hub import hf_hub_download

import argparse
_ap=argparse.ArgumentParser()
_ap.add_argument("--base",default="/home/ravi/Desktop/Ravi/3dgs/data/IMG_0365")  # capture dir with input/
_bargs,_=_ap.parse_known_args()
ROOT="/home/ravi/Desktop/Ravi/SeeDo"
IMG_DIR=f"{_bargs.base}/input"                                 # RAW sharp frames (undistort masks later)
OUT_DIR=f"{_bargs.base}/object_mask_dinosam2"
VID_DIR=f"{_bargs.base}/_sam2_frames"
PROMPTS={1:"mouse", 2:"apple"}                                  # obj_id -> text prompt / label
BOX_T=0.3; TEXT_T=0.25; DEVICE="cuda"

def load_dino():
    repo="ShilongLiu/GroundingDINO"
    cfg=hf_hub_download(repo_id=repo,filename="GroundingDINO_SwinB.cfg.py")
    ck=hf_hub_download(repo_id=repo,filename="groundingdino_swinb_cogcoor.pth")
    a=SLConfig.fromfile(cfg); a.device=DEVICE
    m=build_model(a); m.load_state_dict(clean_state_dict(torch.load(ck,map_location="cpu")["model"]),strict=False)
    return m.eval().to(DEVICE)

frames=sorted(glob.glob(f"{IMG_DIR}/*.jpg"))
stems=[os.path.basename(f).split(".")[0] for f in frames]; N=len(frames)
print(f"frames: {N}  res {Image.open(frames[0]).size}")
if os.path.isdir(VID_DIR): shutil.rmtree(VID_DIR)
os.makedirs(VID_DIR)
for i,f in enumerate(frames): os.symlink(f,f"{VID_DIR}/{i:05d}.jpg")

dino=load_dino()
def dino_box(rgb,prompt):
    _,t=load_image_from_array(rgb)
    b,l,p=predict(model=dino,image=t,caption=prompt,box_threshold=BOX_T,text_threshold=TEXT_T,device=DEVICE)
    if b.shape[0]==0: return None,0.0
    j=int(torch.argmax(l)); return b[j],float(l[j])

# seed search: pick frame where BOTH objects detected with best min-confidence
cand_seeds=[0,N//6,N//3,N//2,2*N//3,5*N//6]
best_seed=None; best_score=-1; best_boxes=None
for s in cand_seeds:
    rgb=np.array(Image.open(frames[s]).convert("RGB"))
    bm,cm=dino_box(rgb,PROMPTS[1]); ba,ca=dino_box(rgb,PROMPTS[2])
    score=min(cm,ca) if (bm is not None and ba is not None) else -1
    print(f"  seed candidate {stems[s]}: mouse conf={cm:.3f} apple conf={ca:.3f} -> min={score:.3f}")
    if score>best_score: best_score=score; best_seed=s; best_boxes=(bm,cm,ba,ca)
assert best_boxes[0] is not None and best_boxes[2] is not None, "DINO failed to find both objects in seed candidates"
bm,cm,ba,ca=best_boxes
print(f"\nSEED frame = {stems[best_seed]} (idx {best_seed})  mouse conf={cm:.3f}  apple conf={ca:.3f}")

# SAM vit_h: boxes -> precise seed masks
seed_rgb=np.array(Image.open(frames[best_seed]).convert("RGB")); H,W=seed_rgb.shape[:2]
sam=sam_model_registry["vit_h"](checkpoint=f"{ROOT}/sam_vit_h_4b8939.pth").to(DEVICE); sp=SamPredictor(sam); sp.set_image(seed_rgb)
seed_masks={}
for oid,box,conf in [(1,bm,cm),(2,ba,ca)]:
    xyxy=(box_ops.box_cxcywh_to_xyxy(box.unsqueeze(0))*torch.tensor([W,H,W,H])).to(DEVICE)
    tb=sp.transform.apply_boxes_torch(xyxy,seed_rgb.shape[:2]).to(DEVICE)
    m,_,_=sp.predict_torch(point_coords=None,point_labels=None,boxes=tb,multimask_output=False)
    seed_masks[oid]=m[0][0].cpu().numpy().astype(bool)
    print(f"  SAM seed mask obj{oid}({PROMPTS[oid]}): {int(seed_masks[oid].sum())} px")
# resolve seed-mask overlap (mouse vs apple) by SAM score is N/A here -> assign overlap to neither's expansion; keep raw for now
overlap_seed=int((seed_masks[1]&seed_masks[2]).sum())
print(f"  seed raw overlap (mouse&apple): {overlap_seed} px")
del dino,sam,sp; torch.cuda.empty_cache()

# SAM2 multi-object propagate
torch.autocast("cuda",dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major>=8:
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
pred=build_sam2_video_predictor("sam2_hiera_l.yaml",f"{ROOT}/segment-anything-2/checkpoints/sam2_hiera_large.pt",device=DEVICE)
state=pred.init_state(video_path=VID_DIR); pred.reset_state(state)
for oid in (1,2):
    pred.add_new_mask(inference_state=state,frame_idx=best_seed,obj_id=oid,mask=seed_masks[oid])
logits_by_frame={}
for fi,oids,lg in pred.propagate_in_video(state):
    logits_by_frame[fi]={oid:lg[i][0].float().cpu().numpy() for i,oid in enumerate(oids)}
if best_seed>0:
    for fi,oids,lg in pred.propagate_in_video(state,reverse=True):
        logits_by_frame[fi]={oid:lg[i][0].float().cpu().numpy() for i,oid in enumerate(oids)}
print(f"\npropagated to {len(logits_by_frame)} / {N} frames")

# build integer masks; resolve overlap by higher logit -> mutually exclusive
os.makedirs(OUT_DIR,exist_ok=True)
stats={1:[],2:[]}; raw_overlap=[]
for i in range(N):
    L=logits_by_frame.get(i,{})
    lm=L.get(1,np.full((H,W),-1e4)); la=L.get(2,np.full((H,W),-1e4))
    mm=lm>0; ma=la>0
    raw=int((mm&ma).sum()); raw_overlap.append(raw)
    # mutual exclusivity: contested pixels go to higher logit
    both=mm&ma
    mouse=mm&~both | (both&(lm>=la))
    apple=ma&~both | (both&(la>lm))
    intmask=np.zeros((H,W),np.uint8); intmask[mouse]=1; intmask[apple]=2
    Image.fromarray(intmask,"L").save(f"{OUT_DIR}/{stems[i]}.png")
    stats[1].append(int(mouse.sum())); stats[2].append(int(apple.sum()))
raw_overlap=np.array(raw_overlap)
print(f"saved {N} integer masks (0=bg,1=mouse,2=apple) -> {OUT_DIR}")
for oid in (1,2):
    c=np.array(stats[oid]); print(f"  {PROMPTS[oid]:6s}: keep_frac median={100*np.median(c)/(H*W):.3f}% min={100*c.min()/(H*W):.3f}% max={100*c.max()/(H*W):.3f}%")
print(f"  raw SAM2 overlap (pre-resolution): median={int(np.median(raw_overlap))} max={int(raw_overlap.max())} px (resolved to 0 in output)")
np.save("/tmp/multi365.npy",dict(stats=stats,raw_overlap=raw_overlap,stems=stems,seed=best_seed,
        conf={1:cm,2:ca},H=H,W=W),allow_pickle=True)
