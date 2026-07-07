"""STEP 3 verify multi-object masks: per-object stability/drift, NO-OVERLAP, swap detection,
overlays (mouse=red, apple=green) incl. closest-objects frame, base/contact tightness."""
import numpy as np, glob, os, argparse
from PIL import Image
from scipy import ndimage
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_ap=argparse.ArgumentParser()
_ap.add_argument("--img",default="/home/ravi/Desktop/Ravi/3dgs/data/IMG_0365/input")
_ap.add_argument("--msk",default="/home/ravi/Desktop/Ravi/3dgs/data/IMG_0365/object_mask_dinosam2")
_ap.add_argument("--out",default="/home/ravi/Desktop/Ravi/3dgs/data/IMG_0365")
_a=_ap.parse_args(); IMG=_a.img; MSK=_a.msk; OUTBASE=_a.out
stems=[os.path.basename(f).split('.')[0] for f in sorted(glob.glob(f"{IMG}/*.jpg"))]
H,W=np.array(Image.open(f"{MSK}/{stems[0]}.png")).shape
diag=np.hypot(H,W); N=len(stems)
LAB={1:"mouse",2:"apple"}

kf={1:[],2:[]}; cent={1:[],2:[]}; overlap_total=0; bad_vals=0
for s in stems:
    m=np.array(Image.open(f"{MSK}/{s}.png"))
    if not set(np.unique(m)).issubset({0,1,2}): bad_vals+=1
    for oid in (1,2):
        o=m==oid; kf[oid].append(100*o.mean())
        ys,xs=np.where(o); cent[oid].append((xs.mean(),ys.mean()) if o.any() else (np.nan,np.nan))
    overlap_total+=int(((m==1)&(m==2)).sum())   # impossible for integer mask -> sanity
for oid in (1,2): kf[oid]=np.array(kf[oid]); cent[oid]=np.array(cent[oid])

print(f"frames={N}  masks={len(glob.glob(MSK+'/*.png'))}  mask values subset of {{0,1,2}}: {bad_vals==0}")
print(f"NO-OVERLAP (integer mask, pixels labeled both): {overlap_total} (must be 0)")
print()
for oid in (1,2):
    c=kf[oid]; print(f"{LAB[oid]:6s} keep_frac%: median={np.median(c):.3f} min={c.min():.3f} max={c.max():.3f} std={c.std():.3f}")

# drift/vanish/swap detection
print("\ndrift / vanish / swap flags:")
flags=[]
for oid in (1,2):
    cj=np.linalg.norm(np.diff(cent[oid],axis=0),axis=1)
    for i in range(N-1):
        r=[]
        if kf[oid][i+1]<0.3: r.append("near-empty")
        if cj[i]>0.10*diag: r.append(f"centroid jump {cj[i]:.0f}px")
        if r: flags.append((stems[i+1],LAB[oid],r))
# swap: both centroids jump at same frame AND they cross
swap_frames=[]
for i in range(N-1):
    j1=np.linalg.norm(cent[1][i+1]-cent[1][i]); j2=np.linalg.norm(cent[2][i+1]-cent[2][i])
    cross=np.linalg.norm(cent[1][i+1]-cent[2][i])+np.linalg.norm(cent[2][i+1]-cent[1][i])
    keep=np.linalg.norm(cent[1][i+1]-cent[1][i])+np.linalg.norm(cent[2][i+1]-cent[2][i])
    if j1>0.08*diag and j2>0.08*diag and cross<keep: swap_frames.append(stems[i+1])
for s,l,r in flags: print(f"  {s} [{l}]: {', '.join(r)}")
print(f"  -> {len(flags)} per-object flags; suspected SWAP frames: {swap_frames if swap_frames else 'NONE'}")

# closest-objects frame (min centroid distance) for occlusion overlay
dist=np.linalg.norm(cent[1]-cent[2],axis=1)
closest=stems[int(np.nanargmin(dist))]
print(f"\nclosest mouse-apple frame: {closest} (centroid dist {np.nanmin(dist):.0f}px)")

# overlays: mouse=red, apple=green
show=["0001",closest,"0066","0044","0109","0130"]
seen=set(); show=[s for s in show if s in stems and not (s in seen or seen.add(s))][:6]
fig,ax=plt.subplots(2,3,figsize=(15,11))
for k,s in enumerate(show):
    img=np.array(Image.open(f"{IMG}/{s}.jpg").convert("RGB")); m=np.array(Image.open(f"{MSK}/{s}.png"))
    ov=img.copy(); ov[m==1]=(0.5*ov[m==1]+np.array([127,0,0])).astype(np.uint8); ov[m==2]=(0.5*ov[m==2]+np.array([0,110,0])).astype(np.uint8)
    a=ax[k//3,k%3]; a.imshow(ov); a.set_title(f"{s}{' [closest]' if s==closest else ''}  mouse={100*(m==1).mean():.2f}% apple={100*(m==2).mean():.2f}%",fontsize=9); a.axis("off")
fig.suptitle("IMG_0365 multi-object masks — mouse=RED, apple=GREEN (tight at base? no cross-bleed?)")
fig.tight_layout(); fig.savefig(f"{OUTBASE}/multi_verify.png",dpi=72); plt.close(fig)
print("wrote data/IMG_0365/multi_verify.png")
# stability plot
fig,a2=plt.subplots(1,1,figsize=(12,3))
a2.plot(kf[1],label='mouse'); a2.plot(kf[2],label='apple'); a2.set_title("per-object keep_frac per frame"); a2.legend(); a2.set_xlabel("frame")
fig.tight_layout(); fig.savefig(f"{OUTBASE}/multi_stability.png",dpi=80); plt.close(fig)
print("wrote data/IMG_0365/multi_stability.png")
