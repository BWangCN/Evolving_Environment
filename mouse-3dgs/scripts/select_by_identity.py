"""STEP 2: select each object by identity from the IMG_0365 multi-class model. label1->mouse,
label2->apple. Two separate files; fresh names; nothing overwritten."""
import os, numpy as np, torch
from plyfile import PlyData, PlyElement

MODEL="output/mouse_apple_0365/point_cloud/iteration_30000"
d=PlyData.read(f"{MODEL}/point_cloud.ply")["vertex"].data; N=len(d)
obj=np.stack([d[f"obj_dc_{i}"] for i in range(16)],1).astype(np.float32)
clf=torch.load(f"{MODEL}/classifier.pth",map_location="cpu")
W=clf["weight"].squeeze(-1).squeeze(-1).numpy(); b=clf["bias"].numpy()
ids=(obj@W.T+b).argmax(1)
print(f"total scene gaussians: {N}")

def save(label,name):
    keep=ids==label; kept=d[keep]
    full=f"output/mouse_apple_0365/{name}.ply"
    new=np.empty(len(kept),dtype=[(n,d.dtype[n]) for n in d.dtype.names])
    for n in d.dtype.names: new[n]=kept[n]
    PlyData([PlyElement.describe(new,"vertex")],text=False).write(full)
    std=[n for n in d.dtype.names if n and not n.startswith("obj_dc")]
    nv=np.empty(len(kept),dtype=[(n,d.dtype[n]) for n in std])
    for n in std: nv[n]=kept[n]
    PlyData([PlyElement.describe(nv,"vertex")],text=False).write(full.replace(".ply","_viewer.ply"))
    print(f"  label {label} -> {name}.ply : {int(keep.sum())} gaussians ({100*keep.sum()/N:.2f}% of scene)")
    return int(keep.sum())

nm=save(1,"mouse_0365_segmented")
na=save(2,"apple_0365_segmented")
print(f"\nmouse={nm}  apple={na}  background={int((ids==0).sum())}  (sum={nm+na+int((ids==0).sum())} == {N})")
np.save("/tmp/ids_0365.npy", ids)
