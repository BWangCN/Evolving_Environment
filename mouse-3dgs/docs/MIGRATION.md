# Migration Runbook — Push to GitHub + Move to the New Machine

Two separate transfers, because they go different ways:

1. **Code + docs → GitHub** (this repo; small, version-controlled).
2. **Data → new machine directly** (multi-GB `.ply`s, captures, weights; NOT in git —
   moved out-of-band over the network).

Do them in that order. Everything below is copy-paste; replace the ALL-CAPS placeholders.

---

## Part 1 — Push code + docs to GitHub (run ON THE WORKSTATION)

### 1a. Drop these files into your project

This bundle (`scripts/`, `docs/`, `README.md`, `.gitignore`, `requirements.txt`) goes at the
root of your GG mouse-pipeline project on the workstation. If the project is already the repo
root, copy the files in alongside your existing code.

### 1b. Initialize git (skip if already a repo)

```bash
cd /home/ravi/PATH_TO_MOUSE_PROJECT     # the GG pipeline dir, NOT RoboSplat
git init
git branch -M main
```

### 1c. The .gitignore does the heavy lifting

The included `.gitignore` excludes everything large/regenerable — `.ply`, captures, weights,
`output/`, COLMAP `sparse/`, env dirs. **Verify nothing huge is staged before committing:**

```bash
git add .
git status                               # scan the list — should be scripts/docs/config only
# Hard check that no big binaries slipped through:
git diff --cached --name-only | xargs -I{} du -h "{}" 2>/dev/null | sort -h | tail -20
```

If anything multi-MB shows up (a stray `.ply`, a capture, a weight file), add it to
`.gitignore` and `git rm --cached <file>` before continuing. This is the step that keeps the
repo clean and cloneable.

### 1d. First commit

```bash
git commit -m "Mouse 3DGS pipeline: scripts, from-scratch README, full project history"
```

### 1e. Create the GitHub repo and push

Create an **empty** repo on GitHub first (no README/…gitignore — you already have them), then:

```bash
# If the workstation has no SSH key registered with GitHub yet:
ssh-keygen -t ed25519 -C "robotixx-workstation"     # accept defaults
cat ~/.ssh/id_ed25519.pub                            # add this at github.com → Settings → SSH keys

git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

> HTTPS alternative (if you prefer a token over SSH): `git remote add origin
> https://github.com/YOUR_USERNAME/YOUR_REPO.git` and authenticate with a fine-grained PAT.

---

## Part 2 — Clone onto the NEW machine

```bash
# On the NEW machine — it needs its OWN key registered with GitHub (keys aren't shared):
ssh-keygen -t ed25519 -C "new-machine"
cat ~/.ssh/id_ed25519.pub                            # add at github.com → Settings → SSH keys

cd ~                                                 # or wherever code should live
git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

Now follow `README.md` on the new machine to build the environment from scratch. **Do the
environment build BEFORE moving data** — no point shipping GBs to a box that can't yet open
them, and the README's smoke test uses only a tiny sample.

For the **full video→object pipeline** (not just the GG half), also re-clone the upstream repos
per `docs/PIPELINE.md`: `gaussian-grouping` (lkeab, `--recursive`) and `SeeDo` (ai4ce, for the
DINO+SAM2 masking front-half, in its own `seedo` env), plus a COLMAP install. The scripts in
`scripts/` run from inside those checkouts.

---

## Part 3 — Move the DATA (workstation → new machine, out-of-band)

The data is NOT in git. Move it directly between machines. Pick whichever fits your network.

**What to move (adjust to your actual layout):**
- `output/mouse_gg_0323/` — the trained GG model + the solid mouse selections.
- `output/mouse_apple_0365/` — the two-object scene, **including `desk_plane.npy`** (the
  persisted plane every move reuses — don't lose it).
- The raw captures / undistorted frames for any scene you'll re-run.
- **Model weights** (all gitignored, moved or re-downloaded out-of-band): `big-lama` / LaMa
  (inpaint), `sam_vit_h_4b8939.pth` (SAM ViT-H, masking), GroundingDINO weights
  (`groundingdino_swint_ogc.pth`) and SAM2 checkpoints for the `seedo` masking env. All are
  re-downloadable from their upstream releases if shipping them is inconvenient.

### Option A — `rsync` over SSH (best: resumable, only copies what's missing)

```bash
# Run on the WORKSTATION, pushing to the new machine.
# -a preserve everything, -v verbose, -z compress, -P show progress + resume partials.
rsync -avzP /home/ravi/PATH_TO_MOUSE_PROJECT/output/mouse_gg_0323 \
      USER@NEW_MACHINE_IP:~/YOUR_REPO/output/

rsync -avzP /home/ravi/PATH_TO_MOUSE_PROJECT/output/mouse_apple_0365 \
      USER@NEW_MACHINE_IP:~/YOUR_REPO/output/
```

If a transfer drops, just re-run the same command — `-P` resumes where it left off.

### Option B — over the NAS (if both machines mount it)

The workstation already mounts NAS storage. If the new machine is on the same lab network,
mounting the same NFS export is the fastest path — the data may not need to "move" at all,
just be referenced (or copied off the NAS locally on the new box).

```bash
# Example — copy from a shared NFS mount on the new machine:
cp -r /mnt/datasets/PROJECT/output/mouse_gg_0323 ~/YOUR_REPO/output/
cp -r /mnt/datasets/PROJECT/output/mouse_apple_0365 ~/YOUR_REPO/output/
```

### Option C — tarball + scp (simple one-shot, good for a single archive)

```bash
# On the WORKSTATION — bundle just what's needed (skip re-compressing .ply much; they're already binary):
tar -cf mouse_data.tar \
    -C /home/ravi/PATH_TO_MOUSE_PROJECT \
    output/mouse_gg_0323 output/mouse_apple_0365

scp mouse_data.tar USER@NEW_MACHINE_IP:~/
# On the NEW machine:
mkdir -p ~/YOUR_REPO/output && tar -xf ~/mouse_data.tar -C ~/YOUR_REPO/
```

### Verify the move landed intact

```bash
# On the NEW machine — the plane file MUST be present and non-empty:
ls -la ~/YOUR_REPO/output/mouse_apple_0365/desk_plane.npy
# Spot-check a .ply opens (Gaussian count sane):
python -c "from plyfile import PlyData; d=PlyData.read('OUTPUT/…/point_cloud.ply'); print(len(d['vertex']))"
```

---

## Order of operations, condensed

1. Workstation: drop bundle → `git init` → verify `.gitignore` caught the big files → commit → push.
2. New machine: clone → build env from README → run the tiny smoke test.
3. Workstation→new: `rsync` the `output/` dirs (esp. `desk_plane.npy`) once the env works.
4. New machine: verify the plane + a `.ply` open, then resume the Phase 7 move/inpaint work.
