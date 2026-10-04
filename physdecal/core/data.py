"""
data.py
Turns the captured GeoPatchCity folder into batches a patch optimiser can use.

Three things have to line up per instance and this module is the only place
they are joined:

  1. the RGB frame                    images/<frame_id>.png
  2. the ground truth vehicle mask    segmentation/<frame_id>.png + palette.json
  3. the projected roof quad          geometry/<frame_id>.json

Why the mask comes from the INSTANCE segmentation and not from the semantic
export: the semantic export is broken in this capture (paint order fault, zero
semantic pixels reached the saved frames). The per-vehicle instance colours in
palette.json are fine, and the union of them is exactly the vehicle-class
ground truth this experiment needs. So nothing has to be re-captured.

Geometry objects are matched to frames.csv rows by their 2D box, because the
geometry JSON stores a class name but not the actor name. If that match ever
fails for more than a handful of rows, something has changed in the exporter
and build_index() says so loudly instead of silently dropping instances.
"""

import csv
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from physdecal import config as N


ROOF_CORNER_IDX = [4, 5, 6, 7]     # gp_camera.oriented_box corner order:
                                   # front-left, front-right, rear-right, rear-left
FLOOR_CORNER_IDX = [0, 1, 2, 3]    # the same rectangle at ground height.
                                   # Needed so a decal on a LOWER panel -- a
                                   # bonnet or a boot lid -- can be placed on
                                   # its own plane instead of floating at roof
                                   # height. At nadir that error is under two
                                   # percent of scale, but at 60 degrees off
                                   # nadir a 0.33 m height error displaces the
                                   # decal by 0.57 m on the ground, which is
                                   # half a panel.


# --------------------------------------------------------------------- index

def _load_palette():
    if not os.path.isfile(N.PALETTE_JSON):
        raise SystemExit("No %s. The instance colours are the ground truth, so "
                         "this experiment cannot run without it." % N.PALETTE_JSON)
    pal = json.load(open(N.PALETTE_JSON))
    return {k: tuple(int(v) for v in bgr)
            for k, bgr in pal.get("colours_bgr", {}).items()}


def _rows_from(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _bbox_of(row):
    return (int(row["bbox_x0"]), int(row["bbox_y0"]),
            int(row["bbox_x1"]), int(row["bbox_y1"]))


def _match_object(geom_objects, row):
    """Find the geometry object whose 2D box matches this CSV row."""
    want = _bbox_of(row)
    best, best_d = None, 1e9
    for o in geom_objects:
        b = tuple(int(v) for v in o["bbox_2d"])
        d = sum(abs(a - c) for a, c in zip(b, want))
        if d < best_d:
            best, best_d = o, d
    return best if best_d <= 4 else None


def _build_index_uncached(split, nadir_only, verbose):
    """
    Returns a list of instance dicts. One dict is one vehicle in one frame.

    Keys: frame_id, target, cls, colour_bgr, roof_quad (4x2 float),
          floor_quad (4x2 float, the same rectangle at ground height),
          bbox, altitude_m, theta_deg, phi_deg, lighting, gsd_m_per_px,
          visible_px, image_path, seg_path
    """
    path = {"train": N.SPLIT_TRAIN, "holdout": N.SPLIT_HOLDOUT,
            "all": N.META_CSV}[split]
    if not os.path.isfile(path):
        raise SystemExit(
            "No %s.\nRun make_splits.py from the GeoPatchCity capture "
            "pipeline before this "
            "experiment. It holds out whole VEHICLES, which is the only "
            "split that does not leak here: two nadir frames of the same "
            "parked car are near duplicates." % path)

    palette = _load_palette()
    rows = _rows_from(path)
    by_frame = {}
    for r in rows:
        by_frame.setdefault(r["frame_id"], []).append(r)

    out = []
    n_unmatched = n_small = n_ignored = n_noquad = 0
    for fid, frows in by_frame.items():
        gpath = os.path.join(N.GEOM_DIR, fid + ".json")
        if not os.path.isfile(gpath):
            continue
        geom = json.load(open(gpath))
        objs = geom.get("objects", [])
        for r in frows:
            if r.get("label_status") == "ignored":
                n_ignored += 1
                continue
            if N.PRIMARY_ONLY and r.get("is_primary") != "1":
                continue
            theta = float(r["theta_deg"])
            if nadir_only and theta > N.NADIR_MAX_THETA_DEG:
                continue
            if int(r["visible_px"]) < N.MIN_INSTANCE_PX:
                n_small += 1
                continue
            o = _match_object(objs, r)
            if o is None:
                n_unmatched += 1
                continue
            corners = np.asarray(o["corners_image"], dtype=np.float32)
            quad = corners[ROOF_CORNER_IDX]
            floor = corners[FLOOR_CORNER_IDX]
            if cv2.contourArea(quad.astype(np.float32)) < N.MIN_ROOF_QUAD_PX:
                n_noquad += 1
                continue
            colour = palette.get(r["target"])
            if colour is None:
                continue
            out.append({
                "frame_id": fid,
                "target": r["target"],
                "cls": r["class"],
                "colour_bgr": colour,
                "roof_quad": quad,
                "floor_quad": floor,
                "bbox": _bbox_of(r),
                "altitude_m": float(r["altitude_m"]),
                "theta_deg": theta,
                "phi_deg": float(r.get("phi_deg", 0.0) or 0.0),
                "lighting": r.get("lighting", ""),
                "gsd_m_per_px": float(r["gsd_across_m_per_px"]),
                "visible_px": int(r["visible_px"]),
                "box_len_m": float(r.get("box_len_m", 0.0) or 0.0),
                "box_hgt_m": float(r.get("box_hgt_m", 0.0) or 0.0),
                "image_path": os.path.join(N.IMAGES_DIR, fid + ".png"),
                "seg_path": os.path.join(N.SEG_DIR, fid + ".png"),
            })

    if verbose:
        print("index[%s%s]: %d instances from %d frames"
              % (split, ", nadir only" if nadir_only else "",
                 len(out), len(set(i["frame_id"] for i in out))))
        print("  dropped: %d ignored, %d under %d px, %d roof quad too small, "
              "%d unmatched to geometry"
              % (n_ignored, n_small, N.MIN_INSTANCE_PX, n_noquad, n_unmatched))
        if n_unmatched > 0.02 * max(len(out), 1):
            print("  >>> More than 2 percent of rows did not match a geometry "
                  "object.\n      The exporter and frames.csv have drifted "
                  "apart. Re-run verify_dataset.py (capture pipeline)\n      before trusting "
                  "anything downstream.")
    return out


def _cache_path(split, nadir_only):
    """One cache file per (split, nadir cut). The cut is in the name because
    changing NADIR_MAX_THETA_DEG changes the contents."""
    return os.path.join(N.CACHE_DIR, "index_%s_%s.pkl"
                        % (split, ("nadir%g" % N.NADIR_MAX_THETA_DEG)
                           if nadir_only else "all"))


def _cache_key(split):
    """Everything that changes what build_index returns. If any of it moves,
    the cache is stale and is rebuilt rather than silently reused."""
    src = {"train": N.SPLIT_TRAIN, "holdout": N.SPLIT_HOLDOUT,
           "all": N.META_CSV}[split]
    return {
        "split_mtime": os.path.getmtime(src) if os.path.isfile(src) else 0,
        "palette_mtime": (os.path.getmtime(N.PALETTE_JSON)
                          if os.path.isfile(N.PALETTE_JSON) else 0),
        "primary_only": N.PRIMARY_ONLY,
        "min_instance_px": N.MIN_INSTANCE_PX,
        "min_roof_quad_px": N.MIN_ROOF_QUAD_PX,
        "has_floor_quad": True,          # bumps the cache when floor_quad was
                                         # added, so a stale pickle without it
                                         # is rebuilt rather than crashing
                                         # multi-panel placement
        "nadir_max_theta": N.NADIR_MAX_THETA_DEG,
        "geom_dir": N.GEOM_DIR,
    }


def build_index(split="train", nadir_only=False, verbose=True, cache=True):
    """
    Cached wrapper around the index build.

    Building the index opens one geometry JSON per frame, which is tens of
    thousands of small reads and takes minutes. Every script in the experiment
    needs the same index, so it is built once and pickled under CACHE_DIR.

    The cache is keyed on the mtime of the split CSV and on every config value
    that changes the result. Any mismatch rebuilds. Delete CACHE_DIR, or pass
    cache=False, to force a rebuild by hand.
    """
    if not cache:
        return _build_index_uncached(split, nadir_only, verbose)

    os.makedirs(N.CACHE_DIR, exist_ok=True)
    path = _cache_path(split, nadir_only)
    key = _cache_key(split)
    if os.path.isfile(path):
        try:
            with open(path, "rb") as fh:
                blob = pickle.load(fh)
            if blob.get("key") == key:
                if verbose:
                    print("index[%s%s]: %d instances (cached, %s)"
                          % (split, ", nadir only" if nadir_only else "",
                             len(blob["index"]), os.path.basename(path)))
                return blob["index"]
            if verbose:
                print("index cache stale (%s), rebuilding"
                      % os.path.basename(path))
        except Exception as e:                    # corrupt or old pickle
            if verbose:
                print("index cache unreadable (%s), rebuilding" % e)

    index = _build_index_uncached(split, nadir_only, verbose)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"key": key, "index": index}, fh, protocol=4)
    os.replace(tmp, path)                          # atomic, so a killed run
    return index                                   # never leaves a half file


# ------------------------------------------------------------------- dataset

def vehicle_mask(seg_bgr, colours):
    """Union of the given instance colours. uint8 {0,1}, HxW."""
    m = np.zeros(seg_bgr.shape[:2], dtype=bool)
    for c in colours:
        m |= np.all(seg_bgr == np.array(c, np.uint8), axis=-1)
    return m.astype(np.uint8)


class NadirPatchDataset(Dataset):
    """
    One item is one vehicle instance.

    Returns a dict of tensors:
        image      3 x H x W  float 0..1, RGB
        gt         1 x H x W  float {0,1}, ALL vehicles in the frame
        inst       1 x H x W  float {0,1}, THIS vehicle only
        quad       4 x 2      float, roof corners in image pixels
        meta       dict of python scalars (collated separately)

    The patch is composited by patch.py at full capture resolution and the
    result is resized once, so the patch goes through exactly the same
    resampling the vehicle does. Resizing first and pasting after would give
    the patch a sharpness no real camera would deliver.
    """

    def __init__(self, index, all_frame_targets=None, max_items=0):
        self.index = index if not max_items else index[:max_items]
        # For the "all vehicles in this frame" ground truth we need every
        # instance colour present, including ones filtered out of the index.
        self.frame_colours = all_frame_targets or self._frame_colours(index)

    @staticmethod
    def _frame_colours(index):
        d = {}
        for it in index:
            d.setdefault(it["frame_id"], set()).add(it["colour_bgr"])
        return {k: sorted(v) for k, v in d.items()}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        it = self.index[i]
        bgr = cv2.imread(it["image_path"], cv2.IMREAD_COLOR)
        seg = cv2.imread(it["seg_path"], cv2.IMREAD_COLOR)
        if bgr is None or seg is None:
            raise RuntimeError("missing frame %s" % it["frame_id"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        gt = vehicle_mask(seg, self.frame_colours.get(it["frame_id"],
                                                      [it["colour_bgr"]]))
        inst = vehicle_mask(seg, [it["colour_bgr"]])

        meta = {k: it[k] for k in
                ("frame_id", "target", "cls", "altitude_m", "theta_deg",
                 "lighting", "gsd_m_per_px", "visible_px")}
        meta["box_len_m"] = it.get("box_len_m", 0.0)
        meta["box_hgt_m"] = it.get("box_hgt_m", 0.0)
        # The box rides with the batch rather than being looked up by
        # position in the index. Anything that consumes a loader can then
        # crop or window without assuming the loader hands items back in
        # index order, which stops being true the moment shuffle is on.
        meta["bbox"] = tuple(int(v) for v in it["bbox"])
        return {
            "image": torch.from_numpy(rgb).permute(2, 0, 1),
            "gt": torch.from_numpy(gt).float().unsqueeze(0),
            "inst": torch.from_numpy(inst).float().unsqueeze(0),
            "quad": torch.from_numpy(it["roof_quad"].copy()).float(),
            "floor": torch.from_numpy(it["floor_quad"].copy()).float(),
            "meta": meta,
        }


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch])
           for k in ("image", "gt", "inst", "quad", "floor")}
    out["meta"] = [b["meta"] for b in batch]
    return out


def make_loader(index, batch_size, shuffle, workers=2, max_items=0):
    ds = NadirPatchDataset(index, max_items=max_items)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        collate_fn=collate, drop_last=shuffle, pin_memory=True)


class _Tee:
    """Mirror a stream to a file. Used to keep a transcript of a run."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def start_log(name):
    """
    Tee stdout into LOG_DIR/<name>_<timestamp>.log and return the path.

    A run that printed a warning three hours in and then scrolled away is a
    run you cannot write up. The transcript is small and it is the only record
    of which warnings fired.
    """
    os.makedirs(N.LOG_DIR, exist_ok=True)
    path = os.path.join(N.LOG_DIR, "%s_%s.log"
                        % (name, time.strftime("%Y%m%d_%H%M%S")))
    fh = open(path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print("log: %s" % path)
    return path


def seed_everything(seed=None):
    seed = N.SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
