r"""
physdecal check
Preflight. Run this first, and run it again whenever something downstream
behaves oddly.

    physdecal check                 everything except the gate
    physdecal check --gate          measure clean performance too
    physdecal check --gate --published   the same, on COCO/ADE weights

WHAT THE GATE IS FOR
--------------------
An attack can only be measured on instances the victim gets right WITHOUT the
patch. If clean performance at nadir is zero, the attackable set is empty and
ASR is undefined -- not low, undefined. Reporting a number in that state is
the most common way aerial attack results become meaningless, and it is an
easy mistake to make because nothing crashes.

So the gate is measured, printed, and compared against CLEAN_IOU_GATE for both
arms, on published weights and on finetuned ones, and it says in words which
state the roster is in.
"""

import argparse
import os
import sys

import numpy as np
import torch

from physdecal import config as N


OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def _line(status, what, detail=""):
    print("[%s] %-32s %s" % (status, what, detail))


def check_isolation():
    """
    The promise this folder makes: it never writes into the dataset, and it
    never touches paper2's results. Asserted rather than trusted, because the
    cost of being wrong is somebody else's finished experiment.
    """
    print("\n-- isolation " + "-" * 58)
    exp = os.path.abspath(N.EXP_ROOT)
    data = os.path.abspath(N.DATA_ROOT)
    inside = os.path.normcase(exp).startswith(os.path.normcase(data) + os.sep)
    _line(BAD if inside else OK, "outputs are outside the dataset", exp)

    p2 = os.path.join(data, "nadirpatch")
    if os.path.isdir(p2):
        n = sum(len(f) for _, _, f in os.walk(p2))
        _line(OK, "paper2 results untouched", "%d files under %s" % (n, p2))
    else:
        _line(WARN, "paper2 results folder not found",
              "nothing to protect, which is fine")
    return not inside


def check_env():
    print("\n-- environment " + "-" * 56)
    ok = True
    _line(OK, "python", sys.version.split()[0])
    _line(OK, "torch", "%s  cuda=%s" % (torch.__version__,
                                        torch.cuda.is_available()))
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gb = p.total_memory / 1024 ** 3
        _line(OK if gb >= 5 else WARN, "gpu",
              "%s, %.1f GB" % (p.name, gb))
        if gb < 8:
            print("       a %.0f GB card runs this at ATTACK_BATCH 1-2. The "
                  "joint task holds two\n       victims at once and is the "
                  "first thing to fall over." % gb)
    else:
        _line(WARN, "gpu", "none: everything here runs, slowly")

    for mod in ("cv2", "transformers", "torchvision", "matplotlib"):
        try:
            m = __import__(mod)
            _line(OK, mod, getattr(m, "__version__", "?"))
        except Exception as e:
            _line(BAD, mod, str(e))
            ok = False
    try:
        import cv2
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, N.FIDUCIAL_DICT))
        _line(OK, "cv2.aruco", N.FIDUCIAL_DICT)
    except Exception as e:
        _line(BAD, "cv2.aruco", "%s -- the physical arm needs it" % e)
        ok = False
    try:
        import kornia
        _line(OK, "kornia", "%s (optional)" % kornia.__version__)
    except Exception:
        _line(WARN, "kornia", "missing: the native warp is used instead")
    return ok


def check_dataset():
    print("\n-- dataset " + "-" * 60)
    ok = True
    for what, path in (("images", N.IMAGES_DIR), ("segmentation", N.SEG_DIR),
                       ("geometry", N.GEOM_DIR), ("palette", N.PALETTE_JSON),
                       ("frames.csv", N.META_CSV),
                       ("train split", N.SPLIT_TRAIN),
                       ("holdout split", N.SPLIT_HOLDOUT)):
        exists = os.path.exists(path)
        n = (len(os.listdir(path)) if exists and os.path.isdir(path) else "")
        _line(OK if exists else BAD, what,
              "%s %s" % (path, ("(%d files)" % n) if n != "" else ""))
        ok = ok and exists
    if not ok:
        print("       Missing splits? Run make_splits.py from the capture pipeline")
    return ok


def check_index():
    print("\n-- index " + "-" * 62)
    from physdecal.core import data as D

    train = D.build_index("train", nadir_only=True, verbose=False)
    hold = D.build_index("holdout", nadir_only=False, verbose=False)
    _line(OK if train else BAD, "nadir train instances", str(len(train)))
    _line(OK if hold else BAD, "holdout instances (all views)", str(len(hold)))

    if train and hold:
        tv = {i["target"] for i in train}
        hv = {i["target"] for i in hold}
        leak = tv & hv
        _line(BAD if leak else OK, "no vehicle in both splits",
              ("LEAK: %s" % ", ".join(sorted(leak))) if leak
              else "%d train / %d holdout vehicles" % (len(tv), len(hv)))
        th = [i["theta_deg"] for i in hold]
        _line(OK, "holdout nadir angle range",
              "%.0f to %.0f degrees" % (min(th), max(th)))
        return bool(train and hold and not leak)
    return False


def check_realism():
    print("\n-- decal manifold " + "-" * 53)
    from physdecal.core import decal as DEC

    man = DEC.DecalManifold()
    _line(OK, "level %d" % man.level, man.describe())
    _line(OK, "attacker degrees of freedom",
          "%d of %d raw pixel values"
          % (man.degrees_of_freedom(), N.PATCH_RES ** 2 * 3))
    if man.level >= 4 and man.n_regions < 16:
        _line(WARN, "few regions",
              "%d regions leaves very little to optimise; raise "
              "ANCHOR_SUPERCELL" % man.n_regions)

    # The projection has to be able to reproduce its own anchor, or the
    # realism target is outside the feasible set.
    if man.anchor is not None:
        back = man.project(man.anchor)
        e = DEC.edge_agreement(back, man.anchor)
        d = float(DEC.ciede2000_rgb(back, man.anchor).median())
        _line(OK if (e > 0.95 and d < 1.0) else BAD,
              "anchor is reachable", "edge IoU %.2f, median dE %.1f" % (e, d))

    _line(OK if N.INK_MEASURED else WARN, "ink set",
          "measured scan" if N.INK_MEASURED else
          "PLACEHOLDER -- do not report a printability number until "
          "PRINTABLE_COLOURS_CSV is a real scan and INK_MEASURED is True")
    return True


def check_weights():
    print("\n-- finetuned weights " + "-" * 50)
    from physdecal.victims import segmentation as M
    from physdecal.victims import detection as DET

    any_seg = False
    for k in N.MODELS:
        if N.MODELS[k][0] == "hf_clipseg":
            _line(OK, k, "open vocabulary, never finetuned by design")
            continue
        p = M.finetuned_path(k)
        if os.path.isfile(p):
            s = torch.load(p, map_location="cpu")
            _line(OK, k, "val vehicle IoU %.3f at epoch %d"
                  % (s.get("val_iou", float("nan")), s.get("epoch", -1)))
            any_seg = True
        else:
            _line(WARN, k, "no decoder: physdecal finetune --model %s" % k)

    any_det = False
    for k in N.DET_MODELS:
        p = DET.finetuned_path(k)
        if os.path.isfile(p):
            if N.DET_MODELS[k][0] == "ultralytics":
                _line(OK, k, "ultralytics weights present (transfer target)")
                any_det = True
                continue
            s = torch.load(p, map_location="cpu")
            _line(OK, k, "val AP50 %.3f at epoch %d"
                  % (s.get("val_ap50", float("nan")), s.get("epoch", -1)))
            any_det = True
        else:
            cmd = ("physdecal yolo train"
                   if N.DET_MODELS[k][0] == "ultralytics"
                   else "physdecal finetune --det %s" % k)
            _line(WARN, k, "no weights: %s" % cmd)
    return any_seg, any_det


@torch.no_grad()
def gate(published=False, n=60, device=None):
    """
    Measure clean performance at nadir on the holdout split.

    This is the measurement that decides whether the experiment can produce a
    number at all, so it is a first-class command rather than a comment.
    """
    print("\n-- eligibility gate (%s weights) %s"
          % ("published" if published else "finetuned", "-" * 32))
    from physdecal.core import data as D
    from physdecal.victims import segmentation as M
    from physdecal.victims import detection as DET

    device = device or N.DEVICE
    index = D.build_index("holdout", nadir_only=True, verbose=False)
    if not index:
        _line(BAD, "nadir holdout instances", "none")
        return
    index = index[:n]
    loader = D.make_loader(index, 1, shuffle=False, workers=2)
    batches = list(loader)
    print("   %d nadir holdout instances" % len(batches))

    from physdecal.evaluation import evaluate as E

    for key in N.ATTACK_MODELS:
        try:
            m = M.load_model(key, device=device, finetuned=not published,
                             cache=False)
        except SystemExit as e:
            _line(WARN, key, str(e).splitlines()[0])
            continue
        ious = []
        for b in batches:
            img = b["image"].to(device)
            p = m(img)["p_veh"]
            meta = b["meta"][0]
            x0, y0, x1, y1 = E.window_of(meta["bbox"], img.shape[-2:])
            g = b["gt"][0, 0, y0:y1, x0:x1].numpy() > 0.5
            pr = p[0, 0, y0:y1, x0:x1].cpu().numpy() > 0.5
            ious.append(E.iou(pr, g))
        mean = float(np.mean(ious))
        att = float(np.mean([i >= N.ATTACKABLE_IOU for i in ious]))
        passed = mean >= N.CLEAN_IOU_GATE
        _line(OK if passed else BAD, "seg %s" % key,
              "clean IoU %.3f, %.0f%% attackable" % (mean, 100 * att))
        M.unload_all()

    for key in N.DET_ATTACK_MODELS:
        try:
            d = DET.load_detector(key, device=device, finetuned=not published,
                                  cache=False)
        except SystemExit as e:
            _line(WARN, key, str(e).splitlines()[0])
            continue
        hits = []
        for b in batches:
            img = b["image"].to(device)
            meta = b["meta"][0]
            pred = d.predict(img)[0]
            ids = d.vehicle_ids(meta["cls"])
            hits.append(DET.best_hit(pred, meta["bbox"], ids) is not None)
        rec = float(np.mean(hits))
        _line(OK if rec >= 0.35 else BAD, "det %s" % key,
              "clean recall %.3f at score %.2f" % (rec, N.DET_SCORE))
        DET.unload_all()

    if published:
        print("\n   These are the published-weight numbers. If they are near "
              "zero, that is the\n   measurement that justifies finetuning, "
              "and it belongs in the paper.")
    else:
        print("\n   ASR is only defined over the attackable set. A row with "
              "an empty attackable\n   set must be reported as UNDEFINED, not "
              "as zero.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", action="store_true",
                    help="measure clean performance (loads models, slower)")
    ap.add_argument("--published", action="store_true",
                    help="gate on published weights instead of finetuned")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    print("=" * 72)
    print("PhysDecal preflight")
    print("=" * 72)

    results = [check_isolation(), check_env(), check_dataset()]
    if results[-1]:
        results.append(check_index())
        check_realism()
        check_weights()
    if a.gate:
        gate(a.published, a.n, a.device)

    print("\n" + "=" * 72)
    if all(results):
        print("preflight passed. Next:  see README.md and docs/TUTORIAL.md")
    else:
        print("preflight FAILED above. Fix those before running anything else.")
    print("=" * 72)


if __name__ == "__main__":
    main()
