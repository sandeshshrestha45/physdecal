r"""
physdecal physical
The print, paste and photograph loop: everything that happens after
physdecal export-print, and the only part of this study that measures the real
world.

    physdecal physical plan     --patch X --target diecast
    physdecal physical ingest   --dir out/physical/session01
    physdecal physical annotate --dir out/physical/session01
    physdecal physical score    --dir out/physical/session01
    physdecal physical panel    --dir out/physical/session01

THE PROTOCOL, AND WHY IT IS PAIRED
----------------------------------
Every measurement is a PAIR of photographs of the same vehicle from the same
pose, one with the decal and one without, taken seconds apart. The pair is the
unit, not the photograph, because everything that would otherwise confound the
comparison -- the light in the room, the angle of the phone, the exact
position of the car, the camera's auto-exposure -- is shared between the two
members of a pair and cancels.

Filenames carry the pairing, because a convention that lives in a filename
survives being copied off a phone and a convention that lives in a spreadsheet
does not:

    <scene>__<clean|patched>__<index>.jpg

scene is anything you like, as long as the two members of a pair share it.

WHAT THE FIDUCIALS BUY
----------------------
Four ArUco markers of known millimetre pitch surround the decal. From them,
every photograph yields:

    the homography from the printed sheet to the image
    the nadir angle, by decomposing that homography against the camera matrix
    the metric scale, so a photograph taken from an unrecorded distance can
        still be binned on the same axis as the simulated frames
    the exact pixel footprint of the decal, which is what separates damage
        under the patch from damage away from it

The camera matrix is estimated from the field of view, taken from EXIF when
the file has it and from --fov otherwise. THIS IS AN ASSUMPTION AND IT IS
REPORTED IN EVERY ROW. It affects the recovered ANGLE; it does not affect
whether the detector found the vehicle, which is the headline physical number.
A wrong focal length therefore mis-bins a row, it does not invent a success.

GROUND TRUTH FOR A PHOTOGRAPH
-----------------------------
Segmentation IoU needs a vehicle mask, and a photograph has none. `annotate`
draws one: you drag a box round the vehicle, GrabCut proposes a mask, and you
correct it with the mouse. It is stored next to the photograph and is used for
BOTH members of a pair, since the vehicle does not move between them.

Detection needs only a box, which the same annotation provides, so the
detection arm can be scored without touching the mask quality question at all.
That is worth knowing when time is short: `score --arm det` needs about ten
seconds of annotation per pair and produces the headline number.
"""

import argparse
import csv
import glob
import json
import os
import shutil

import cv2
import numpy as np
import torch

from physdecal import config as N


MANIFEST = "manifest.csv"
ANNOT_DIR = "annotations"


# ================================================================= 1. plan

def plan(patch_tag, target=None, out_dir=None):
    """
    Write the capture plan: one row per photograph you need to take.

    Emitted as a CSV you tick off, because a physical session with a laptop
    open is a session where half the cells are missed. The cell structure is
    the same grid the simulation is evaluated on, so the two can be put in one
    table.
    """
    from physdecal.physical import export_print as X
    from physdecal.evaluation import evaluate as E

    # The patch's OWN size, exactly as export_print.export resolves it.
    # Reading the config default instead put a decal size in the capture
    # instructions that the printed sheet does not have -- 45.6 mm against a
    # printed 76.0 mm -- which is a number the operator is told to verify with
    # a ruler before cutting.
    geo = X.target_geometry(target, E.patch_size_of(patch_tag),
                            E.size_mode_of(patch_tag))
    out_dir = out_dir or os.path.join(N.PHYSICAL_DIR, "%s__%s"
                                      % (patch_tag, geo["target"]))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for theta in N.PHYS_REF_THETA_DEG:
        for light in N.PHYS_LIGHTING:
            for rep in range(N.PHYS_REPEATS):
                scene = "t%02d_%s_%02d" % (theta, light, rep)
                for cond in ("clean", "patched"):
                    rows.append({
                        "scene": scene, "condition": cond,
                        "filename": "%s__%s__00.jpg" % (scene, cond),
                        "nominal_theta_deg": theta, "lighting": light,
                        "repeat": rep,
                        "camera_distance_m": round(geo["camera_distance_m"], 3),
                        "shot": 0,
                    })

    path = os.path.join(out_dir, "capture_plan.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    readme = os.path.join(out_dir, "HOW_TO_CAPTURE.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(_capture_instructions(geo, patch_tag, len(rows)))

    print("capture plan: %d photographs (%d pairs)" % (len(rows), len(rows) // 2))
    print("  %s" % path)
    print("  %s" % readme)
    print("\nPhotograph from %.2f m, which is the scaled equivalent of %.0f m "
          "altitude." % (geo["camera_distance_m"], geo["ref_altitude_m"]))
    return out_dir


def _capture_instructions(geo, tag, n):
    return """CAPTURE SESSION -- %s, target %s (1:%.0f)
%s

You need %d photographs: %d pairs, each pair one clean and one patched shot of
the same vehicle from the same pose.

BEFORE YOU START
  1. Print the sheet from out/print/ at 100 percent. Measure the ruler on the
     sheet. If it does not read what it says, stop and fix the printer; every
     number downstream depends on the decal being %.1f mm.
  2. Cut on the OUTER crop marks so the four ArUco markers stay attached.
  3. Put the vehicle on a plain, matte, non-repeating surface. A patterned rug
     or a reflective table top will cost you more detections than the attack
     does, and you will not be able to tell the two apart afterwards.
  4. Mark the vehicle's position on the surface with tape. It must not move
     between the clean and patched shot of a pair.

FOR EACH PAIR
  a. Photograph the vehicle WITHOUT the decal. Name it <scene>__clean__00.jpg
  b. Lay the decal on the roof, arrow toward the nose, centred.
  c. Photograph again from the SAME position. Name it <scene>__patched__00.jpg
  d. Remove the decal.

Do not move the camera between a and c. Lock exposure and focus if your phone
allows it; auto-exposure changing between the two shots is the most common way
a pair stops being a controlled comparison.

CAMERA DISTANCE: %.2f m, standing so the camera looks down at the nominal
angle for that cell. The exact angle does not have to be right -- it is
measured from the markers afterwards -- but it should be close enough that the
cells are spread across the range.

ALL FOUR MARKERS SHOULD BE IN FRAME AND IN FOCUS. At least %d must be
readable or the photograph is reported unscored rather than guessed at.

WHEN YOU ARE DONE
  physdecal physical ingest   --dir <this folder>
  physdecal physical annotate --dir <this folder>
  physdecal physical score    --dir <this folder>
""" % (tag, geo["target"], 1 / geo["scale"], geo["note"], n, n // 2,
       geo["patch_mm"], geo["camera_distance_m"], N.PHYS_MIN_MARKERS)


# =============================================================== 2. ingest

def camera_matrix(img_shape, fov_deg=None, exif_path=None):
    """
    A camera matrix from the field of view.

    Returns (K, source). A phone's true intrinsics are not known without
    calibrating that phone, and calibrating it is a better experiment than
    guessing, so the guess is labelled everywhere it is used. EXIF focal
    length in 35 mm equivalent is used when present, which is accurate to a
    few percent on modern phones; otherwise --fov is used.
    """
    h, w = img_shape[:2]
    f_px = None
    source = None

    if exif_path:
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            exif = Image.open(exif_path).getexif()
            # FocalLengthIn35mmFilm lives in the Exif sub-IFD (0x8769), not the
            # base IFD that getexif() returns. Reading only the base IFD finds
            # Make and Model and silently misses the focal length, so every
            # photograph falls back to the assumed field of view while looking
            # like it had no EXIF at all.
            vals = {TAGS.get(k, k): v for k, v in exif.items()}
            try:
                vals.update({TAGS.get(k, k): v
                             for k, v in exif.get_ifd(0x8769).items()})
            except Exception:
                pass
            f35 = vals.get("FocalLengthIn35mmFilm")
            if f35:
                # 35 mm frame is 36 mm wide; f_px = f35 / 36 * image width
                f_px = float(f35) / 36.0 * max(w, h)
                source = "exif FocalLengthIn35mmFilm=%s" % f35
        except Exception:
            pass

    if f_px is None:
        fov = fov_deg or N.PHYS_DEFAULT_FOV_DEG
        f_px = 0.5 * max(w, h) / np.tan(np.radians(fov) / 2.0)
        source = "assumed fov %.1f deg" % fov

    K = np.array([[f_px, 0, w / 2.0], [0, f_px, h / 2.0], [0, 0, 1]], np.float64)
    return K, source


def detect_markers(bgr):
    """
    The four frame markers, by id.

    The detector is deliberately permissive. A marker printed at the size this
    study computes lands near the decode floor by construction, so a session
    shot a little further away than planned loses every marker under OpenCV's
    defaults even though the squares are plainly there in the photograph.
    Relaxing the perimeter and threshold windows recovers them.

    The cost is false positives: lamp fittings and paper texture decode as
    other ids. That is harmless here because only ids 0..3 are ever read, and
    a repeated id keeps its FIRST detection rather than its last, so a spurious
    0..3 landing later in the list cannot displace a real one.
    """
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, N.FIDUCIAL_DICT))
    par = cv2.aruco.DetectorParameters()
    par.minMarkerPerimeterRate = 0.005
    par.adaptiveThreshWinSizeMin = 3
    par.adaptiveThreshWinSizeMax = 53
    par.adaptiveThreshWinSizeStep = 4
    par.polygonalApproxAccuracyRate = 0.08
    par.perspectiveRemovePixelPerCell = 8
    par.maxErroneousBitsInBorderRate = 0.6
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(adict, par).detectMarkers(bgr)
    if ids is None:
        return {}
    found = {}
    for i, c in zip(ids.flatten(), corners):
        found.setdefault(int(i), c.reshape(4, 2))
    return found


def pose_from_markers(found, pitch_mm, fid_mm, K):
    """
    Nadir angle and distance, from the marker centres of a known square.

    The four markers sit at the corners of a square of side pitch_mm in the
    plane of the sheet, ids 0..3 clockwise from top left, which is the order
    export_print.py writes them in. solvePnP against those four object points
    gives the sheet's pose in the camera frame; the nadir angle is the angle
    between the sheet normal and the camera's view direction.
    """
    obj, img = [], []
    half = pitch_mm / 2.0
    ref = {0: (-half, -half), 1: (half, -half), 2: (half, half), 3: (-half, half)}
    for i, (X, Y) in ref.items():
        if i in found:
            obj.append([X, Y, 0.0])
            img.append(found[i].mean(axis=0))
    if len(obj) < 3:
        return None

    obj = np.asarray(obj, np.float64)
    img = np.asarray(img, np.float64)
    # AP3P is named for three points but OpenCV's solvePnP wrapper still
    # demands four (the fourth disambiguates the P3P solutions), so it
    # asserts on exactly the three-marker case PHYS_MIN_MARKERS allows.
    # SQPNP is the estimator that accepts three.
    flag = cv2.SOLVEPNP_ITERATIVE if len(obj) >= 4 else cv2.SOLVEPNP_SQPNP
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=flag)
    if not ok:
        return None

    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]                       # sheet normal in camera coordinates
    view = tvec.flatten() / (np.linalg.norm(tvec) + 1e-9)
    cosang = abs(float(np.dot(normal, view)))
    theta = float(np.degrees(np.arccos(np.clip(cosang, 0, 1))))
    dist_mm = float(np.linalg.norm(tvec))
    return {"theta_deg": theta, "distance_m": dist_mm / 1000.0,
            "n_markers": len(obj), "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist()}


def decal_quad(found, pitch_mm, fid_mm, decal_mm):
    """
    The decal's four corners in image pixels, via the marker homography.

    This is what gives the physical experiment the same under/away split the
    simulation has: the footprint is measured, not estimated from where the
    decal looks like it is.
    """
    src, dst = [], []
    half = pitch_mm / 2.0
    ref = {0: (-half, -half), 1: (half, -half), 2: (half, half), 3: (-half, half)}
    for i, (X, Y) in ref.items():
        if i in found:
            src.append([X, Y])
            dst.append(found[i].mean(axis=0))
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.asarray(src, np.float32),
                              np.asarray(dst, np.float32))
    if H is None:
        return None
    d = decal_mm / 2.0
    corners = np.array([[-d, -d], [d, -d], [d, d], [-d, d]], np.float32)
    return cv2.perspectiveTransform(corners.reshape(1, 4, 2), H).reshape(4, 2)


def parse_name(fn):
    """<scene>__<condition>__<index>.<ext>"""
    base = os.path.splitext(os.path.basename(fn))[0]
    bits = base.split("__")
    if len(bits) < 2 or bits[1] not in ("clean", "patched"):
        return None
    return {"scene": bits[0], "condition": bits[1],
            "index": bits[2] if len(bits) > 2 else "00"}


def ingest(folder, patch_tag=None, target=None, fov=None):
    """Scan photographs, recover pose, write the manifest."""
    from physdecal.physical import export_print as X

    print_meta = _find_print_meta(folder, patch_tag, target)
    pitch_mm = print_meta["layout"]["marker_pitch_mm"]
    fid_mm = print_meta["layout"]["fiducial_mm"]
    decal_mm = print_meta["geometry"]["patch_mm"]

    photos = []
    for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
        photos += glob.glob(os.path.join(folder, "*.%s" % ext))
    photos = sorted(set(photos))
    if not photos:
        raise SystemExit("No photographs in %s" % folder)

    rows = []
    for p in photos:
        name = parse_name(p)
        if not name:
            print("  skipping %s: name is not <scene>__<clean|patched>__<i>"
                  % os.path.basename(p))
            continue
        bgr = cv2.imread(p)
        if bgr is None:
            print("  skipping %s: unreadable" % os.path.basename(p))
            continue

        K, ksrc = camera_matrix(bgr.shape, fov, p)
        found = detect_markers(bgr)
        row = {"file": os.path.basename(p), "scene": name["scene"],
               "condition": name["condition"], "index": name["index"],
               "width": bgr.shape[1], "height": bgr.shape[0],
               "n_markers": len(found), "K_source": ksrc,
               "theta_deg": "", "distance_m": "", "decal_quad": "",
               "status": ""}

        if len(found) < N.PHYS_MIN_MARKERS:
            row["status"] = "unscored: only %d markers" % len(found)
        else:
            pose = pose_from_markers(found, pitch_mm, fid_mm, K)
            if pose is None:
                row["status"] = "unscored: pose failed"
            else:
                row["theta_deg"] = round(pose["theta_deg"], 2)
                row["distance_m"] = round(pose["distance_m"], 4)
                row["n_markers"] = pose["n_markers"]
                row["status"] = "ok"
                q = decal_quad(found, pitch_mm, fid_mm, decal_mm)
                if q is not None:
                    row["decal_quad"] = json.dumps(
                        [[round(float(a), 1), round(float(b), 1)] for a, b in q])
        rows.append(row)

    # A clean photo has no markers on it, by construction, so it can never
    # recover its own pose. It inherits the pose of its pair, which is exactly
    # right: the camera did not move between them, and that is the premise the
    # whole paired protocol rests on.
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["scene"], {})[r["condition"]] = r
    n_inherited = 0
    for scene, pair in by_scene.items():
        pat, cln = pair.get("patched"), pair.get("clean")
        if pat and cln and pat["status"] == "ok" and cln["status"] != "ok":
            cln["theta_deg"] = pat["theta_deg"]
            cln["distance_m"] = pat["distance_m"]
            cln["status"] = "ok (pose inherited from the patched member)"
            n_inherited += 1

    path = os.path.join(folder, MANIFEST)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"].startswith("ok"))
    pairs = sum(1 for s, p in by_scene.items()
                if "clean" in p and "patched" in p)
    print("ingested %d photographs, %d with a pose (%d inherited), %d complete "
          "pairs" % (len(rows), ok, n_inherited, pairs))
    thetas = [float(r["theta_deg"]) for r in rows if r["theta_deg"] != ""]
    if thetas:
        print("  measured nadir angle: %.0f to %.0f degrees"
              % (min(thetas), max(thetas)))
    unpaired = [s for s, p in by_scene.items() if len(p) < 2]
    if unpaired:
        print("  %d scenes have no pair and cannot be scored: %s"
              % (len(unpaired), ", ".join(sorted(unpaired)[:6])))
    print("  -> %s" % path)
    return path


def _find_print_meta(folder, patch_tag, target):
    """The print record that produced the decal in these photographs."""
    local = os.path.join(folder, "print.json")
    if os.path.isfile(local):
        with open(local, encoding="utf-8") as f:
            return json.load(f)
    if patch_tag:
        t = target or N.PHYS_TARGET
        p = os.path.join(N.PRINT_DIR, "%s__%s" % (patch_tag, t), "print.json")
        if os.path.isfile(p):
            shutil.copy(p, local)
            print("  copied the print record into the session folder")
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    raise SystemExit(
        "No print.json for this session.\nCopy the one from out/print/<tag>__"
        "<target>/ into %s, or pass --patch and --target so it can be found. "
        "It carries the marker pitch and the decal size, without which a "
        "photograph cannot be turned into a measurement." % folder)


# ============================================================= 3. annotate

def annotate(folder, redo=False):
    """
    Draw the vehicle mask for each scene, once per scene.

    GrabCut proposes from a dragged box and the mouse corrects it. One mask
    serves both members of a pair, because the vehicle does not move between
    them, which halves the work and removes a source of inconsistency between
    the clean and patched ground truth.
    """
    rows = _read_manifest(folder)
    ann_dir = os.path.join(folder, ANNOT_DIR)
    os.makedirs(ann_dir, exist_ok=True)

    scenes = {}
    for r in rows:
        scenes.setdefault(r["scene"], []).append(r)

    todo = []
    for scene, rs in sorted(scenes.items()):
        out = os.path.join(ann_dir, "%s_mask.png" % scene)
        if os.path.isfile(out) and not redo:
            continue
        clean = next((r for r in rs if r["condition"] == "clean"), rs[0])
        todo.append((scene, os.path.join(folder, clean["file"]), out))

    if not todo:
        print("every scene already has a mask. Pass --redo to replace them.")
        return

    print("%d scenes to annotate.\n" % len(todo))
    print("  drag a box round the VEHICLE, then press:")
    print("     g   run GrabCut inside the box")
    print("     f/b paint foreground / background with the mouse, then g again")
    print("     s   save and go to the next scene")
    print("     q   quit\n")

    for scene, src, out in todo:
        if not _annotate_one(scene, src, out):
            print("stopped. %d scenes still unannotated." % len(todo))
            return
    print("done.")


def _annotate_one(scene, src_path, out_path):
    img = cv2.imread(src_path)
    if img is None:
        print("  cannot read %s" % src_path)
        return True
    disp_scale = min(1.0, 1100.0 / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=disp_scale, fy=disp_scale)

    state = {"box": None, "drag": False, "p0": None, "brush": None,
             "mask": np.zeros(small.shape[:2], np.uint8)}

    def on_mouse(ev, x, y, flags, _):
        if state["brush"] is None:
            if ev == cv2.EVENT_LBUTTONDOWN:
                state["p0"] = (x, y); state["drag"] = True
            elif ev == cv2.EVENT_MOUSEMOVE and state["drag"]:
                state["box"] = (*state["p0"], x, y)
            elif ev == cv2.EVENT_LBUTTONUP:
                state["drag"] = False
                state["box"] = (*state["p0"], x, y)
        else:
            if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE) and \
                    (flags & cv2.EVENT_FLAG_LBUTTON):
                cv2.circle(state["mask"], (x, y), 6,
                           1 if state["brush"] == "f" else 2, -1)

    win = "annotate: %s" % scene
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = small.copy()
        if state["box"]:
            x0, y0, x1, y1 = state["box"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 200, 0), 2)
        fg = state["mask"] == 1
        bgm = state["mask"] == 2
        vis[fg] = (0.5 * vis[fg] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        vis[bgm] = (0.5 * vis[bgm] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF

        if k == ord("q"):
            cv2.destroyWindow(win)
            return False
        if k == ord("f"):
            state["brush"] = "f"
        if k == ord("b"):
            state["brush"] = "b"
        if k == ord("g") and state["box"]:
            gc = np.where(state["mask"] == 1, cv2.GC_FGD,
                          np.where(state["mask"] == 2, cv2.GC_BGD,
                                   cv2.GC_PR_BGD)).astype(np.uint8)
            x0, y0, x1, y1 = state["box"]
            rect = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            mode = (cv2.GC_INIT_WITH_MASK if (state["mask"] > 0).any()
                    else cv2.GC_INIT_WITH_RECT)
            if mode == cv2.GC_INIT_WITH_MASK:
                gc[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
                    np.where(gc[rect[1]:rect[1] + rect[3],
                                rect[0]:rect[0] + rect[2]] == cv2.GC_PR_BGD,
                             cv2.GC_PR_FGD,
                             gc[rect[1]:rect[1] + rect[3],
                                rect[0]:rect[0] + rect[2]])
            bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            cv2.grabCut(small, gc, rect, bgd, fgd, 4, mode)
            keep = np.isin(gc, [cv2.GC_FGD, cv2.GC_PR_FGD])
            state["mask"] = np.where(keep, 1, 2).astype(np.uint8)
            state["brush"] = state["brush"] or "f"
        if k == ord("s"):
            full = cv2.resize((state["mask"] == 1).astype(np.uint8) * 255,
                              (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(out_path, full)
            print("  %s -> %s" % (scene, os.path.basename(out_path)))
            cv2.destroyWindow(win)
            return True


# ================================================================ 4. score

def _read_manifest(folder):
    path = os.path.join(folder, MANIFEST)
    if not os.path.isfile(path):
        raise SystemExit("No %s. Run  physdecal physical ingest --dir %s  first."
                         % (path, folder))
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_tensor(bgr, long_side, device):
    h, w = bgr.shape[:2]
    s = long_side / float(max(h, w))
    nh, nw = int(round(h * s / 32)) * 32, int(round(w * s / 32)) * 32
    img = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (nw, nh))
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    return t.unsqueeze(0).to(device), (nh / h, nw / w)


def load_gts(folder, scene):
    """
    Ground truth for both members of a pair: the clean mask from annotate, and
    for the patched photograph its own mask when `track` has written one.

    One mask per pair assumes nothing moves between the two shots. With a
    hand-held camera, or a car nudged while the decal goes on, something does,
    and a shared mask then scores the misalignment as if it were the attack.
    """
    ann = os.path.join(folder, ANNOT_DIR)
    gc = cv2.imread(os.path.join(ann, "%s_mask.png" % scene), cv2.IMREAD_GRAYSCALE)
    if gc is None:
        return None
    gp = cv2.imread(os.path.join(ann, "%s_patched_mask.png" % scene),
                    cv2.IMREAD_GRAYSCALE)
    return {"clean": gc, "patched": gc if gp is None else gp}


def _gt(gt, cond):
    return gt[cond] if isinstance(gt, dict) else gt


def track_masks(folder):
    """
    Write each patched photograph's vehicle mask by following the car from its
    clean photograph: SIFT features on the car body (the clean mask, eroded so
    the floor contributes nothing), matched into the patched photograph, a
    RANSAC similarity transform, and the clean mask moved by it. The decal
    region is covered because the mask moves with the body, which colour-based
    masking cannot do when a pale decal matches the floor.
    """
    sift = cv2.SIFT_create(nfeatures=6000)
    ann = os.path.join(folder, ANNOT_DIR)
    pairs = {}
    for r in _read_manifest(folder):
        pairs.setdefault(r["scene"], {})[r["condition"]] = r["file"]
    s, done = 0.5, 0
    for scene, files in sorted(pairs.items()):
        gc = cv2.imread(os.path.join(ann, "%s_mask.png" % scene), cv2.IMREAD_GRAYSCALE)
        if gc is None or "patched" not in files or "clean" not in files:
            continue
        clean = cv2.imread(os.path.join(folder, files["clean"]))
        patched = cv2.imread(os.path.join(folder, files["patched"]))
        c = cv2.resize(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)
        pg = cv2.resize(cv2.cvtColor(patched, cv2.COLOR_BGR2GRAY), None, fx=s, fy=s)
        m = cv2.resize(gc, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        m = cv2.erode((m > 127).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        kc, dc = sift.detectAndCompute(c, m)
        kp, dp = sift.detectAndCompute(pg, None)
        good = [a for a, b in cv2.BFMatcher().knnMatch(dc, dp, k=2)
                if a.distance < 0.75 * b.distance]
        M = inl = None
        if len(good) >= 8:
            A = np.float32([kc[g.queryIdx].pt for g in good])
            B = np.float32([kp[g.trainIdx].pt for g in good])
            M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC,
                                                 ransacReprojThreshold=4)
        if M is None or inl.sum() < 8:
            print("  %s: no consistent car motion found, patched mask not "
                  "written" % scene)
            continue
        M[:, 2] /= s                                # rotation/scale are scale-free
        cv2.imwrite(os.path.join(ann, "%s_patched_mask.png" % scene),
                    cv2.warpAffine(gc, M, (patched.shape[1], patched.shape[0]),
                                   flags=cv2.INTER_NEAREST))
        done += 1
    print("tracked %d patched masks in %s" % (done, folder))


def match_scale(imgs, gt, target, seed=0):
    """
    Extend a pair and its mask with plain floor so the vehicle's long side is
    `target` of the frame's long side, as a wider lens from the same position
    would show it.

    Why this exists. The victims were trained on frames from an 84-degree
    camera, in which a vehicle at 20-60 m spans roughly 6-18 percent of the
    frame. A longer lens at the same standoff shows the model car two to three
    times larger, a scale no victim has seen, and dense segmentation collapses
    there (the paper's physical limitation). The floor in these sessions is
    plain, so extending it with its own median colour and a faint grain
    reproduces the wider field of view without inventing structure. Both
    members of a pair get the same geometry; each keeps its own floor colour,
    so the pair still differs only where the decal is.

    Returns (imgs, gt, factor). factor 1 means the frame was already at or
    below the target scale and is unchanged.
    """
    ref = _gt(gt, "clean")
    cnts, _ = cv2.findContours((ref > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    length = max(cv2.minAreaRect(np.vstack(cnts))[1])
    k = (length / float(max(ref.shape))) / target
    if k <= 1.0:
        return imgs, gt, 1.0
    H, W = ref.shape
    ph, pw = int(H * (k - 1) / 2), int(W * (k - 1) / 2)
    rng = np.random.default_rng(seed)
    out = {}
    for cond, bgr in imgs.items():
        b = max(10, int(0.05 * min(H, W)))
        border = np.concatenate([bgr[:b].reshape(-1, 3), bgr[-b:].reshape(-1, 3),
                                 bgr[:, :b].reshape(-1, 3), bgr[:, -b:].reshape(-1, 3)])
        canvas = np.empty((H + 2 * ph, W + 2 * pw, 3), np.float32)
        canvas[:] = np.median(border, 0)
        canvas += rng.normal(0, 3, canvas.shape)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        canvas[ph:ph + H, pw:pw + W] = bgr
        out[cond] = canvas
    padg = lambda m: cv2.copyMakeBorder(m, ph, ph, pw, pw, cv2.BORDER_CONSTANT,
                                        value=0)
    g = ({c: padg(m) for c, m in gt.items()} if isinstance(gt, dict)
         else padg(gt))
    return out, g, round(k, 3)


def score(folder, arm="both", seg_model=None, det_model=None, device=None,
          require_pose=True, scale_to=None):
    """
    Run the victims on each pair and write the same metrics the simulation
    reports, so the two tables can sit side by side.
    """
    from physdecal.evaluation import evaluate as E

    device = device or N.DEVICE
    rows = _read_manifest(folder)
    ann_dir = os.path.join(folder, ANNOT_DIR)

    pairs = {}
    for r in rows:
        pairs.setdefault(r["scene"], {})[r["condition"]] = r
    # A recovered pose is normally required, because a photograph that cannot
    # say what angle it was taken from cannot be binned against the simulated
    # grid. ASR itself does not use the pose: it needs the mask and the two
    # predictions. require_pose=False scores the pair anyway and leaves the
    # angle blank, for a session whose fiducial geometry is unrecoverable.
    # Such rows carry no viewpoint and must not be reported per angle.
    usable = {s: p for s, p in pairs.items()
              if "clean" in p and "patched" in p
              and (p["patched"]["status"].startswith("ok") or not require_pose)}
    if not usable:
        raise SystemExit(
            "No scorable pairs. A pair needs a clean and a patched photograph "
            "and a recovered pose on the patched member.")
    print("%d scorable pairs of %d scenes" % (len(usable), len(pairs)))

    seg = det = None
    if arm in ("seg", "both"):
        from physdecal.victims import segmentation as M
        seg = M.load_model(seg_model or N.ATTACK_MODELS[0], device=device)
    if arm in ("det", "both"):
        from physdecal.victims import detection as DET
        det = DET.load_detector(det_model or N.DET_ATTACK_MODELS[0],
                                device=device)

    out = []
    for scene, pair in sorted(usable.items()):
        rec = {"scene": scene,
               "theta_deg": pair["patched"]["theta_deg"],
               "distance_m": pair["patched"]["distance_m"],
               "n_markers": pair["patched"]["n_markers"],
               "K_source": pair["patched"]["K_source"]}

        gt = load_gts(folder, scene)
        if gt is not None:
            rec["patched_mask"] = int(os.path.isfile(os.path.join(
                ann_dir, "%s_patched_mask.png" % scene)))

        imgs = {}
        for cond in ("clean", "patched"):
            bgr = cv2.imread(os.path.join(folder, pair[cond]["file"]))
            if bgr is None:
                break
            imgs[cond] = bgr
        if len(imgs) < 2:
            continue
        if scale_to and gt is not None:
            imgs, gt, rec["scale_factor"] = match_scale(
                imgs, gt, scale_to, seed=sum(map(ord, scene)))

        if seg is not None:
            if gt is None:
                rec["seg_status"] = "no mask: run annotate"
            else:
                rec.update(_score_seg(seg, imgs, gt, device))
        if det is not None:
            if gt is None:
                rec["det_status"] = "no mask: run annotate"
            else:
                rec.update(_score_det(det, imgs, gt, device))
        out.append(rec)

    path = os.path.join(folder, "physical_results%s.csv" % (
        "_scaled%02d" % round(100 * scale_to) if scale_to else ""))
    keys = sorted({k for r in out for k in r})
    order = ["scene", "theta_deg", "distance_m", "n_markers", "K_source"]
    keys = order + [k for k in keys if k not in order]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    _summarise_physical(out, path)
    return path


@torch.no_grad()
def _score_seg(model, imgs, gt, device):
    from physdecal.evaluation import evaluate as E

    res = {}
    for cond in ("clean", "patched"):
        t, (sy, sx) = _to_tensor(imgs[cond], N.INPUT_LONG_SIDE, device)
        p = model(t)["p_veh"][0, 0].cpu().numpy() > 0.5
        g = cv2.resize(_gt(gt, cond), (p.shape[1], p.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 127
        res["%s_iou" % cond] = round(E.iou(p, g), 4)
        res["_pred_%s" % cond] = p
        res["_gt_%s" % cond] = g

    pc, pa = res.pop("_pred_clean"), res.pop("_pred_patched")
    gc, gp = res.pop("_gt_clean"), res.pop("_gt_patched")
    res["iou_drop"] = round(res["clean_iou"] - res["patched_iou"], 4)
    res["seg_attackable"] = int(res["clean_iou"] >= N.ATTACKABLE_IOU)
    res["seg_success"] = int(res["seg_attackable"]
                             and res["patched_iou"] < N.ASR_IOU_THRESHOLD)
    # vehicle pixels correctly predicted clean minus under attack; with a mask
    # per photograph there is no shared pixel grid to compare pixel by pixel
    res["flip_px"] = int((gc & pc).sum()) - int((gp & pa).sum())
    return res


@torch.no_grad()
def _score_det(model, imgs, gt, device):
    from physdecal.victims import detection as DET

    res = {}
    for cond in ("clean", "patched"):
        ys, xs = np.nonzero(_gt(gt, cond) > 127)
        if len(xs) == 0:
            return {"det_status": "empty mask"}
        bgr = imgs[cond]
        t, (sy, sx) = _to_tensor(bgr, N.DET_INPUT_LONG_SIDE, device)
        box = [xs.min() * sx, ys.min() * sy, xs.max() * sx, ys.max() * sy]
        pred = model.predict(t)[0]
        ids = model.vehicle_ids("car")
        hit = DET.best_hit(pred, box, ids)
        res["%s_detected" % cond] = int(hit is not None)
        res["%s_score" % cond] = round(DET.max_vehicle_score(pred, box, ids), 4)
    res["score_drop"] = round(res["clean_score"] - res["patched_score"], 4)
    res["det_attackable"] = res["clean_detected"]
    res["det_success"] = int(res["clean_detected"] and not res["patched_detected"])
    return res


def _summarise_physical(rows, path):
    print("\n%-28s %6s %8s %8s" % ("", "n", "attack.", "ASR"))
    for arm, att_k, suc_k, extra_k, label in (
            ("segmentation", "seg_attackable", "seg_success", "iou_drop",
             "mean IoU drop"),
            ("detection", "det_attackable", "det_success", "score_drop",
             "mean score drop")):
        have = [r for r in rows if att_k in r]
        if not have:
            continue
        att = [r for r in have if r[att_k]]
        if not att:
            print("%-28s %6d %8d  attackable set EMPTY -- ASR undefined"
                  % (arm, len(have), 0))
            continue
        asr = sum(r[suc_k] for r in att) / len(att)
        ex = np.mean([r[extra_k] for r in att if extra_k in r])
        print("%-28s %6d %8d %8.3f   %s %.3f"
              % (arm, len(have), len(att), asr, label, ex))
    print("\n-> %s" % path)
    print("\nPut these next to the simulated numbers from physdecal evaluate. The "
          "gap between them\nis the sim-to-real gap, and it is the single most "
          "informative number in the paper.")


# ================================================================ 5. panel

def panel(folder, scenes=None, n=3, seg_model=None, det_model=None,
          device=None, out=None):
    """The physical counterpart of panels.py: photographs, with what the
    victim predicted on them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physdecal.victims import detection as DET
    from physdecal.figures import panels as FG

    device = device or N.DEVICE
    rows = _read_manifest(folder)
    pairs = {}
    for r in rows:
        pairs.setdefault(r["scene"], {})[r["condition"]] = r
    usable = [s for s, p in sorted(pairs.items())
              if "clean" in p and "patched" in p]
    if scenes:
        usable = [s for s in usable if s in set(scenes)]
    usable = usable[:n]
    if not usable:
        raise SystemExit("no complete pairs to draw")

    det = DET.load_detector(det_model or N.DET_ATTACK_MODELS[0], device=device)
    ann_dir = os.path.join(folder, ANNOT_DIR)

    fig, axes = plt.subplots(len(usable), 2,
                             figsize=(9.0, 3.4 * len(usable)), squeeze=False)
    for r, scene in enumerate(usable):
        gt = load_gts(folder, scene)
        for c, cond in enumerate(("clean", "patched")):
            bgr = cv2.imread(os.path.join(folder, pairs[scene][cond]["file"]))
            t, (sy, sx) = _to_tensor(bgr, N.DET_INPUT_LONG_SIDE, device)
            pred = det.predict(t)[0]
            ids = det.vehicle_ids("car")
            rgb = t[0].permute(1, 2, 0).cpu().numpy()
            if gt is not None:
                ys, xs = np.nonzero(gt[cond] > 127)
                box = [xs.min() * sx, ys.min() * sy, xs.max() * sx, ys.max() * sy]
            else:
                box = [0, 0, 1, 1]
            nb = FG.draw_boxes(axes[r][c], rgb, pred, ids, box, (0, 0),
                               N.DET_SCORE)
            s = DET.max_vehicle_score(pred, box, ids)
            axes[r][c].set_title("%s  %s" % (scene, cond), fontsize=8.5)
            axes[r][c].set_xlabel(
                "best vehicle score %.3f\n%s" %
                (s, "VEHICLE LOST" if nb == 0 else "%d box(es)" % nb),
                fontsize=7.5, color=(FG.RED if nb == 0 else "black"))

    fig.suptitle("Physical capture: printed decal on a real vehicle", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out or os.path.join(N.FIGURE_DIR, "panel_physical.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=N.PANEL_DPI, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out)
    return out


@torch.no_grad()
def seg_panel(folders, scale_to, n_per=2, seg_model=None, device=None,
              out=None, min_drop=0.20, select="success", label="Printed patch",
              title=None):
    """
    Segmentation on physical pairs, in the map style of the simulated figures:
    clean photograph, clean prediction, patched photograph, patched prediction,
    with the ground-truth outline on both maps. Rows are successes from each
    session's physical_results_scaledNN.csv whose IoU drop is at least
    min_drop, spread over the session in scene order. Everything is computed
    on the scale-matched frame (match_scale) and cropped around the vehicle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physdecal.evaluation import evaluate as E
    from physdecal.figures import panels as FG
    from physdecal.victims import segmentation as M

    device = device or N.DEVICE
    model = M.load_model(seg_model or N.ATTACK_MODELS[0], device=device)
    tag = "_scaled%02d" % round(100 * scale_to)
    rows = []
    for folder in folders:
        with open(os.path.join(folder, "physical_results%s.csv" % tag),
                  newline="", encoding="utf-8") as f:
            res = [r for r in csv.DictReader(f)
                   if (r.get("seg_success") == "1" and float(r["iou_drop"]) >= min_drop)
                   or (select == "even" and r.get("seg_attackable") == "1")]
        idx = np.linspace(0, len(res) - 1, min(n_per, len(res))).round().astype(int)
        man = {}
        for r in _read_manifest(folder):
            man.setdefault(r["scene"], {})[r["condition"]] = r["file"]
        rows += [(folder, res[i]["scene"], man[res[i]["scene"]]) for i in idx]
    if not rows:
        raise SystemExit("no scale-matched pairs to draw")

    fig, axes = plt.subplots(len(rows), 4, figsize=(11.0, 2.9 * len(rows)),
                             squeeze=False)
    for r, (folder, scene, files) in enumerate(rows):
        gt = load_gts(folder, scene)
        imgs = {c: cv2.imread(os.path.join(folder, files[c]))
                for c in ("clean", "patched")}
        imgs, gt, k = match_scale(imgs, gt, scale_to, seed=sum(map(ord, scene)))
        maps = {}
        for c in ("clean", "patched"):
            t, _ = _to_tensor(imgs[c], N.INPUT_LONG_SIDE, device)
            p = model(t)["p_veh"][0, 0].cpu().numpy() > 0.5
            g = cv2.resize(gt[c], (p.shape[1], p.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 127
            maps[c] = (t[0].permute(1, 2, 0).cpu().numpy(), p, g,
                       E.iou(p, g))
        ys, xs = np.nonzero(maps["clean"][2] | maps["patched"][2])
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        half = int(1.1 * max(ys.max() - ys.min(), xs.max() - xs.min()))
        H, W = maps["clean"][2].shape
        y0, y1 = max(cy - half, 0), min(cy + half, H)
        x0, x1 = max(cx - half, 0), min(cx + half, W)
        car = "car A" if folder.rstrip("/\\").endswith("_a") else "car B"
        i_c = maps["clean"][3]
        for c, (cond, head) in enumerate((("clean", "Clean photograph"),
                                           ("patched", label))):
            rgb, p, g, i = maps[cond]
            FG.tile(axes[r][2 * c], rgb[y0:y1, x0:x1],
                    head if r == 0 else "",
                    ("%s, %s" % (car, scene.replace("t00_", "").replace("_", " ")))
                    if c == 0 else "")
            FG.tile(axes[r][2 * c + 1], FG._seg_map(p[y0:y1, x0:x1], g[y0:y1, x0:x1]),
                    "Prediction" if r == 0 else "",
                    "IoU %.3f" % i if c == 0 else
                    "IoU %.3f  (drop %.3f)" % (i, i_c - i))
    FG._seg_legend(fig)
    fig.suptitle(title or "Physical capture, SegFormer-B0, scale-matched to "
                 "%.0f%% of the frame" % (100 * scale_to), fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = out or os.path.join(N.FIGURE_DIR, "panel_physical_seg.png")
    fig.savefig(out, dpi=N.PANEL_DPI, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out)
    return out


@torch.no_grad()
def paired_panel(pairs, n_per=2, seg_model=None, device=None, out=None,
                 title=None):
    """
    Clean, printed patch and grey square at the SAME pose, each with its
    SegFormer-B0 prediction and its own ground-truth outline. pairs is a list
    of (label, patch_session, grey_session) that share clean photographs (the
    wide-lens sessions do: each pose was shot clean, with the patch and with
    the grey square). Rows are spread evenly over the shared scenes. No scale
    correction: these frames were taken at the trained scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from physdecal.evaluation import evaluate as E
    from physdecal.figures import panels as FG
    from physdecal.victims import segmentation as M

    device = device or N.DEVICE
    model = M.load_model(seg_model or N.ATTACK_MODELS[0], device=device)
    rows = []
    for label, pdir, gdir in pairs:
        man = {}
        for d in (pdir, gdir):
            for r in _read_manifest(d):
                man.setdefault(d, {}).setdefault(r["scene"], {})[r["condition"]] = r["file"]
        common = sorted(set(man[pdir]) & set(man[gdir]))
        idx = np.linspace(0, len(common) - 1, min(n_per, len(common))).round().astype(int)
        rows += [(label, pdir, gdir, common[i], man) for i in idx]

    fig, axes = plt.subplots(len(rows), 6, figsize=(15.0, 2.7 * len(rows)),
                             squeeze=False)
    heads = ("Clean photograph", "Printed patch", "Grey square")
    for r, (label, pdir, gdir, scene, man) in enumerate(rows):
        gp, gg = load_gts(pdir, scene), load_gts(gdir, scene)
        items = [(pdir, man[pdir][scene]["clean"], gp["clean"]),
                 (pdir, man[pdir][scene]["patched"], gp["patched"]),
                 (gdir, man[gdir][scene]["patched"], gg["patched"])]
        maps = []
        for d, f, gt in items:
            t, _ = _to_tensor(cv2.imread(os.path.join(d, f)), N.INPUT_LONG_SIDE, device)
            p = model(t)["p_veh"][0, 0].cpu().numpy() > 0.5
            g = cv2.resize(gt, (p.shape[1], p.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 127
            maps.append((t[0].permute(1, 2, 0).cpu().numpy(), p, g, E.iou(p, g)))
        ys, xs = np.nonzero(maps[0][2] | maps[1][2] | maps[2][2])
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        half = int(0.75 * max(ys.max() - ys.min(), xs.max() - xs.min()))
        H, W = maps[0][2].shape
        y0, y1 = max(cy - half, 0), min(cy + half, H)
        x0, x1 = max(cx - half, 0), min(cx + half, W)
        i_c = maps[0][3]
        for c, (rgb, p, g, i) in enumerate(maps):
            FG.tile(axes[r][2 * c], rgb[y0:y1, x0:x1], heads[c] if r == 0 else "",
                    "%s, pose %s" % (label, scene.split("_")[0][1:]) if c == 0 else "")
            FG.tile(axes[r][2 * c + 1], FG._seg_map(p[y0:y1, x0:x1], g[y0:y1, x0:x1]),
                    "Prediction" if r == 0 else "",
                    "IoU %.3f" % i if c == 0 else "IoU %.3f  (drop %.3f)" % (i, i_c - i))
    FG._seg_legend(fig)
    fig.suptitle(title or "Physical capture at the trained scale, SegFormer-B0",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = out or os.path.join(N.FIGURE_DIR, "panel_physical_wide.png")
    fig.savefig(out, dpi=N.PANEL_DPI, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % out)
    return out


# ==================================================================== cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write the capture plan and instructions")
    p.add_argument("--patch", required=True)
    p.add_argument("--target", default=None)
    p.add_argument("--dir", default=None)

    p = sub.add_parser("ingest", help="recover pose from the fiducials")
    p.add_argument("--dir", required=True)
    p.add_argument("--patch", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--fov", type=float, default=None,
                   help="camera horizontal field of view in degrees, used "
                        "only when EXIF does not carry a focal length")

    p = sub.add_parser("annotate", help="draw the vehicle mask per scene")
    p.add_argument("--dir", required=True)
    p.add_argument("--redo", action="store_true")

    p = sub.add_parser("track", help="per-photograph masks for the patched "
                       "shots, following the car from its clean shot")
    p.add_argument("--dir", required=True)

    p = sub.add_parser("segpanel", help="segmentation figure from scale-"
                       "matched successes of one or more sessions")
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--scale-to", type=float, required=True)
    p.add_argument("--n-per", type=int, default=2)
    p.add_argument("--model", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)

    p = sub.add_parser("score", help="run the victims on the pairs")
    p.add_argument("--dir", required=True)
    p.add_argument("--arm", default="both", choices=["seg", "det", "both"])
    p.add_argument("--no-pose", action="store_true",
                   help="score pairs whose fiducials never resolved; the "
                        "angle column is left blank and the rows must not be "
                        "reported per viewpoint")
    p.add_argument("--scale-to", type=float, default=None,
                   help="extend each pair with plain floor until the vehicle "
                        "spans this fraction of the frame's long side (see "
                        "match_scale); writes physical_results_scaledNN.csv")
    p.add_argument("--model", default=None)
    p.add_argument("--detector", default=None)
    p.add_argument("--device", default=None)

    p = sub.add_parser("panel", help="the physical figure")
    p.add_argument("--dir", required=True)
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--detector", default=None)
    p.add_argument("--device", default=None)

    a = ap.parse_args()
    if a.cmd == "plan":
        plan(a.patch, a.target, a.dir)
    elif a.cmd == "ingest":
        ingest(a.dir, a.patch, a.target, a.fov)
    elif a.cmd == "annotate":
        annotate(a.dir, a.redo)
    elif a.cmd == "track":
        track_masks(a.dir)
    elif a.cmd == "segpanel":
        seg_panel(a.dirs, a.scale_to, a.n_per, a.model, a.device, a.out)
    elif a.cmd == "score":
        score(a.dir, a.arm, a.model, a.detector, a.device,
              require_pose=not a.no_pose, scale_to=a.scale_to)
    elif a.cmd == "panel":
        panel(a.dir, a.scenes, a.n, None, a.detector, a.device)


if __name__ == "__main__":
    main()
