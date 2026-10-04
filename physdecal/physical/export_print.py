r"""
physdecal export-print
Turns an optimised patch into something you can put on a printer, cut out, and
stick on a car roof.

    physdecal export-print --patch segformer_b0_joint_r4 --target diecast
    physdecal export-print --patch segformer_b0_joint_r4 --target real
    physdecal export-print --patch X --target all --sheet A3
    physdecal export-print --scale-table

THE SCALE ARGUMENT, WHICH IS THE WHOLE REASON A MODEL CAR IS ALLOWED
--------------------------------------------------------------------
A 1.2 m decal on a real car photographed from 20 m, and the same decal printed
at 1/18 scale on a die-cast model photographed from 1.11 m, present the camera
with the same thing. Three quantities have to match and all three do, by
construction:

  patch to roof area ratio   scale cancels: both the decal and the roof are
                             divided by s, so the fraction of the roof covered
                             is identical
  angular subtense           a target of size L at distance d subtends L/d;
                             scaling both by s leaves it unchanged
  ground sample distance     pixels across the vehicle is L/(d*ifov), and
                             again s cancels

What does NOT scale, and must be reported rather than waved away:

  depth of field             a phone at 1.1 m has a much shallower depth of
                             field than a drone at 20 m. Stop down, and say so
  atmosphere                 there is none indoors. A real 60 m capture has
                             haze and the model car experiment cannot show it
  surface finish             a die-cast model is glossier than car paint, and
                             the specular behaviour of the roof around the
                             decal is therefore not faithful
  decal thickness            a sheet of paper at 1/18 scale is equivalent to a
                             4 cm thick slab on a real roof. It casts a shadow
                             no real vinyl would

These are the honest limits of a scaled physical experiment. They are printed
by --scale-table and they belong in the paper next to the physical numbers.

THE FIDUCIAL BORDER
-------------------
Four ArUco markers are printed around the patch at a known millimetre pitch.
They are not decoration. They give every photograph, taken freehand from an
unknown pose, an exact homography back to patch coordinates, which recovers:
the true metric scale, the viewing angle, and the exact pixel footprint of the
decal in the photo. Without them, "I photographed it from about 40 degrees"
is the resolution of the physical experiment. With them, the angle is measured
per frame and the physical results can be binned on the same axis as the
simulated ones.

Cut along the OUTER crop marks and the markers stay attached. That is
deliberate: they must remain visible in the photograph. Cut along the inner
line only after the capture session is finished, if you want the decal alone.
"""

import argparse
import json
import math
import os

import cv2
import numpy as np
import torch

from physdecal import config as N
from physdecal.evaluation import evaluate as E


MM_PER_INCH = 25.4


def mm_to_px(mm, dpi):
    return int(round(mm / MM_PER_INCH * dpi))


# ------------------------------------------------------------- scale algebra

def target_geometry(target=None, patch_size_m=None, size_mode=None):
    """
    Everything metric about one physical target, in one dict.

    Raises rather than guessing if the patch does not fit on the roof: a decal
    wider than the roof it is supposedly stuck to is not a scaled version of
    the simulated attack, and silently letting it overhang would make the
    physical and digital numbers incomparable.
    """
    target = target or N.PHYS_TARGET
    size_mode = size_mode or N.PATCH_SIZE_MODE
    if target not in N.PHYS_TARGETS:
        raise SystemExit("Unknown target '%s'. Known: %s"
                         % (target, ", ".join(N.PHYS_TARGETS)))
    s, veh_L, roof_L, roof_W, note = N.PHYS_TARGETS[target]

    if size_mode == "roof":
        # The decal is the largest square this target's roof panel admits,
        # which is exactly what mode "roof" optimised for. Nothing to check:
        # it fits by construction.
        p_m = min(roof_L, roof_W) * N.ROOF_FILL
    else:
        p_m = patch_size_m or N.PATCH_SIZE_M
        if p_m > min(roof_L, roof_W) + 1e-6:
            raise SystemExit("\n".join([
                "A %.2f m decal does not fit on a %.2f x %.2f m roof panel."
                % (p_m, roof_L, roof_W),
                "Three honest options, and widening PHYS_TARGETS to silence "
                "this check is not one of them:",
                "  - optimise with --size-mode roof, which sizes the decal per "
                "vehicle to the",
                "    largest square that fits and reports coverage fraction "
                "instead of metres",
                "  - reduce the size to at most %.2f m and re-optimise"
                % (min(roof_L, roof_W) * N.ROOF_FILL),
                "  - scope the claim to a larger-roofed vehicle, which for "
                "this dataset means",
                "    GP_Car_09_BoxTruck or GP_Car_10_TruckCab (2.13 m of roof)",
            ]))

    patch_mm = p_m * s * 1000.0
    dist_m = N.PHYS_REF_ALTITUDE_M * s

    # Size the marker from the camera, then cap it against the decal. Beyond
    # about a quarter of the decal the marker frame dominates the print and
    # stops being practical -- at 1:1 and 20 m the uncapped size is 258 mm
    # around a 1200 mm decal, which is a 1.8 m sheet of markers.
    want_mm = fiducial_mm_for(dist_m)
    cap_mm = 0.25 * patch_mm
    capped = want_mm > cap_mm
    fid_mm = min(want_mm, cap_mm)
    fid_px = expected_marker_px(fid_mm, dist_m)

    return {
        "target": target, "scale": s, "note": note,
        "size_mode": size_mode,
        "vehicle_len_m": veh_L, "roof_len_m": roof_L, "roof_wid_m": roof_W,
        "patch_size_m": p_m,
        "patch_mm": patch_mm,
        "model_vehicle_len_mm": veh_L * s * 1000.0,
        "patch_roof_area_frac": (p_m ** 2) / (roof_L * roof_W),
        "camera_distance_m": N.PHYS_REF_ALTITUDE_M * s,
        "ref_altitude_m": N.PHYS_REF_ALTITUDE_M,
        "fiducial_mm": fid_mm,
        "fiducial_px_expected": round(fid_px, 1),
        "fiducial_capped": capped,
    }


def fiducial_mm_for(distance_m, photo_long_side=None, fov_deg=None,
                    min_px=None):
    """
    How big the printed marker has to be to be decodable from distance_m.

    marker_px = f_px * marker_mm / distance_mm, so inverting for the pixel
    floor gives the millimetres. See the note in config.py: scaling the marker
    with the target instead of with the camera produces markers that are
    photographed at about nine pixels and never detected at all.
    """
    photo_long_side = photo_long_side or N.PHYS_PHOTO_LONG_SIDE
    fov_deg = fov_deg or N.PHYS_DEFAULT_FOV_DEG
    min_px = min_px or N.FIDUCIAL_MIN_PX
    f_px = 0.5 * photo_long_side / math.tan(math.radians(fov_deg) / 2.0)
    return min_px * (distance_m * 1000.0) / f_px


def expected_marker_px(marker_mm, distance_m, photo_long_side=None,
                       fov_deg=None):
    """How many pixels that marker will actually occupy in the photograph."""
    photo_long_side = photo_long_side or N.PHYS_PHOTO_LONG_SIDE
    fov_deg = fov_deg or N.PHYS_DEFAULT_FOV_DEG
    f_px = 0.5 * photo_long_side / math.tan(math.radians(fov_deg) / 2.0)
    return f_px * marker_mm / (distance_m * 1000.0)


def marker_warning(geo):
    """
    What to do when the capped marker is too small to be detected.

    Printed rather than raised, because none of the remedies is a failure:
    they are all legitimate capture choices, and which one is right depends on
    equipment this script cannot see.
    """
    return [
        "  !! MARKERS WILL PROBABLY NOT BE DETECTED at about %.0f px; ArUco "
        "wants roughly %d." % (geo["fiducial_px_expected"], N.FIDUCIAL_MIN_PX),
        "     The marker was capped at a quarter of the decal so the print "
        "stays practical.",
        "     Any of these fixes it, and all of them are legitimate:",
        "       - photograph from closer, and say so. The pose is measured "
        "per frame, so a",
        "         shorter distance changes which bin a row lands in, not "
        "whether it is valid.",
        "       - raise PHYS_PHOTO_LONG_SIDE if your camera shoots larger "
        "than %d px." % N.PHYS_PHOTO_LONG_SIDE,
        "       - print the marker frame as a SEPARATE sheet at full size and "
        "lay it around",
        "         the vehicle. The markers establish the ground plane, not "
        "the roof, so they",
        "         do not have to share a sheet with the decal.",
        "       - accept an unrecovered pose and report the physical rows "
        "without angle bins.",
    ]


def scale_table(patch_size_m=None, size_mode=None):
    """The metric equivalence table, printed and written to the print folder."""
    # A target the decal does not fit is reported as such, not raised. The
    # table exists to show WHICH targets a given decal can be printed for, and
    # aborting on the first mismatch hides every target that would have worked.
    rows, unfit = [], []
    for t in N.PHYS_TARGETS:
        try:
            rows.append(target_geometry(t, patch_size_m, size_mode))
        except SystemExit:
            _, _, rl, rw, note = N.PHYS_TARGETS[t]
            unfit.append((t, min(rl, rw) * N.ROOF_FILL, note))
    w = ("%-9s %7s %10s %10s %11s %11s %9s")
    lines = [w % ("target", "scale", "vehicle", "decal", "roof frac",
                  "camera at", "marker"),
             w % ("", "", "mm", "mm", "of roof", "m", "mm")]
    lines.append("-" * 72)
    for r in rows:
        lines.append(w % (r["target"], "1:%.0f" % (1 / r["scale"]),
                          "%.0f" % r["model_vehicle_len_mm"],
                          "%.1f" % r["patch_mm"],
                          "%.3f" % r["patch_roof_area_frac"],
                          "%.2f" % r["camera_distance_m"],
                          "%.1f" % r["fiducial_mm"]))
    lines.append("")
    lines.append("All three rows present the camera with the same angular")
    lines.append("geometry: the decal covers the same fraction of the roof and")
    lines.append("subtends the same angle. Camera distance is the scaled")
    lines.append("equivalent of %.0f m altitude." % N.PHYS_REF_ALTITUDE_M)
    lines.append("")
    lines.append("What does NOT scale, and must be stated with any physical")
    lines.append("number: depth of field, atmosphere, surface gloss, and the")
    lines.append("thickness of the printed sheet relative to the vehicle.")
    if unfit:
        lines.append("")
        lines.append("NOT PRINTABLE for this decal -- it exceeds the roof panel:")
        for t, mx, note in unfit:
            lines.append("  %-16s max %.2f m   (%s)" % (t, mx, note))
    return "\n".join(lines), rows


# ------------------------------------------------------------------ the sheet

def _put(img, text, org, scale=0.5, colour=(60, 60, 60), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def _crop_marks(sheet, x0, y0, x1, y1, dpi, colour=(120, 120, 120)):
    """L-shaped marks at the four corners, outside the cut line."""
    L = mm_to_px(5, dpi)
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                             (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(sheet, (cx, cy), (cx + dx * L, cy), colour, 1)
        cv2.line(sheet, (cx, cy), (cx, cy + dy * L), colour, 1)


def _ruler(sheet, x0, y, x1, dpi, colour=(120, 120, 120)):
    """A 10 mm-tick scale bar. The single cheapest check that the print came
    out at the right size: measure it before you stick anything down."""
    n = int((x1 - x0) / mm_to_px(10, dpi))
    cv2.line(sheet, (x0, y), (x0 + n * mm_to_px(10, dpi), y), colour, 1)
    for i in range(n + 1):
        x = x0 + i * mm_to_px(10, dpi)
        cv2.line(sheet, (x, y), (x, y - mm_to_px(2, dpi)), colour, 1)
    _put(sheet, "%d mm" % (n * 10), (x0, y + mm_to_px(4, dpi)), 0.42, colour)


def build_sheet_image(patch_rgb, geo, dpi=None, sheet=None, tag=""):
    """
    Compose the printable artwork: the decal at exact physical size, the
    fiducial border, crop marks, a ruler and a caption block.

    Returns (image BGR uint8, layout dict in millimetres).
    """
    dpi = dpi or N.PRINT_DPI
    sheet = sheet or N.SHEET
    if sheet not in N.SHEET_MM:
        raise SystemExit("Unknown sheet '%s'. Known: %s"
                         % (sheet, ", ".join(N.SHEET_MM)))

    patch_mm = geo["patch_mm"]
    fid_mm = geo["fiducial_mm"]
    gap_mm = max(3.0, fid_mm * 0.30)          # white quiet zone around markers
    border_mm = fid_mm + 2 * gap_mm
    art_mm = patch_mm + 2 * border_mm

    art_px = mm_to_px(art_mm, dpi)
    art = np.full((art_px, art_px, 3), 255, np.uint8)

    # --- the decal, resampled to exact physical size
    p_px = mm_to_px(patch_mm, dpi)
    # INTER_NEAREST on purpose: the patch is a flat-colour design and the
    # printed artefact should carry the ink boundaries the optimiser chose,
    # not a resampler's interpolation of them.
    decal = cv2.resize(patch_rgb, (p_px, p_px), interpolation=cv2.INTER_NEAREST)
    off = mm_to_px(border_mm, dpi)
    art[off:off + p_px, off:off + p_px] = cv2.cvtColor(decal, cv2.COLOR_RGB2BGR)

    # --- the cut line, exactly the decal boundary
    cv2.rectangle(art, (off - 1, off - 1), (off + p_px, off + p_px),
                  (170, 170, 170), 1)

    # --- fiducials, one per corner, ids 0..3 clockwise from top left
    ids_used = []
    if N.FIDUCIAL:
        adict = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, N.FIDUCIAL_DICT))
        f_px = mm_to_px(fid_mm, dpi)
        g_px = mm_to_px(gap_mm, dpi)
        corners = [(g_px, g_px), (art_px - g_px - f_px, g_px),
                   (art_px - g_px - f_px, art_px - g_px - f_px),
                   (g_px, art_px - g_px - f_px)]
        for i, (mx, my) in enumerate(corners):
            m = cv2.aruco.generateImageMarker(adict, i, f_px)
            art[my:my + f_px, mx:mx + f_px] = np.dstack([m] * 3)
            ids_used.append(i)

        # The marker CENTRE pitch is what capture.py solves the homography
        # against, so it is recorded rather than recomputed there.
        pitch_mm = art_mm - 2 * gap_mm - fid_mm
    else:
        pitch_mm = None

    # --- orientation arrow, in the quiet zone and NEVER on the decal.
    # The optimiser fixed the patch orientation in the vehicle frame: +y runs
    # nose to tail. A decal stuck on rotated 90 degrees is a different attack
    # from the one that was optimised, and the mistake is invisible in the
    # photograph afterwards. Drawing on the decal itself would change the
    # texture the optimiser produced, so the arrow lives in the border.
    ax = art_px // 2
    ay0 = mm_to_px(gap_mm * 0.5, dpi) + mm_to_px(fid_mm * 0.15, dpi)
    ay1 = off - mm_to_px(1.0, dpi)
    if ay1 > ay0 + 4:
        cv2.arrowedLine(art, (ax, ay1), (ax, ay0), (90, 90, 90),
                        max(1, mm_to_px(0.5, dpi)), tipLength=0.35)
        _put(art, "NOSE", (ax + mm_to_px(1.5, dpi), ay0 + mm_to_px(2.0, dpi)),
             0.36 * max(1.0, dpi / 300.0), (90, 90, 90))

    _crop_marks(art, 0, 0, art_px - 1, art_px - 1, dpi)
    return art, {
        "sheet": sheet, "dpi": dpi, "patch_mm": patch_mm,
        "art_mm": art_mm, "fiducial_mm": fid_mm, "quiet_zone_mm": gap_mm,
        "marker_pitch_mm": pitch_mm, "marker_ids": ids_used,
        "decal_offset_mm": border_mm,
    }


def tile_onto_sheets(art, layout, dpi=None, sheet=None):
    """
    Split the artwork across as many sheets as it needs.

    Overlap is real overlap, not a butt joint: TILE_OVERLAP_MM of the same
    pixels are printed on both sheets so there is something to align and glue.
    A butt joint leaves a white hairline that the camera sees as an edge, and
    an edge across a decal is a feature the optimiser never put there.
    """
    dpi = dpi or N.PRINT_DPI
    sheet = sheet or layout["sheet"]
    sw_mm, sh_mm = N.SHEET_MM[sheet]
    m = N.SHEET_MARGIN_MM
    usable_w = mm_to_px(sw_mm - 2 * m, dpi)
    usable_h = mm_to_px(sh_mm - 2 * m, dpi)
    ov = mm_to_px(N.TILE_OVERLAP_MM, dpi)

    H, W = art.shape[:2]
    step_w = max(usable_w - ov, 1)
    step_h = max(usable_h - ov, 1)
    nx = max(1, math.ceil(max(W - ov, 1) / step_w))
    ny = max(1, math.ceil(max(H - ov, 1) / step_h))

    sheets = []
    for ry in range(ny):
        for rx in range(nx):
            page = np.full((mm_to_px(sh_mm, dpi), mm_to_px(sw_mm, dpi), 3),
                           255, np.uint8)
            x0, y0 = rx * step_w, ry * step_h
            piece = art[y0:y0 + usable_h, x0:x0 + usable_w]
            px, py = mm_to_px(m, dpi), mm_to_px(m, dpi)
            page[py:py + piece.shape[0], px:px + piece.shape[1]] = piece
            sheets.append(((rx, ry), page, (x0, y0)))
    return sheets, (nx, ny)


def caption_block(page, geo, layout, tag, idx, total, dpi):
    """Identify the sheet on the sheet. A stack of anonymous printouts is
    unusable a week later, and an unidentified decal in a photograph cannot be
    matched back to the run that produced it."""
    if not N.PRINT_CAPTION:
        return
    h = page.shape[0]
    y = h - mm_to_px(6, dpi)
    txt = ("%s | %s 1:%.0f | decal %.1f mm | %d dpi | sheet %d/%d"
           % (tag, geo["target"], 1 / geo["scale"], geo["patch_mm"],
              dpi, idx, total))
    _put(page, txt, (mm_to_px(N.SHEET_MARGIN_MM, dpi), y), 0.45)
    _ruler(page, mm_to_px(N.SHEET_MARGIN_MM, dpi), y - mm_to_px(6, dpi),
           mm_to_px(N.SHEET_MARGIN_MM + 60, dpi), dpi)


def export(patch_tag, target=None, dpi=None, sheet=None, out_dir=None,
           device="cpu"):
    dpi = dpi or N.PRINT_DPI
    sheet = sheet or N.SHEET
    patch = E.load_patch(patch_tag, device)
    size_m = E.patch_size_of(patch_tag)
    geo = target_geometry(target, size_m, E.size_mode_of(patch_tag))

    rgb = (patch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    art, layout = build_sheet_image(rgb, geo, dpi, sheet, patch_tag)
    sheets, (nx, ny) = tile_onto_sheets(art, layout, dpi, sheet)

    out_dir = out_dir or os.path.join(N.PRINT_DIR,
                                      "%s__%s" % (patch_tag, geo["target"]))
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "artwork.png"), art)
    pages = []
    for i, ((rx, ry), page, origin) in enumerate(sheets, 1):
        caption_block(page, geo, layout, patch_tag, i, len(sheets), dpi)
        p = os.path.join(out_dir, "sheet_%02d.png" % i)
        cv2.imwrite(p, page)
        pages.append(p)

    # One PDF, page size exactly the sheet, so the printer is not asked to
    # scale. "Fit to page" is the single most common way a printed patch comes
    # out the wrong size, and the caption ruler is there to catch it.
    pdf = os.path.join(out_dir, "print_%s_%s.pdf" % (patch_tag, geo["target"]))
    _write_pdf(pages, pdf, sheet, dpi)

    meta = {"patch": patch_tag, "geometry": geo, "layout": layout,
            "sheets": len(pages), "grid": [nx, ny],
            "print_instructions": [
                "Print at 100 percent. Turn OFF 'fit to page' and 'scale to "
                "fit'; the decal must come out at exactly %.1f mm."
                % geo["patch_mm"],
                "Measure the ruler on the sheet before cutting. If it does "
                "not read the printed length, the scaling is wrong and every "
                "physical number will be wrong with it.",
                "Cut on the OUTER crop marks so the four ArUco markers stay "
                "attached. physdecal physical needs at least %d of them visible "
                "in each photograph to recover the pose." % N.PHYS_MIN_MARKERS,
                "Matte paper or matte vinyl. Gloss produces a specular "
                "highlight that saturates the sensor and destroys the decal "
                "in exactly the frames you most want.",
                "Stick it centred on the roof with the printed arrow pointing "
                "toward the vehicle nose, matching PATCH_CENTRE_UV and the "
                "vehicle-frame orientation the optimiser assumed.",
            ]}
    with open(os.path.join(out_dir, "print.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n%s -> %s" % (patch_tag, out_dir))
    print("  target        %s (1:%.0f)  %s"
          % (geo["target"], 1 / geo["scale"], geo["note"]))
    print("  decal         %.1f mm square, %.1f%% of the roof area"
          % (geo["patch_mm"], 100 * geo["patch_roof_area_frac"]))
    print("  artwork       %.1f mm square including the marker border"
          % layout["art_mm"])
    print("  markers       %d x %.1f mm, centre pitch %.1f mm, about %.0f px "
          "in the photo" % (len(layout["marker_ids"]), layout["fiducial_mm"],
                            layout["marker_pitch_mm"] or 0,
                            geo["fiducial_px_expected"]))
    if geo["fiducial_px_expected"] < N.FIDUCIAL_MIN_PX:
        for line in marker_warning(geo):
            print(line)
    print("  sheets        %d (%dx%d) on %s at %d dpi"
          % (len(pages), nx, ny, sheet, dpi))
    print("  photograph from %.2f m to match %.0f m altitude"
          % (geo["camera_distance_m"], geo["ref_altitude_m"]))
    print("  -> %s" % pdf)
    for line in meta["print_instructions"]:
        print("     * %s" % line)
    return out_dir


def _write_pdf(page_paths, pdf_path, sheet, dpi):
    """
    Multi-page PDF at exact physical page size.

    PIL rather than reportlab: it is already a dependency, and a PDF of
    full-page images at a declared resolution is all this needs. The
    resolution argument is what makes the page come out at its true size
    instead of at whatever the viewer assumes.
    """
    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in page_paths]
    imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:],
                 resolution=float(dpi))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch", default=None)
    ap.add_argument("--target", default=None,
                    help="diecast, rc, real, or 'all'")
    ap.add_argument("--sheet", default=None, choices=list(N.SHEET_MM))
    ap.add_argument("--dpi", type=int, default=None)
    ap.add_argument("--scale-table", action="store_true",
                    help="print the metric equivalence table and stop")
    a = ap.parse_args()

    os.makedirs(N.PRINT_DIR, exist_ok=True)
    if a.scale_table:
        size = E.patch_size_of(a.patch) if a.patch else N.PATCH_SIZE_M
        mode = E.size_mode_of(a.patch) if a.patch else None
        txt, _ = scale_table(size, mode)
        print(txt)
        p = os.path.join(N.PRINT_DIR, "scale_table.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        print("\nwrote %s" % p)
        return

    if not a.patch:
        raise SystemExit("--patch is required (or use --scale-table)")
    targets = (list(N.PHYS_TARGETS) if a.target == "all"
               else [a.target or N.PHYS_TARGET])
    for t in targets:
        export(a.patch, t, a.dpi, a.sheet)


if __name__ == "__main__":
    main()
