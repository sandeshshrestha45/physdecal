"""
segmentation.py
One interface over six segmentation architectures and one vision-language
model, so the attack code never branches on which victim it is talking to.

Every model is wrapped to accept a batch of RGB images in [0,1] at capture
resolution and return

    logits   B x K x H x W    upsampled to the input size
    p_veh    B x 1 x H x W    probability that the pixel is a vehicle

p_veh is the only quantity the attack optimises against, and it is defined the
same way for every model: the summed posterior over that model's vehicle
classes. For CLIPSeg there is no class list, so it is the sigmoid of the
logit map for the text prompt.

NOTHING HERE IS TRAINED. Every checkpoint is used as published. If a model
fails the clean-IoU gate in physdecal check, the fallback is the ROI protocol
first and decoder-only finetuning second, in that order, and whichever you
used has to be stated next to the number.

Class indices are LOOKED UP BY NAME from each checkpoint's own id2label, never
hardcoded. Hardcoding ADE20K indices is how you end up reporting a rigorous
attack on the "wall" class.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from physdecal import config as N


# Substrings searched in each checkpoint's label names. Order does not matter.
VEHICLE_NAME_HINTS = ["car", "truck", "bus", "van", "minibike", "motorbike",
                      "motorcycle", "pickup"]
# Names that contain a hint but are not vehicles. ADE20K is full of these.
VEHICLE_NAME_BLOCK = ["carpet", "cart", "carport", "card", "cabinet",
                      "escalator", "caravan roof", "bus stop", "carousel",
                      "case", "cascade", "carriage return"]

PASCAL_VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
              "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
              "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
              "train", "tvmonitor"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _vehicle_indices(id2label):
    """Pick vehicle class indices by name. Prints what it chose, every time."""
    keep = []
    for i, name in sorted(id2label.items()):
        low = str(name).lower()
        if any(b in low for b in VEHICLE_NAME_BLOCK):
            continue
        first = low.split(",")[0].strip()
        if any(h == first or h in first.split() for h in VEHICLE_NAME_HINTS):
            keep.append((i, name))
    if not keep:
        raise SystemExit("No vehicle class found in this checkpoint's labels. "
                         "Print them with physdecal check --labels and add the "
                         "right name to VEHICLE_NAME_HINTS.")
    print("    vehicle classes: %s" % ", ".join("%d=%s" % (i, n) for i, n in keep))
    return [i for i, _ in keep]


def _round32(v):
    return int(max(32, round(v / 32.0) * 32))


class SegVictim(nn.Module):
    """Common wrapper. Subclasses only implement _raw_logits."""

    def __init__(self, key, mean, std, veh_idx):
        super().__init__()
        self.key = key
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.veh_idx = veh_idx

    # --- to be provided by subclasses
    def _raw_logits(self, x_norm, want_attn=False):
        raise NotImplementedError

    def resize_to_model(self, x01):
        """Resize a full-resolution batch to the model input grid."""
        h, w = x01.shape[-2:]
        s = N.INPUT_LONG_SIDE / float(max(h, w))
        return F.interpolate(x01, size=(_round32(h * s), _round32(w * s)),
                             mode="bilinear", align_corners=False)

    def forward(self, x01, out_size=None, want_attn=False, want_logits=False):
        """
        x01     B x 3 x H x W in [0,1], RGB, capture resolution
        returns dict(logits, p_veh, attn, logits_at)

        ORDER MATTERS AND IT IS A MEMORY DECISION, NOT A STYLE ONE.

        The vehicle posterior is reduced to ONE channel at MODEL resolution,
        and only that single channel is upsampled to the capture grid. The
        obvious alternative -- upsample the class logits first, then softmax --
        is mathematically identical and costs about twenty times the memory,
        because it holds a 150-channel tensor at 720 x 1280. At batch 4 that is
        roughly 1.1 GB for the tensor, again for the softmax, and again for the
        backward graph, which is most of a 6 GB card spent on nothing.

        Full-resolution class logits are produced only when want_logits is set.
        Grad-CAM needs them and runs at batch 1, where the cost is irrelevant.
        The targeted objective needs them too, which is one reason the default
        objective is 'vanish'.
        """
        out_size = out_size or x01.shape[-2:]
        xm = self.resize_to_model(x01)
        xn = (xm - self.mean) / self.std
        logits, attn = self._raw_logits(xn, want_attn)

        # m_veh is the vehicle log-odds, log p/(1-p), computed from the logits
        # rather than from p so it does not saturate. It is what the attack
        # optimises; p_veh is what evaluation thresholds. See seg_attack_loss.
        if self.veh_idx is None:                     # single-channel, CLIPSeg
            p = torch.sigmoid(logits)
            m = logits
        else:
            p = torch.softmax(logits, dim=1)[:, self.veh_idx].sum(1, keepdim=True)
            veh = torch.zeros(logits.shape[1], dtype=torch.bool,
                              device=logits.device)
            veh[self.veh_idx] = True
            m = (torch.logsumexp(logits[:, veh], dim=1, keepdim=True)
                 - torch.logsumexp(logits[:, ~veh], dim=1, keepdim=True))
        p = F.interpolate(p, size=out_size, mode="bilinear", align_corners=False)
        m = F.interpolate(m, size=out_size, mode="bilinear", align_corners=False)

        if want_logits:
            logits = F.interpolate(logits, size=out_size, mode="bilinear",
                                   align_corners=False)
        return {"logits": logits, "p_veh": p.clamp(1e-6, 1 - 1e-6),
                "m_veh": m,
                "attn": attn, "logits_at": "input" if want_logits else "model"}


class TorchvisionVictim(SegVictim):
    def __init__(self, key, name):
        import torchvision
        fn = getattr(torchvision.models.segmentation, name)
        net = fn(weights="DEFAULT").eval()
        id2label = {i: n for i, n in enumerate(PASCAL_VOC)}
        super().__init__(key, IMAGENET_MEAN, IMAGENET_STD,
                         _vehicle_indices(id2label))
        self.net = net

    def _raw_logits(self, x, want_attn=False):
        return self.net(x)["out"], None


class HFSegVictim(SegVictim):
    """SegFormer, UPerNet and DPT all follow the same call pattern."""

    def __init__(self, key, ckpt, family):
        from transformers import (SegformerForSemanticSegmentation,
                                  UperNetForSemanticSegmentation,
                                  DPTForSemanticSegmentation)
        cls = {"hf_segformer": SegformerForSemanticSegmentation,
               "hf_upernet": UperNetForSemanticSegmentation,
               "hf_dpt": DPTForSemanticSegmentation}[family]
        net = cls.from_pretrained(ckpt).eval()
        id2label = {int(k): v for k, v in net.config.id2label.items()}
        super().__init__(key, IMAGENET_MEAN, IMAGENET_STD,
                         _vehicle_indices(id2label))
        self.net = net
        self.family = family

    def _raw_logits(self, x, want_attn=False):
        kw = {}
        if want_attn:
            kw["output_attentions"] = True
        try:
            out = self.net(pixel_values=x, **kw)
        except (TypeError, ValueError):
            out = self.net(pixel_values=x)
        return out.logits, getattr(out, "attentions", None)


class ClipSegVictim(SegVictim):
    """
    Open-vocabulary arm. The 'class' is a sentence, which is why this model is
    interesting here: an attack that suppresses 'a car' can be re-tested
    against 'a parked vehicle seen from above' with no retraining, and that is
    a genuinely different generalisation question from the closed-set models.
    """

    def __init__(self, key, ckpt, prompt):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        self.proc = CLIPSegProcessor.from_pretrained(ckpt)
        net = CLIPSegForImageSegmentation.from_pretrained(ckpt).eval()
        super().__init__(key, CLIP_MEAN, CLIP_STD, None)
        self.net = net
        self.prompt = prompt
        tok = self.proc.tokenizer([prompt], return_tensors="pt", padding=True)
        self.register_buffer("input_ids", tok["input_ids"])
        self.register_buffer("attention_mask", tok["attention_mask"])

    def resize_to_model(self, x01):
        return F.interpolate(x01, size=(352, 352), mode="bilinear",
                             align_corners=False)

    def _raw_logits(self, x, want_attn=False):
        b = x.shape[0]
        out = self.net(input_ids=self.input_ids.expand(b, -1),
                       attention_mask=self.attention_mask.expand(b, -1),
                       pixel_values=x)
        lg = out.logits
        if lg.dim() == 3:
            lg = lg.unsqueeze(1)
        return lg, None

    def set_prompt(self, prompt):
        tok = self.proc.tokenizer([prompt], return_tensors="pt", padding=True)
        dev = self.input_ids.device
        self.input_ids = tok["input_ids"].to(dev)
        self.attention_mask = tok["attention_mask"].to(dev)
        self.prompt = prompt


# ------------------------------------------------- decoder-only finetuning
#
# Every checkpoint in the roster was trained on ground-level photographs and
# none of them can see a car from directly overhead: physdecal check --clean
# measures 0.000 clean IoU at nadir under both protocols. A model that cannot
# find the vehicle cannot be attacked, so the eligibility gate blocks the whole
# experiment until something is done about it.
#
# The something is decoder-only finetuning. The BACKBONE STAYS FROZEN, which
# is the point: the independent variable of this study is the backbone's
# attention scope, and finetuning the backbone would mean the ConvNeXt-vs-Swin
# comparison was partly measuring our own training run rather than the
# published architectures. Only the decode head moves, every arm gets the same
# treatment, and every number produced this way is labelled 'finetuned'.

BINARY_LABELS = ["background", "vehicle"]
VEHICLE_INDEX_BINARY = [1]


def decoder_modules(model):
    """The part that is allowed to move. Everything else is frozen."""
    net = model.net
    if hasattr(net, "decode_head"):                  # HF segformer / upernet
        return [net.decode_head]
    if hasattr(net, "classifier"):                   # torchvision fcn / dlv3
        return [net.classifier]
    raise SystemExit(
        "Do not know which module is the decoder for '%s'. Add it to "
        "segmentation.decoder_modules." % model.key)


def _replace_last_conv(module, out_ch):
    """Swap the final 1x1 classifier for one with out_ch channels."""
    if isinstance(getattr(module, "classifier", None), nn.Conv2d):
        old = module.classifier
        module.classifier = nn.Conv2d(old.in_channels, out_ch,
                                      kernel_size=old.kernel_size,
                                      stride=old.stride, padding=old.padding)
        return module.classifier
    if isinstance(module, nn.Sequential) and isinstance(module[-1], nn.Conv2d):
        old = module[-1]
        module[-1] = nn.Conv2d(old.in_channels, out_ch,
                               kernel_size=old.kernel_size,
                               stride=old.stride, padding=old.padding)
        return module[-1]
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise SystemExit("No Conv2d to replace in %s" % type(module).__name__)
    raise SystemExit(
        "Could not identify the final classifier in %s. Print the module tree "
        "with physdecal check --layers and extend _replace_last_conv."
        % type(module).__name__)


def make_binary(model):
    """
    Turn a K-class semantic head into a background/vehicle head.

    Two classes rather than a re-trained 150-class head, because the attack
    only ever consumes p_veh. It also removes the class-name lookup entirely,
    so there is no way to end up attacking 'wall' by accident.

    NOTE: this makes ATTACK_OBJECTIVE = 'targeted' unavailable, since there is
    no 'road' class left to push toward. The default objective is 'vanish',
    which is unaffected.
    """
    for mod in decoder_modules(model):
        _replace_last_conv(mod, len(BINARY_LABELS))
    if getattr(model.net, "aux_classifier", None) is not None:
        model.net.aux_classifier = None              # torchvision aux head
    model.veh_idx = list(VEHICLE_INDEX_BINARY)
    model.label_space = "binary"
    return model


def finetuned_path(key):
    return os.path.join(N.FINETUNE_DIR, "%s_decoder.pt" % key)


# ------------------------------------------------------------------ factory

_CACHE = {}


def load_model(key, device=None, cache=True, finetuned=None):
    ckey = (key, bool(N.USE_FINETUNED if finetuned is None else finetuned))
    if cache and ckey in _CACHE:
        return _CACHE[ckey]
    if key not in N.MODELS:
        raise SystemExit("Unknown model '%s'. Known: %s"
                         % (key, list(N.MODELS)))
    family, ckpt, scope, space = N.MODELS[key]
    print("loading %-22s  %-12s  attention=%s" % (key, family, scope))
    if family == "torchvision":
        m = TorchvisionVictim(key, ckpt)
    elif family in ("hf_segformer", "hf_upernet", "hf_dpt"):
        m = HFSegVictim(key, ckpt, family)
    elif family == "hf_clipseg":
        m = ClipSegVictim(key, ckpt, N.CLIPSEG_PROMPTS[0])
    else:
        raise SystemExit("Unknown family '%s'" % family)

    if finetuned is None:
        finetuned = N.USE_FINETUNED
    if finetuned and family == "hf_clipseg":
        # By design, not an oversight. CLIPSeg's class is a sentence, so
        # it is never finetuned; it runs on published weights even inside
        # a finetuned sweep. The results row still says weights=published,
        # so the two can never be confused in a table.
        print("    clipseg is the open-vocabulary arm and is never "
              "finetuned. Using published weights.")
        finetuned = False
    if finetuned:
        path = finetuned_path(key)
        if not os.path.isfile(path):
            raise SystemExit(
                "No %s.\nUSE_FINETUNED is on but this model has no finetuned "
                "decoder. Run\n  physdecal finetune --model %s"
                % (path, key))
        make_binary(m)
        state = torch.load(path, map_location="cpu")
        m.load_state_dict(state["decoder"], strict=False)
        print("    finetuned decoder loaded, val vehicle IoU %.3f at epoch %d"
              % (state.get("val_iou", float("nan")), state.get("epoch", -1)))

    m = m.to(device or N.DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)          # only the patch is a free variable
    m.attention_scope = scope
    if not finetuned:
        m.label_space = space
    m.finetuned = bool(finetuned)
    if cache:
        _CACHE[ckey] = m
    return m


def unload_all():
    _CACHE.clear()
    torch.cuda.empty_cache()


def get_module(model, dotted):
    """Resolve 'backbone.layer4' against a model. Used by Grad-CAM."""
    obj = model.net
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj
