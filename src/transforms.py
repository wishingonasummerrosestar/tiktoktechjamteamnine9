"""
Platform redistribution simulator.

Models the damage an image accumulates as it travels across the internet:
uploaded, resized by a CDN, cropped for a thumbnail, screenshotted on a phone,
re-encoded by each platform it passes through.

Two things distinguish this from ordinary augmentation:

  1. Operations are CHAINED, not applied in isolation. A real image goes
     through a sequence, and the second operation's artifacts land on top of
     the first's.
  2. The ORDER is randomised. resize->jpeg leaves a different residue than
     jpeg->resize, because JPEG's block structure is imprinted on whatever
     the resize produced.

Severity values are taken verbatim from the challenge specification.

Used in three places:
  - training   (cache.py)    random chains, for robustness
  - evaluation (evaluate.py) isolated transforms, because that is what is graded
  - inference  (predict.py)  a fixed view set, for the robustness passport
"""

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# --------------------------------------------------------------------------
# Individual transforms. Each takes a PIL image plus a severity and returns a
# PIL image of the SAME dimensions, so operations can be chained freely.
# --------------------------------------------------------------------------

def t_jpeg(img, quality):
    """Lossy re-encode. Real-world analog: uploading to social media."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def t_blur(img, sigma):
    """Gaussian blur. Real-world analog: out-of-focus capture."""
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def t_resize(img, scale):
    """Downscale then upscale back. Real-world analog: thumbnail generation.

    The round trip is the point: information lost on the way down cannot be
    recovered on the way up, so the image keeps its original dimensions while
    losing high-frequency detail.
    """
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                       Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def t_noise(img, sigma, rng=None):
    """Additive Gaussian noise. Real-world analog: low-light sensor noise.

    Takes an optional numpy Generator so DataLoader workers do not all draw
    from the same global stream (see chain_with_params).
    """
    gen = rng if rng is not None else np.random.default_rng()
    a = np.asarray(img).astype(np.float32) / 255.0
    a = a + gen.normal(0, sigma, a.shape)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def t_jitter(img, amount, rng=None):
    """Brightness / contrast / saturation shift. Analog: filter apps."""
    r = rng if rng is not None else random
    for enhancer in (ImageEnhance.Brightness,
                     ImageEnhance.Contrast,
                     ImageEnhance.Color):
        img = enhancer(img).enhance(1.0 + r.uniform(-amount, amount))
    return img


def t_crop(img, frac):
    """Centre crop then restore size. Analog: profile-picture framing."""
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def t_screenshot(img, target_w):
    """Phone screenshot: rescale to device width, re-encode, restore size.

    Not in the official transform list, but it is the most common
    redistribution path on a mobile-first platform.
    """
    w, h = img.size
    nh = max(1, int(h * target_w / w))
    out = img.resize((target_w, nh), Image.BILINEAR)
    return t_jpeg(out, 85).resize((w, h), Image.BILINEAR)


# --------------------------------------------------------------------------
# Registry. Severities are exactly those specified by the challenge.
# --------------------------------------------------------------------------

OPS = {
    "jpeg":   (t_jpeg,       [90, 70, 50, 30]),
    "blur":   (t_blur,       [0.5, 1.0, 2.0]),
    "resize": (t_resize,     [0.5, 0.25]),
    "noise":  (t_noise,      [0.02, 0.05, 0.10]),
    "jitter": (t_jitter,     [0.2]),
    "crop":   (t_crop,       [0.8]),
    "screen": (t_screenshot, [1080]),
}

# Ops that accept an `rng` keyword for reproducible randomness.
_RNG_AWARE = {"noise", "jitter"}

# Conditions used for the graded evaluation: isolated transforms at the
# official severities. `screen` is excluded — it is a training-time and
# inference-time transform, not one of the specified evaluation conditions.
CONDITIONS = [("clean", None, None)] + [
    (f"{op}_{sev}", op, sev)
    for op, (_, sevs) in OPS.items() if op != "screen"
    for sev in sevs
]


# --------------------------------------------------------------------------
# Chaining
# --------------------------------------------------------------------------

def chain_with_params(img, rng, n_min=1, n_max=3):
    """Apply 1-3 randomly chosen ops in random order.

    `rng` must be a `random.Random` instance. Seed it per-item rather than
    sharing one object across DataLoader workers: forked workers inherit a
    copy of the generator state and would otherwise all produce identical
    degradations.

    Returns (degraded_image, {op_name: severity}). The parameter dict is kept
    so runs are reproducible and so degradation severity is available as a
    free training signal for any future degradation-aware component.
    """
    names = rng.sample(list(OPS.keys()), rng.randint(n_min, n_max))
    params = {}
    for name in names:
        fn, severities = OPS[name]
        sev = rng.choice(severities)
        if name in _RNG_AWARE:
            sub = np.random.default_rng(rng.randrange(2 ** 31))
            img = fn(img, sev, rng=sub) if name == "noise" else fn(img, sev, rng=rng)
        else:
            img = fn(img, sev)
        params[name] = sev
    return img, params


def random_chain(img, rng, n_min=1, n_max=3):
    """chain_with_params, discarding the parameter dict."""
    return chain_with_params(img, rng, n_min, n_max)[0]


# --------------------------------------------------------------------------
# Named real-world pipelines, for demonstration and error analysis
# --------------------------------------------------------------------------

PLATFORM_CHAINS = {
    "whatsapp":   [("resize", 0.5), ("jpeg", 50)],
    "instagram":  [("crop", 0.8), ("resize", 0.5), ("jpeg", 70)],
    "screenshot": [("screen", 1080), ("jpeg", 70)],
    "thumbnail":  [("resize", 0.25), ("jpeg", 90)],
}


def platform_chain(img, name):
    """Apply a named, fixed redistribution pipeline."""
    for op, sev in PLATFORM_CHAINS[name]:
        img = OPS[op][0](img, sev)
    return img


# --------------------------------------------------------------------------
# Robustness passport views
#
# FIXED, not random: the stability score compares the spread of predictions
# across views, so every image must be perturbed identically or the numbers
# are not comparable between images.
# --------------------------------------------------------------------------

PASSPORT_VIEWS = [
    ("clean",    lambda im: im),
    ("jpeg50",   lambda im: t_jpeg(im, 50)),
    ("blur1",    lambda im: t_blur(im, 1.0)),
    ("resize50", lambda im: t_resize(im, 0.5)),
    ("noise05",  lambda im: t_noise(im, 0.05, rng=np.random.default_rng(0))),
]
