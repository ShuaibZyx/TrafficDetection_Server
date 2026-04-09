
SECURITY_MAP = {
    1: 3,
    2: 3,
    3: 4,
    4: 5,
    5: 5,
    6: 5,
    7: 5,
}

SIZE_CONFIG = {
    1: dict(
        scale=1.5, L=(3.8, 5.2), W_ratio=0.4, W=(1.6, 2.0), H_ratio=0.32, H=(1.4, 1.6)
    ),
    2: dict(
        scale=1.5, L=(4.4, 5.8), W_ratio=0.41, W=(1.8, 2.2), H_ratio=0.38, H=(1.9, 2.4)
    ),
    3: dict(
        scale=2.2,
        L=(10.0, 14.0),
        W_ratio=0.21,
        W=(2.4, 2.7),
        H_ratio=0.25,
        H=(3.0, 3.5),
    ),
    4: dict(
        scale=1.2, L=(2.2, 3.2), W_ratio=0.5, W=(1.2, 1.5), H_ratio=0.55, H=(1.4, 1.8)
    ),
    5: dict(scale=1.0, L=(1.5, 2.0), W_ratio=0.45, W=(0.65, 0.9), H_mode="h"),
    6: dict(scale=1.0, L=(1.6, 2.0), W_ratio=0.42, W=(0.55, 0.8), H_mode="h"),
    7: dict(scale=1.0, L=(0.45, 0.7), W_mode="same_as_L", H_mode="h75"),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_size(tclass, w, h, y, img_h):
    cfg = SIZE_CONFIG.get(tclass)
    if not cfg:
        return (1.0, 1.0, 1.0)

    scale_base = 0.028 + (y + h * 0.5) / img_h * 0.045
    L = clamp(w * scale_base * cfg["scale"], *cfg["L"])

    if cfg.get("W_mode") == "same_as_L":
        W = clamp(L, *cfg["L"])
    else:
        W = clamp(L * cfg["W_ratio"], *cfg["W"])

    if cfg.get("H_mode") == "h":
        H = clamp(h * scale_base * 0.8, 1.6, 1.85)
    elif cfg.get("H_mode") == "h75":
        H = clamp(h * scale_base * 0.75, 1.55, 1.9)
    else:
        H = clamp(L * cfg["H_ratio"], *cfg["H"])

    return round(L, 2), round(W, 2), round(H, 2)


def compute_attributes(tlwh, tid, tclass, img_h, tid_cache):
    x, y, w, h = tlwh

    # ---------- security ----------
    security = SECURITY_MAP.get(tclass, -1)

    # ---------- size（缓存） ----------
    size = tid_cache.get(tid)
    if size is None:
        size = compute_size(tclass, w, h, y, img_h)
        tid_cache[tid] = size

    return security, size
