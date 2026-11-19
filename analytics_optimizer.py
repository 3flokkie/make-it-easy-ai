#!/usr/bin/env python3
"""
analytics_optimizer.py — self-learning YouTube Analytics feedback loop for Make It Easy AI.

Per channel token:
  1. Data API v3: uploads playlist → videos.list (snippet, statistics, contentDetails)
     + local attribution from live_clip_history.json / series_history.json / longform_history.json.
  2. Analytics API v2 (needs yt-analytics.readonly): views, engagedViews, averageViewDuration,
     averageViewPercentage, likes, comments, shares, subscribersGained per video.
     Old tokens without that scope → clean fallback to Data API statistics.
  3. Viral Score (0..100) = weighted percentile ranks (retention first, then engagement, subs, shares, views).
  4. EMA/bandit weight updates (clamped [0.2, 5.0]) for cover hooks + titles, optimal clip length,
     spike threshold; Gemini analyses top vs bottom and returns structured JSON recommendations.
  5. Writes scripts/optimizer_config.json (+ optimizer_summary.md). live_chat_clipper.py reads it.

Usage
  .venv/bin/python scripts/analytics_optimizer.py                     # all existing tokens
  .venv/bin/python scripts/analytics_optimizer.py --token token_channel2.pickle --days 14
  .venv/bin/python scripts/analytics_optimizer.py --dry-run           # compute + print, write nothing
  .venv/bin/python scripts/analytics_optimizer.py --mock-test         # offline self-test, no network

Enable real analytics on an old token: delete the token file and re-auth once
(uploader.SCOPES now includes yt-analytics.readonly).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

CONFIG_FILE = SCRIPT_DIR / "optimizer_config.json"
SUMMARY_FILE = SCRIPT_DIR / "optimizer_summary.md"
LIVE_HISTORY = SCRIPT_DIR / "live_clip_history.json"
SERIES_HISTORY = SCRIPT_DIR / "series_history.json"
LONGFORM_HISTORY = SCRIPT_DIR / "longform_history.json"

DEFAULT_TOKENS = ["token.pickle", "token_channel2.pickle", "token_en.pickle", "token_de.pickle"]
ANALYTICS_METRICS = ["views", "engagedViews", "averageViewDuration", "averageViewPercentage",
                     "likes", "comments", "shares", "subscribersGained"]
GEMINI_MODEL = "gemini-2.5-flash"

W_MIN, W_MAX = 0.2, 5.0          # clamp for every learned weight
EMA_ALPHA = 0.5                  # how fast weights move toward the new target
LEARN_RATE = 1.0                 # target = exp(lr * z * confidence)
DEFAULT_CLIP_SEC, CLIP_MIN, CLIP_MAX = 40, 15, 59
DEFAULT_SPIKE, SPIKE_MIN, SPIKE_MAX = 3.0, 2.0, 6.0

# Viral Score components: (feature, weight). Retention first, as requested.
SCORE_WEIGHTS_ANALYTICS = [("avp", 0.30), ("engaged_ratio", 0.20), ("subs_per_1k", 0.20),
                           ("shares_per_1k", 0.15), ("log_views", 0.15)]
SCORE_WEIGHTS_DATA_API = [("like_rate", 0.40), ("comment_rate", 0.20), ("log_views", 0.40)]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def iso_duration_sec(s: str | None) -> float | None:
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", s or "")
    if not m or not any(m.groups()):
        return None
    d, h, mi, se = (float(x) if x else 0.0 for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def base_title(title: str) -> str:
    """Strip the '(spoken words)' insert the clipper adds, so titles map back to VIRAL_TITLES."""
    return re.sub(r"\s+", " ", re.sub(r"\s*\([^)]*\)\s*(?=#shorts)", " ", title or "")).strip()


def clipper_catalog() -> tuple[list[str], list[str]]:
    try:
        import live_chat_clipper as lcc
        return list(lcc.VIRAL_COVER_HOOKS), list(lcc.VIRAL_TITLES)
    except Exception:  # noqa: BLE001
        return [], []


# ─────────────────────────── fetch ───────────────────────────

def fetch_channel_videos(yt, max_videos: int = 100) -> tuple[dict, list[dict]]:
    ch = yt.channels().list(part="snippet,contentDetails", mine=True).execute()
    item = ch["items"][0]
    uploads = item["contentDetails"]["relatedPlaylists"]["uploads"]
    ids: list[str] = []
    page = None
    while len(ids) < max_videos:
        res = yt.playlistItems().list(part="contentDetails", playlistId=uploads, maxResults=50,
                                      pageToken=page).execute()
        ids += [i["contentDetails"]["videoId"] for i in res.get("items", [])]
        page = res.get("nextPageToken")
        if not page:
            break
    videos: list[dict] = []
    for i in range(0, len(ids[:max_videos]), 50):
        res = yt.videos().list(part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50])).execute()
        for v in res.get("items", []):
            st = v.get("statistics", {})
            videos.append({
                "id": v["id"],
                "title": v["snippet"].get("title", ""),
                "published_at": v["snippet"].get("publishedAt"),
                "duration_sec": iso_duration_sec(v.get("contentDetails", {}).get("duration")),
                "views": int(st.get("viewCount", 0) or 0),
                "likes": int(st.get("likeCount", 0) or 0),
                "comments": int(st.get("commentCount", 0) or 0),
            })
    return {"id": item["id"], "title": item["snippet"].get("title", "")}, videos


def fetch_analytics(yta, days: int) -> dict[str, dict]:
    end = date.today()
    start = end - timedelta(days=days)
    metrics = list(ANALYTICS_METRICS)
    for attempt in range(2):
        try:
            res = yta.reports().query(ids="channel==MINE", startDate=start.isoformat(), endDate=end.isoformat(),
                                      dimensions="video", metrics=",".join(metrics), sort="-views",
                                      maxResults=50).execute()
            break
        except Exception as e:  # noqa: BLE001 — engagedViews is newer; retry without it once
            if attempt == 0 and "engagedViews" in metrics:
                log(f"⚠️  analytics query failed ({str(e)[:120]}) — retrying without engagedViews")
                metrics.remove("engagedViews")
                continue
            raise
    cols = [h["name"] for h in res.get("columnHeaders", [])]
    out: dict[str, dict] = {}
    for row in res.get("rows", []) or []:
        rec = dict(zip(cols, row))
        out[rec.pop("video")] = rec
    return out


def load_local_metadata() -> dict:
    live = _read_json(LIVE_HISTORY, [])
    by_vid = {}
    for e in live if isinstance(live, list) else []:
        vid = e.get("uploaded_id") if isinstance(e, dict) else None
        if vid:
            by_vid[vid] = e
    series = _read_json(SERIES_HISTORY, {})
    longf = _read_json(LONGFORM_HISTORY, {})
    return {
        "live": by_vid,
        "series_titles": set((series or {}).get("used_titles", []) if isinstance(series, dict) else []),
        "longform_titles": set((longf or {}).get("used_titles", []) if isinstance(longf, dict) else []),
    }


def merge_metadata(videos: list[dict], meta: dict, analytics: dict[str, dict] | None) -> list[dict]:
    titles = {base_title(t) for t in clipper_catalog()[1]}
    for v in videos:
        h = meta["live"].get(v["id"])
        if h:
            v["source"] = "live_clip"
            v["hook_headline"] = h.get("hook_headline")
            v["clip_title"] = base_title(h.get("title") or v["title"])
            v["spike_ratio"] = h.get("spike_ratio") or (
                round(h["window_rate"] / h["baseline_rate"], 2) if h.get("baseline_rate") else None)
            v["clip_sec"] = h.get("clip_sec") or v.get("duration_sec")
        elif any(t and t in v["title"] for t in meta["series_titles"]):
            v["source"] = "series"
        elif any(t and t in v["title"] for t in meta["longform_titles"]):
            v["source"] = "longform"
        else:
            v["source"] = "other"
        if not v.get("clip_title") and base_title(v["title"]) in titles:
            v["clip_title"] = base_title(v["title"])  # older clips without a history title
        if analytics and v["id"] in analytics:
            v.update({k: analytics[v["id"]][k] for k in analytics[v["id"]]})
            v["has_analytics"] = True
    return videos


# ─────────────────────────── scoring ───────────────────────────

def _features(v: dict, use_analytics: bool) -> dict[str, float]:
    views = float(v.get("views") or 0)
    f = {"log_views": math.log10(1 + views)}
    if use_analytics:
        f["avp"] = float(v.get("averageViewPercentage") or 0)
        ev = v.get("engagedViews")
        f["engaged_ratio"] = float(ev) / views if ev is not None and views > 0 else 0.0
        f["subs_per_1k"] = 1000 * float(v.get("subscribersGained") or 0) / max(views, 1)
        f["shares_per_1k"] = 1000 * float(v.get("shares") or 0) / max(views, 1)
    else:
        f["like_rate"] = float(v.get("likes") or 0) / max(views, 1)
        f["comment_rate"] = float(v.get("comments") or 0) / max(views, 1)
    return f


def _pct_rank(values: list[float]) -> list[float]:
    """Average-rank percentile in [0,1] (ties share a rank) — robust to outliers and scale."""
    n = len(values)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = ((i + j) / 2) / (n - 1)
        i = j + 1
    return ranks


def compute_viral_scores(videos: list[dict], use_analytics: bool) -> list[dict]:
    """Adds v['viral_score'] (0..100) and v['features']; returns videos sorted best→worst."""
    if not videos:
        return []
    spec = SCORE_WEIGHTS_ANALYTICS if use_analytics else SCORE_WEIGHTS_DATA_API
    feats = [_features(v, use_analytics) for v in videos]
    ranks = {name: _pct_rank([f[name] for f in feats]) for name, _ in spec}
    total_w = sum(w for _, w in spec)
    for i, v in enumerate(videos):
        v["features"] = {k: round(x, 4) for k, x in feats[i].items()}
        v["viral_score"] = round(100 * sum(w * ranks[name][i] for name, w in spec) / total_w, 2)
    return sorted(videos, key=lambda x: x["viral_score"], reverse=True)


# ─────────────────────────── learning ───────────────────────────

def update_arm_weights(prev: dict[str, float], arms: list[str], scored: list[dict], key: str) -> dict[str, dict]:
    """EMA/bandit update. For each arm: z = (mean score of its videos − channel mean)/std,
    target = exp(lr·z·confidence), w' = clamp((1−α)·w + α·target). The target is absolute (not w·…), so
    re-counting the same 28-day evidence on daily runs converges instead of compounding to the clamp.
    Unseen arms drift 10% back to 1.0."""
    scores = [v["viral_score"] for v in scored]
    mu = sum(scores) / len(scores) if scores else 50.0
    sd = math.sqrt(sum((s - mu) ** 2 for s in scores) / len(scores)) if len(scores) > 1 else 0.0
    sd = max(sd, 5.0)
    out: dict[str, dict] = {}
    for arm in dict.fromkeys([*arms, *prev.keys()]):
        w = float(prev.get(arm, 1.0)) if isinstance(prev.get(arm, 1.0), (int, float)) else 1.0
        hits = [v["viral_score"] for v in scored if v.get(key) == arm]
        if hits:
            n = len(hits)
            z = (sum(hits) / n - mu) / sd
            conf = n / (n + 2)  # few samples → small step
            target = math.exp(LEARN_RATE * z * conf)
            w_new = (1 - EMA_ALPHA) * w + EMA_ALPHA * target
            out[arm] = {"weight": round(clamp(w_new, W_MIN, W_MAX), 4), "n": n,
                        "mean_score": round(sum(hits) / n, 2), "z": round(z, 3)}
        else:
            out[arm] = {"weight": round(clamp(w + 0.1 * (1.0 - w), W_MIN, W_MAX), 4), "n": 0,
                        "mean_score": None, "z": None}
    return out


def _ema(prev: float, target: float, alpha: float = 0.3) -> float:
    return (1 - alpha) * prev + alpha * target


def recommend_clip_sec(scored: list[dict], prev: float) -> float:
    pts = [(v["viral_score"], float(v.get("clip_sec") or v.get("duration_sec") or 0)) for v in scored]
    pts = [(s, d) for s, d in pts if 5 <= d <= 180]
    if len(pts) < 3:
        return prev
    top = pts[: max(2, len(pts) // 3)]  # scored is sorted best→worst
    target = sum(s * d for s, d in top) / max(sum(s for s, _ in top), 1e-6)
    return round(clamp(_ema(prev, target), CLIP_MIN, CLIP_MAX), 1)


def recommend_spike_mult(scored: list[dict], prev: float) -> float:
    pts = [(v["viral_score"], float(v["spike_ratio"])) for v in scored if v.get("spike_ratio")]
    if len(pts) < 3:
        return prev
    top = sorted(r for _, r in pts[: max(2, len(pts) // 2)])
    q25 = top[int(0.25 * (len(top) - 1))]  # weakest spike that still produced a top clip
    return round(clamp(_ema(prev, q25 * 0.6), SPIKE_MIN, SPIKE_MAX), 2)


# ─────────────────────────── Gemini ───────────────────────────

def _brief(v: dict) -> dict:
    keys = ["title", "source", "hook_headline", "clip_title", "duration_sec", "viral_score", "views",
            "averageViewPercentage", "engagedViews", "subscribersGained", "shares", "likes", "comments", "spike_ratio"]
    return {k: v.get(k) for k in keys if v.get(k) is not None}


def gemini_prompt(channel: str, top: list[dict], bottom: list[dict], hooks: dict, titles: dict) -> str:
    return (
        "Ты аналитик YouTube Shorts для канала " f"«{channel}». Сравни лучшие и худшие ролики по Viral Score "
        "(досмотр важнее всего, затем engagedViews/views, подписки, репосты, просмотры).\n"
        f"TOP: {json.dumps([_brief(v) for v in top], ensure_ascii=False)}\n"
        f"BOTTOM: {json.dumps([_brief(v) for v in bottom], ensure_ascii=False)}\n"
        f"Текущие веса обложек: {json.dumps(hooks, ensure_ascii=False)}\n"
        f"Текущие веса заголовков: {json.dumps(titles, ensure_ascii=False)}\n"
        "Верни СТРОГО JSON без markdown:\n"
        '{"insights": [str], "hook_adjustments": {"<точный текст обложки>": float 0.5-1.5}, '
        '"title_adjustments": {"<точный текст заголовка>": float 0.5-1.5}, '
        '"recommended_clip_sec": int, "recommended_spike_mult": float, '
        '"recommended_topics": [str], "avoid_patterns": [str], '
        '"new_hook_ideas": [str до 25 символов, CAPS], "new_title_ideas": [str до 90 символов, с #shorts]}'
    )


def ask_gemini(prompt: str) -> dict | None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        log("⚠️  GEMINI_API_KEY missing — skipping Gemini analysis")
        return None
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        res = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4),
        )
        return parse_gemini_json(res.text or "")
    except Exception as e:  # noqa: BLE001
        log(f"⚠️  Gemini analysis failed: {str(e)[:160]}")
        return None


def parse_gemini_json(text: str) -> dict | None:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def apply_gemini_adjustments(arms: dict[str, dict], adj) -> None:
    """Gemini nudges are soft: multiplier clamped to [0.7, 1.3], result re-clamped to [W_MIN, W_MAX]."""
    if not isinstance(adj, dict):
        return
    for text, mult in adj.items():
        if text in arms and isinstance(mult, (int, float)) and math.isfinite(mult):
            arms[text]["weight"] = round(clamp(arms[text]["weight"] * clamp(float(mult), 0.7, 1.3), W_MIN, W_MAX), 4)
            arms[text]["gemini_mult"] = round(float(mult), 3)


def _str_list(x, limit: int = 10) -> list[str]:
    return [str(i).strip() for i in x if str(i).strip()][:limit] if isinstance(x, list) else []


# ─────────────────────────── per-channel pipeline ───────────────────────────

def optimize_channel(token: str, channel_title: str, videos: list[dict], data_source: str, prev: dict,
                     gemini_fn=ask_gemini) -> dict:
    use_analytics = data_source == "analytics"
    scored = compute_viral_scores([v for v in videos if (v.get("views") or 0) > 0] or videos, use_analytics)
    hooks_cat, titles_cat = clipper_catalog()
    prev_w = prev.get("weights", {}) if isinstance(prev, dict) else {}
    hook_arms = update_arm_weights(prev_w.get("cover_hooks", {}) or {}, hooks_cat, scored, "hook_headline")
    title_arms = update_arm_weights(prev_w.get("titles", {}) or {}, titles_cat, scored, "clip_title")

    prev_rec = prev.get("recommended", {}) if isinstance(prev, dict) else {}
    clip_sec = recommend_clip_sec(scored, float(prev_rec.get("clip_sec", DEFAULT_CLIP_SEC)))
    spike = recommend_spike_mult(scored, float(prev_rec.get("spike_mult", DEFAULT_SPIKE)))

    k = max(1, min(5, len(scored) // 3)) if scored else 0
    top, bottom = scored[:k], scored[-k:] if k else []
    gem = None
    if scored and gemini_fn:
        gem = gemini_fn(gemini_prompt(channel_title, top, bottom,
                                      {a: d["weight"] for a, d in hook_arms.items()},
                                      {a: d["weight"] for a, d in title_arms.items()}))
    gem = gem if isinstance(gem, dict) else {}
    apply_gemini_adjustments(hook_arms, gem.get("hook_adjustments"))
    apply_gemini_adjustments(title_arms, gem.get("title_adjustments"))
    if isinstance(gem.get("recommended_clip_sec"), (int, float)):
        clip_sec = round(clamp((clip_sec + float(gem["recommended_clip_sec"])) / 2, CLIP_MIN, CLIP_MAX), 1)
    if isinstance(gem.get("recommended_spike_mult"), (int, float)):
        spike = round(clamp((spike + float(gem["recommended_spike_mult"])) / 2, SPIKE_MIN, SPIKE_MAX), 2)

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channel_title": channel_title,
        "data_source": data_source,
        "videos_analyzed": len(scored),
        "runs": int(prev.get("runs", 0) if isinstance(prev, dict) else 0) + 1,
        "weights": {"cover_hooks": {a: d["weight"] for a, d in hook_arms.items()},
                    "titles": {a: d["weight"] for a, d in title_arms.items()}},
        "arm_stats": {"cover_hooks": hook_arms, "titles": title_arms},
        "recommended": {
            "clip_sec": clip_sec,
            "spike_mult": spike,
            "topics": _str_list(gem.get("recommended_topics")),
            "avoid_patterns": _str_list(gem.get("avoid_patterns")),
            "new_hook_ideas": _str_list(gem.get("new_hook_ideas")),
            "new_title_ideas": _str_list(gem.get("new_title_ideas")),
        },
        "insights": _str_list(gem.get("insights")),
        "top_videos": [{**_brief(v), "id": v["id"]} for v in top],
        "bottom_videos": [{**_brief(v), "id": v["id"]} for v in bottom],
    }


def build_global(channels: dict) -> dict:
    """Cross-channel weights = geometric mean of channel weights (used when the clipper token is unknown)."""
    agg: dict[str, dict[str, list[float]]] = {"cover_hooks": {}, "titles": {}}
    for ch in channels.values():
        for kind in agg:
            for text, w in ((ch.get("weights") or {}).get(kind) or {}).items():
                agg[kind].setdefault(text, []).append(float(w))
    return {"weights": {kind: {t: round(math.exp(sum(map(math.log, ws)) / len(ws)), 4) for t, ws in d.items()}
                        for kind, d in agg.items()}}


def render_summary(cfg: dict) -> str:
    lines = [f"# Analytics Optimizer — {cfg.get('updated_at')}", ""]
    for token, ch in cfg.get("channels", {}).items():
        r = ch.get("recommended", {})
        lines += [f"## {ch.get('channel_title') or token} (`{token}`)",
                  f"- source: **{ch.get('data_source')}**, videos: {ch.get('videos_analyzed')}, runs: {ch.get('runs')}",
                  f"- clip_sec: **{r.get('clip_sec')}**, spike_mult: **{r.get('spike_mult')}**", "",
                  "| kind | text | weight | n | mean score |", "|---|---|---|---|---|"]
        for kind in ("cover_hooks", "titles"):
            for text, st in sorted(ch.get("arm_stats", {}).get(kind, {}).items(), key=lambda x: -x[1]["weight"]):
                lines.append(f"| {kind} | {text} | {st['weight']} | {st['n']} | {st['mean_score']} |")
        lines.append("")
        for title, key in (("Top", "top_videos"), ("Bottom", "bottom_videos")):
            lines.append(f"**{title}:** " + "; ".join(f"{v.get('title', v['id'])[:50]} ({v.get('viral_score')})"
                                                      for v in ch.get(key, [])))
        for title, key in (("Insights", None), ("Topics", "topics"), ("Avoid", "avoid_patterns"),
                           ("New hooks", "new_hook_ideas"), ("New titles", "new_title_ideas")):
            items = ch.get("insights", []) if key is None else r.get(key, [])
            if items:
                lines += ["", f"**{title}:**", *[f"- {i}" for i in items]]
        lines.append("")
    return "\n".join(lines) + "\n"


def save_config(cfg: dict, config_path: Path, summary_path: Path) -> None:
    cfg["global"] = build_global(cfg.get("channels", {}))
    tmp = config_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(config_path)  # atomic: the clipper never sees a half-written file
    summary_path.write_text(render_summary(cfg), encoding="utf-8")


def load_config(path: Path) -> dict:
    cfg = _read_json(path, {})
    if not isinstance(cfg, dict) or not isinstance(cfg.get("channels"), dict):
        cfg = {"version": 1, "channels": {}}
    return cfg


def run_token(token: str, days: int, prev: dict, meta: dict) -> dict | None:
    from uploader import ScopeMissingError, _token_path, get_analytics_service, get_authenticated_service

    if not _token_path(token).exists():
        log(f"⏭️  {token}: not found — skipping (no interactive auth from the optimizer)")
        return None
    try:
        yt = get_authenticated_service(token)
        channel, videos = fetch_channel_videos(yt)
    except Exception as e:  # noqa: BLE001
        log(f"⚠️  {token}: Data API failed: {str(e)[:200]}")
        return None
    log(f"📺 {token}: {channel['title']} — {len(videos)} recent uploads")

    analytics, source = None, "data_api"
    try:
        analytics = fetch_analytics(get_analytics_service(token), days)
        source = "analytics"
        log(f"📈 {token}: Analytics API rows: {len(analytics)}")
    except ScopeMissingError as e:
        log(f"ℹ️  {e}\n    → falling back to Data API v3 statistics (views/likes/comments)")
    except Exception as e:  # noqa: BLE001
        log(f"⚠️  {token}: Analytics API failed ({str(e)[:160]}) → Data API fallback")

    videos = merge_metadata(videos, meta, analytics)
    if source == "analytics":
        videos = [v for v in videos if v.get("has_analytics")]
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = [v for v in videos if v.get("published_at") and
                  datetime.fromisoformat(v["published_at"].replace("Z", "+00:00")) >= cutoff]
        videos = recent if len(recent) >= 10 else videos[:max(10, len(recent))]
    if not videos:
        log(f"⚠️  {token}: no videos with data in the last {days} days")
        return None
    return optimize_channel(token, channel["title"], videos, source, prev)


# ─────────────────────────── mock test ───────────────────────────

def _mock_dataset(rng: random.Random) -> tuple[list[dict], dict]:
    hooks, titles = clipper_catalog()
    assert hooks and titles, "could not import VIRAL_COVER_HOOKS / VIRAL_TITLES from live_chat_clipper"
    quality_h = {hooks[0]: 0.9, hooks[1]: 0.6, hooks[2]: 0.45, hooks[-1]: 0.1}  # hooks[0] best, hooks[-1] worst
    quality_t = {titles[0]: 0.85, titles[1]: 0.55, titles[2]: 0.4, titles[-1]: 0.15}
    videos, live = [], {}
    for i in range(24):
        h, t = hooks[i % len(hooks)], titles[(i // 2) % len(titles)]
        q = 0.6 * quality_h[h] + 0.4 * quality_t[t] + rng.uniform(-0.05, 0.05)
        views = int(500 + 20000 * q)
        vid = f"mock{i:03d}"
        dur = 25 + 20 * (1 - q) + rng.uniform(-2, 2)  # shorter clips retain better in this world
        videos.append({"id": vid, "title": t.replace("#shorts", "(lol) #shorts"), "duration_sec": dur,
                       "published_at": datetime.now(timezone.utc).isoformat(), "views": views,
                       "likes": int(views * (0.02 + 0.06 * q)), "comments": int(views * 0.004 * q)})
        live[vid] = {"uploaded_id": vid, "hook_headline": h, "title": t.replace("#shorts", "(lol) #shorts"),
                     "window_rate": 1.0 + 2 * q, "baseline_rate": 0.2, "clip_sec": round(dur, 1)}
        live[vid]["_analytics"] = {"views": views, "engagedViews": int(views * (0.5 + 0.45 * q)),
                                   "averageViewDuration": dur * (0.4 + 0.6 * q),
                                   "averageViewPercentage": 30 + 70 * q, "likes": videos[-1]["likes"],
                                   "comments": videos[-1]["comments"], "shares": int(views * 0.01 * q),
                                   "subscribersGained": int(views * 0.003 * q)}
    analytics = {vid: e.pop("_analytics") for vid, e in live.items()}
    meta = {"live": live, "series_titles": set(), "longform_titles": set()}
    return merge_metadata(videos, meta, analytics), {"hooks": hooks, "titles": titles}


def mock_test() -> int:
    import live_chat_clipper as lcc

    rng = random.Random(42)
    checks: list[tuple[str, bool]] = []
    ok = lambda name, cond: checks.append((name, bool(cond)))  # noqa: E731

    videos, cat = _mock_dataset(rng)
    hooks, titles = cat["hooks"], cat["titles"]

    # 1) scoring
    scored = compute_viral_scores([dict(v) for v in videos], use_analytics=True)
    ok("scores in [0,100]", all(0 <= v["viral_score"] <= 100 for v in scored))
    ok("sorted best→worst", all(scored[i]["viral_score"] >= scored[i + 1]["viral_score"] for i in range(len(scored) - 1)))
    ok("best video uses best hook", scored[0]["hook_headline"] == hooks[0])
    ok("worst video uses worst hook", scored[-1]["hook_headline"] == hooks[-1])
    ok("title attribution strips '(spoken)'", all(v.get("clip_title") in titles for v in videos))
    fb = compute_viral_scores([{k: x for k, x in v.items() if k not in ANALYTICS_METRICS or k in
                                ("views", "likes", "comments")} for v in videos], use_analytics=False)
    ok("Data-API fallback scoring works", len(fb) == len(videos) and fb[0]["viral_score"] > fb[-1]["viral_score"])

    # 2) weight updates via full pipeline with a fake Gemini
    fake_gem = {"insights": ["Short punchy hooks retain best"], "hook_adjustments": {hooks[0]: 1.2, "UNKNOWN": 9},
                "title_adjustments": {titles[-1]: 0.8}, "recommended_clip_sec": 30, "recommended_spike_mult": 3.5,
                "recommended_topics": ["police chases"], "avoid_patterns": ["slow intros"],
                "new_hook_ideas": ["HE ACTUALLY DID IT"], "new_title_ideas": ["GTA 6 IMPOSSIBLE ESCAPE 😱 #shorts"]}
    prompts: list[str] = []
    gem_fn = lambda p: (prompts.append(p), json.loads(json.dumps(fake_gem)))[1]  # noqa: E731
    ch = optimize_channel("token_mock.pickle", "Mock Channel", [dict(v) for v in videos], "analytics", {},
                          gemini_fn=gem_fn)
    hw, tw = ch["weights"]["cover_hooks"], ch["weights"]["titles"]
    ok("Gemini prompt built with TOP/BOTTOM", prompts and "TOP:" in prompts[0] and "BOTTOM:" in prompts[0])
    ok("best hook weight > 1", hw[hooks[0]] > 1.0)
    ok("worst hook weight < 1", hw[hooks[-1]] < 1.0)
    ok("best title weight > worst title", tw[titles[0]] > tw[titles[-1]])
    ok("unknown Gemini arm ignored", "UNKNOWN" not in hw)
    ok("clip_sec within bounds", CLIP_MIN <= ch["recommended"]["clip_sec"] <= CLIP_MAX)
    ok("spike_mult within bounds", SPIKE_MIN <= ch["recommended"]["spike_mult"] <= SPIKE_MAX)
    ok("Gemini topics/avoid stored", ch["recommended"]["topics"] and ch["recommended"]["avoid_patterns"])

    # repeated runs: EMA compounding must stay inside the clamp
    prev, before = ch, ch
    for _ in range(30):
        before = prev
        prev = optimize_channel("token_mock.pickle", "Mock Channel", [dict(v) for v in videos], "analytics",
                                prev, gemini_fn=gem_fn)
    allw = [*prev["weights"]["cover_hooks"].values(), *prev["weights"]["titles"].values()]
    ok(f"weights clamped to [{W_MIN}, {W_MAX}] after 31 runs", all(W_MIN <= w <= W_MAX for w in allw))
    ok("repeated identical evidence converges (no compounding)",
       all(abs(prev["weights"]["cover_hooks"][h] - before["weights"]["cover_hooks"][h]) < 0.01 for h in hooks))
    ok("hook ranking preserved after convergence",
       [prev["weights"]["cover_hooks"][h] for h in hooks] == sorted((prev["weights"]["cover_hooks"][h] for h in hooks), reverse=True))
    # extreme evidence must still be clamped
    ext = update_arm_weights({hooks[0]: 4.9}, hooks, [{"viral_score": 100, "hook_headline": hooks[0]}] * 50
                             + [{"viral_score": 0, "hook_headline": hooks[-1]}] * 50, "hook_headline")
    ok("extreme evidence clamped", all(W_MIN <= d["weight"] <= W_MAX for d in ext.values()))
    ok("run counter increments", prev["runs"] == 31)

    # 3) config generation (temp dir, real optimizer_config.json untouched)
    with tempfile.TemporaryDirectory() as td:
        cfg_path, sum_path = Path(td) / "optimizer_config.json", Path(td) / "optimizer_summary.md"
        cfg = {"version": 1, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "channels": {"token_mock.pickle": prev}}
        save_config(cfg, cfg_path, sum_path)
        loaded = json.loads(cfg_path.read_text(encoding="utf-8"))
        ok("optimizer_config.json written + valid JSON", loaded["channels"]["token_mock.pickle"]["weights"])
        ok("global weights block present", loaded["global"]["weights"]["cover_hooks"])
        ok("summary markdown written", "Mock Channel" in sum_path.read_text(encoding="utf-8"))

        # 4) live_chat_clipper weighted selection
        r = random.Random(7)
        picks = [lcc.weighted_pick(hooks, "cover_hooks", "token_mock.pickle", cfg_path, r) for _ in range(4000)]
        freq = {h: picks.count(h) / len(picks) for h in hooks}
        wts = loaded["channels"]["token_mock.pickle"]["weights"]["cover_hooks"]
        exp = {h: wts[h] / sum(wts[x] for x in hooks) for h in hooks}
        ok("clipper weighted pick follows weights (±3%)", all(abs(freq[h] - exp[h]) < 0.03 for h in hooks))
        ok("clipper uses global block for unknown token",
           lcc.load_optimizer_weights("cover_hooks", "token_other.pickle", cfg_path) == loaded["global"]["weights"]["cover_hooks"])
        tpicks = {lcc.weighted_pick(titles, "titles", "token_mock.pickle", cfg_path, r) for _ in range(500)}
        ok("clipper title pick returns valid titles", tpicks <= set(titles))

        bad = Path(td) / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        ok("malformed config → uniform fallback", lcc.weighted_pick(hooks, "cover_hooks", None, bad) in hooks
           and lcc.load_optimizer_weights("cover_hooks", None, bad) == {})
        bad.write_text(json.dumps({"channels": {"t": {"weights": {"cover_hooks": {hooks[0]: "x", hooks[1]: -1}}}}}))
        ok("garbage weight values ignored", lcc.load_optimizer_weights("cover_hooks", "t", bad) == {})
        missing = Path(td) / "nope.json"
        ok("missing config → uniform fallback", lcc.weighted_pick(hooks, "cover_hooks", None, missing) in hooks)
        uni = [lcc.weighted_pick(hooks, "cover_hooks", None, missing, r) for _ in range(4000)]
        ok("uniform fallback ≈ 25% each", all(abs(uni.count(h) / 4000 - 1 / len(hooks)) < 0.03 for h in hooks))

    ok("parse_gemini_json handles ```json fences", parse_gemini_json('```json\n{"a": 1}\n```') == {"a": 1})
    ok("iso_duration_sec('PT1M5S') == 65", iso_duration_sec("PT1M5S") == 65)

    width = max(len(n) for n, _ in checks)
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name.ljust(width)}")
    failed = [n for n, p in checks if not p]
    print(f"\n🧪 mock test: {len(checks) - len(failed)}/{len(checks)} passed")
    print(f"   top hook weights: { {h: prev['weights']['cover_hooks'][h] for h in hooks} }")
    return 1 if failed else 0


# ─────────────────────────── CLI ───────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-learning YouTube Analytics feedback loop → optimizer_config.json",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--token", default=None, help=f"single token pickle (default: all existing of {DEFAULT_TOKENS})")
    p.add_argument("--days", type=int, default=28, help="analytics lookback window")
    p.add_argument("--dry-run", action="store_true", help="compute + print, do not write config/summary")
    p.add_argument("--mock-test", action="store_true", help="offline self-test (no network, no real config writes)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mock_test:
        return mock_test()

    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR / ".env")

    cfg = load_config(CONFIG_FILE)
    meta = load_local_metadata()
    tokens = [args.token] if args.token else DEFAULT_TOKENS
    updated = 0
    for token in tokens:
        res = run_token(token, args.days, cfg["channels"].get(token, {}), meta)
        if not res:
            continue
        cfg["channels"][token] = res
        updated += 1
        r = res["recommended"]
        log(f"🧠 {token}: clip_sec={r['clip_sec']} spike_mult={r['spike_mult']} "
            f"hooks={res['weights']['cover_hooks']}")
    if not updated:
        log("⚠️  no channel produced results — config unchanged")
        return 1
    cfg["version"] = 1
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.dry_run:
        cfg["global"] = build_global(cfg["channels"])
        print(render_summary(cfg))
        log("🧪 dry-run: nothing written")
    else:
        save_config(cfg, CONFIG_FILE, SUMMARY_FILE)
        log(f"💾 saved {CONFIG_FILE.name} + {SUMMARY_FILE.name} ({updated} channel(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
