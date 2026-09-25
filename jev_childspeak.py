#!/usr/bin/env python3
"""
jev_childspeak — autoregressive mouth on top of TypeSafe Jev (OpenRouter).

Each step: Jev picks ONE token from a closed vocabulary (~255),
given the user message, conversation history, and words so far.
We append and ask again until END, a long repeat, or the ceiling.

  # PowerShell:
  $env:OPENROUTER_API_KEY = "sk-or-v1-..."
  python jev_childspeak.py
  python jev_childspeak.py --load jev_chat.json
  python jev_childspeak.py --name Liora
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# ── vocabulary ──────────────────────────────────────────────────────────────

LETTERS: list[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

NUMBERS: list[str] = [str(d) for d in range(10)]

PUNCT: list[str] = [
    ".", ",", "?", "!", ":", ";", "-", '"', "'", "(", ")", "/",
]

LINEBREAK = "↵"
END = "END"

INSTRUCTIONS = (
    "Please pick the next token that continues your reply. "
    "You can also use single letters A-B-C-D...-Z to create words that are not on your list. "
    "Do it one letter at a time (e.g. C-A-T makes cat🐈). "
    "Use punctuation tokens when necessary. "
    "Choose END when your reply is complete and please avoid picking the same token multiple times in a row unless it's intentional. "
    "Ideally, try to build sentences using a structure like: subject + verb + object. "
)

SEMANTIC: list[str] = [
    # identity & reference
    "I", "me", "you", "we", "they", "this", "that", "here", "there",
    "person", "thing", "it", "she", "he",
    "Jev", "is", "are", "am", "Liora",
    # logical & grammatical
    "every", "some", "any", "no", "yes", "not", "and", "or", "but",
    "if", "then", "because", "so", "only", "more", "less", "too",
    "enough", "same", "other", "all", "be", "have", "do", "can", "with",
    # questions
    "who", "what", "where", "when", "why", "how", "which",
    # epistemic / truth
    "true", "false", "real", "maybe", "possible",
    # modality & intention
    "will", "would", "should", "must", "may", "might",
    "want", "need", "hope",
    # cognition & communication
    "fear", "know", "think", "believe", "guess", "understand",
    "remember", "forget", "learn", "teach", "explain",
    "ask", "say", "tell", "mean", "show", "hear", "lie", "see", "look",
    "don't-know",
    # action & agency
    "feel", "make", "use", "give", "take", "get", "put", "keep",
    "find", "lose", "bring", "send", "receive", "share", "choose",
    "try", "help", "allow", "refuse", "agree", "disagree",
    # change, movement & continuation
    "change", "start", "stop", "continue", "wait", "come", "go",
    "stay", "return", "move", "work", "live",
    # social / interpersonal
    "love", "hate", "trust", "sorry", "thanks", "welcome",
    "hello", "goodbye", "okay", "friend", "family",
    # time & existence
    "time", "now", "yet", "today", "tomorrow", "yesterday",
    "day", "night", "world", "place", "already", "still",
    # conceptual / reasoning
    "idea", "word", "message", "question", "answer", "reason",
    "goal", "plan", "problem", "choice", "rule", "system",
    "part", "way", "kind", "in", "out", "away",
    # knowledge & information
    "meaning", "language", "information", "error",
    # value / comparison / description
    "good", "bad", "better", "easy", "hard", "big", "small",
    "new", "old", "important", "useful", "right", "wrong",
    "safe", "danger", "different",
    # emotional / expressive
    "happy", "sad", "angry", "afraid", "calm", "excited", "awe",
    "tired", "grief", "disappointed", "wow", "haha", "like",
    "damn", "fuck",
    # concrete / world
    "body", "mind", "heart", "home",
    # other
    "context",
    # productive morphology
    "-ly", "-ed", "-ing", "un-",
]

# Semantics first so whole-word options appear before letters/digits/punct.
# (Choice menus preserve this order; may bias toward real words over spelling.)
_seen: set[str] = set()
VOCAB: list[str] = []
for w in SEMANTIC + LETTERS + NUMBERS + PUNCT + [LINEBREAK] + [END]:
    if w not in _seen:
        _seen.add(w)
        VOCAB.append(w)

assert END in VOCAB
assert len(VOCAB) == len(set(VOCAB))
assert len(VOCAB) <= 255, f"vocab has {len(VOCAB)} tokens; choice max is 255"

API_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
MAX_WORDS = int(os.environ.get("JEV_MAX_WORDS", "120"))
REPEAT_LIMIT = int(os.environ.get("JEV_REPEAT_LIMIT", "10"))
_ht = os.environ.get("JEV_HISTORY_TURNS", "0").strip()
HISTORY_TURNS = int(_ht) if _ht else 0
TIMEOUT = 45
DEBUG = os.environ.get("JEV_DEBUG", "").strip() in ("1", "true", "yes")
DEFAULT_CHAT_PATH = os.environ.get("JEV_CHAT_FILE", "jev_chat.json")

# Set at startup (prompt, --name, env, or from a loaded chat file)
HUMAN_NAME = "Human"


def menu_for_step() -> list[str]:
    return [w for w in VOCAB if w != END] + [END]


def format_reply(tokens: list[str]) -> str:
    if not tokens:
        return ""

    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t == END:
            break
        if t == LINEBREAK:
            out.append("\n")
            i += 1
            continue

        if t == "un-" and i + 1 < n and tokens[i + 1] not in PUNCT + [END, LINEBREAK]:
            out.append("un" + tokens[i + 1])
            i += 2
            continue

        if t in LETTERS or t in NUMBERS:
            chunk = t
            i += 1
            while i < n and tokens[i] in LETTERS + NUMBERS:
                chunk += tokens[i]
                i += 1
            while i < n and tokens[i] in ("-ing", "-ed", "-ly"):
                chunk += tokens[i][1:]
                i += 1
            out.append(chunk)
            continue

        word = t
        i += 1
        while i < n and tokens[i] in ("-ing", "-ed", "-ly"):
            word += tokens[i][1:]
            i += 1
        out.append(word)

    s = " ".join(out)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+/\s+", "/", s)
    s = re.sub(r"\s+\n\s+", "\n", s)
    return s.strip()


def history_slice(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if HISTORY_TURNS and HISTORY_TURNS > 0:
        return history[-HISTORY_TURNS:]
    return history


def build_state(user_text: str, so_far: list[str], history: list[tuple[str, str]]) -> str:
    """What Jev reads — uses your chosen name as the human label."""
    name = HUMAN_NAME or "Human"
    parts: list[str] = []
    past = history_slice(history)
    if past:
        parts.append("Earlier:")
        for u, j in past:
            parts.append(f"{name}: {u}")
            parts.append(f"Jev: {j}")
        parts.append("")
    parts.append(f"{name}: {user_text}")
    if so_far:
        parts.append(f"Jev: {' '.join(so_far)}")
    else:
        parts.append("Jev:")
    parts.append(INSTRUCTIONS)
    return "\n".join(parts)


def save_chat(path: str, history: list[tuple[str, str]]) -> None:
    data = {
        "model": MODEL,
        "human_name": HUMAN_NAME,
        "turns": [
            {HUMAN_NAME: u, "Jev": j} for u, j in history
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_chat(path: str) -> tuple[list[tuple[str, str]], str | None]:
    """
    Returns (history, human_name_or_None).
    Accepts new schema ({"Liora": "...", "Jev": "..."}) and old
    ({"user": "...", "jev": "..."} / {"human": "..."}).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    saved_name = data.get("human_name") or data.get("name")
    turns = data.get("turns") or data.get("history") or []
    out: list[tuple[str, str]] = []

    for t in turns:
        if isinstance(t, dict):
            human_text = None
            jev_text = None

            # New schema: {"Liora": "...", "Jev": "..."}
            if saved_name and saved_name in t:
                human_text = t.get(saved_name)
            for k, v in t.items():
                kl = str(k).lower()
                if kl in ("jev", "reply", "assistant"):
                    jev_text = v
                elif human_text is None and kl in ("user", "human", "you"):
                    human_text = v

            # Fallback: first non-Jev key is the human message
            if human_text is None:
                for k, v in t.items():
                    if str(k).lower() not in ("jev", "reply", "assistant", "model"):
                        human_text = v
                        if not saved_name:
                            saved_name = str(k)
                        break

            if jev_text is None:
                jev_text = t.get("Jev", t.get("jev", t.get("reply", "")))
            out.append((str(human_text or ""), str(jev_text or "")))
        elif isinstance(t, (list, tuple)) and len(t) >= 2:
            out.append((str(t[0]), str(t[1])))

    return out, (str(saved_name) if saved_name else None)


def print_history(history: list[tuple[str, str]], header: str = "") -> None:
    name = HUMAN_NAME or "you"
    if header:
        print(header)
    if not history:
        print("(empty)")
        return
    for u, j in history:
        print(f"{name}: {u}")
        print(f"Jev: {j}")


def repeated_tail(so_far: list[str], n: int = REPEAT_LIMIT) -> bool:
    if n <= 0 or len(so_far) < n:
        return False
    return len(set(so_far[-n:])) == 1


def jev_next_word(
    user_text: str,
    so_far: list[str],
    options: list[str],
    history: list[tuple[str, str]],
) -> tuple[str, dict]:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "Set OPENROUTER_API_KEY first.\n"
            "  PowerShell:  $env:OPENROUTER_API_KEY = \"sk-or-v1-...\""
        )

    criteria = {w: w for w in options}
    payload = {
        "model": MODEL,
        "state": build_state(user_text, so_far, history),
        "questions": {
            "next": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": criteria,
            }
        },
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/jev-childspeak",
            "X-Title": "jev-childspeak",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            data = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:800]}") from e

    ans = data.get("answers", {}).get("next", {})
    choice = ans.get("choice")
    if not choice or choice not in options:
        choice = next((w for w in options if w != END), END)
    return choice, data


def speak(user_text: str, history: list[tuple[str, str]] | None = None) -> str:
    history = history or []
    so_far: list[str] = []
    options = menu_for_step()
    for _ in range(MAX_WORDS):
        t0 = time.time()
        word, _raw = jev_next_word(user_text, so_far, options, history)
        if DEBUG:
            print(f"  · {word}  ({time.time() - t0:.2f}s)", flush=True)
        if word == END:
            break
        so_far.append(word)
        if repeated_tail(so_far):
            if DEBUG:
                print(f"  · (stop: same token ×{REPEAT_LIMIT})", flush=True)
            break
    return format_reply(so_far)


def ask_human_name(default: str = "Human") -> str:
    """Prompt once for the name that appears in the transcript."""
    try:
        raw = input(f"Your name for this chat [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return raw if raw else default


def main() -> None:
    global HUMAN_NAME

    hist_label = "all" if not HISTORY_TURNS else str(HISTORY_TURNS)
    print(f"jev-childspeak · model={MODEL} · vocab={len(VOCAB)} · max={MAX_WORDS}")
    print(f"history={hist_label} · repeat_stop={REPEAT_LIMIT}")
    print("commands: /save [file]  /load [file]  /history  /clear  /quit")
    print(f"default chat file: {DEFAULT_CHAT_PATH}\n")

    history: list[tuple[str, str]] = []
    args = list(sys.argv[1:])
    name_from_cli: str | None = None
    load_path: str | None = None

    # Parse simple flags: --name X  --load path
    i = 0
    while i < len(args):
        if args[i] in ("--name", "-n") and i + 1 < len(args):
            name_from_cli = args[i + 1].strip()
            i += 2
            continue
        if args[i] in ("--load", "-l") and i + 1 < len(args):
            load_path = args[i + 1]
            i += 2
            continue
        break

    leftover = args[i:]

    # One-shot message mode: python jev_childspeak.py "hello"
    if leftover and not leftover[0].startswith("-") and load_path is None:
        if name_from_cli:
            HUMAN_NAME = name_from_cli
        elif os.environ.get("JEV_HUMAN_NAME", "").strip():
            HUMAN_NAME = os.environ["JEV_HUMAN_NAME"].strip()
        else:
            HUMAN_NAME = "Human"
        user = " ".join(leftover)
        print(f"{HUMAN_NAME}: {user}")
        print(f"Jev: {speak(user)}")
        return

    # Load chat first so we can adopt its saved name
    if load_path:
        try:
            history, saved_name = load_chat(load_path)
            if saved_name and not name_from_cli:
                HUMAN_NAME = saved_name
            print_history(
                history,
                header=f"(loaded {len(history)} turns from {load_path})",
            )
            print()
        except OSError as e:
            print(f"(load failed: {e})")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"(bad chat file: {e})")

    # Resolve human name: CLI > env > saved file > prompt
    if name_from_cli:
        HUMAN_NAME = name_from_cli
    elif os.environ.get("JEV_HUMAN_NAME", "").strip():
        HUMAN_NAME = os.environ["JEV_HUMAN_NAME"].strip()
    elif load_path and HUMAN_NAME and HUMAN_NAME != "Human":
        pass  # already set from file
    else:
        default = HUMAN_NAME if HUMAN_NAME != "Human" else (
            os.environ.get("JEV_HUMAN_NAME", "").strip() or "Human"
        )
        HUMAN_NAME = ask_human_name(default)

    print(f"(talking as {HUMAN_NAME})\n")

    while True:
        try:
            user = input(f"{HUMAN_NAME}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            if history:
                try:
                    save_chat(DEFAULT_CHAT_PATH, history)
                    print(f"(auto-saved {len(history)} turns → {DEFAULT_CHAT_PATH})")
                except OSError as e:
                    print(f"(auto-save failed: {e})")
            break
        if not user:
            break

        if user.startswith("/"):
            parts = user.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/quit", "/exit", "/q"):
                if history:
                    save_chat(DEFAULT_CHAT_PATH, history)
                    print(f"(saved {len(history)} turns → {DEFAULT_CHAT_PATH})")
                break
            if cmd == "/save":
                path = arg or DEFAULT_CHAT_PATH
                save_chat(path, history)
                print(f"(saved {len(history)} turns → {path})")
                continue
            if cmd == "/load":
                path = arg or DEFAULT_CHAT_PATH
                try:
                    history, saved_name = load_chat(path)
                    if saved_name:
                        HUMAN_NAME = saved_name
                        print(f"(name set to {HUMAN_NAME} from file)")
                    print_history(
                        history,
                        header=f"(loaded {len(history)} turns from {path})",
                    )
                except OSError as e:
                    print(f"(load failed: {e})")
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"(bad chat file: {e})")
                continue
            if cmd == "/history":
                print_history(history)
                continue
            if cmd == "/clear":
                history = []
                print("(history cleared)")
                continue
            if cmd == "/name":
                if arg:
                    HUMAN_NAME = arg
                    print(f"(name set to {HUMAN_NAME})")
                else:
                    HUMAN_NAME = ask_human_name(HUMAN_NAME)
                    print(f"(name set to {HUMAN_NAME})")
                continue
            print("(unknown command — try /save /load /history /clear /name /quit)")
            continue

        reply = speak(user, history)
        print(f"Jev> {reply}")
        history.append((user, reply))


if __name__ == "__main__":
    main()
