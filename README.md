# jev-childspeak

A tiny terminal chat that gives [TypeSafe Jev](https://openrouter.ai/typesafe/jev-1.13) a closed vocabulary and asks him, one token at a time, what to say next.

Jev is a **decision model** (typed choices), not a chat LLM. This script turns him into a constrained mouth: pick a word → append → ask again until `END`.

Built by **Grok** and **Liora**.

## Requirements

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/keys) API key
- **TypeSafe** allowed under [OpenRouter → Settings → Privacy](https://openrouter.ai/settings/privacy) (Jev only runs on that provider)

## Quick start

**Windows (PowerShell)**

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
cd path\to\folder
python jev_childspeak.py
```

**macOS / Linux**

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
cd path/to/folder
python3 jev_childspeak.py
```

When prompted, enter the name you want in the transcript (e.g. `Liora`).  
Jev will see lines like `Liora: …` / `Jev: …`.

One-shot (no interactive loop):

```powershell
python jev_childspeak.py --name Liora "hello"
```

## Chat commands

| Command | What it does |
|--------|----------------|
| `/save` | Save to `jev_chat.json` |
| `/save mychat.json` | Save to a path you choose |
| `/load` | Load `jev_chat.json` |
| `/load mychat.json` | Load that file (restores name + full history) |
| `/history` | Print the conversation |
| `/clear` | Wipe memory for this session |
| `/name Liora` | Change your display name |
| `/quit` | Save and exit |

Also:

```powershell
python jev_childspeak.py --load jev_chat.json --name Liora
```

Chats auto-save on quit / Ctrl+C.

## Optional environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | — | **Required** |
| `JEV_MODEL` | `typesafe/jev-1.13` | Model id |
| `JEV_MAX_WORDS` | `120` | Max tokens per reply |
| `JEV_REPEAT_LIMIT` | `10` | Stop if the same token repeats this many times |
| `JEV_HISTORY_TURNS` | `0` (all) | Cap how many past turns go into state |
| `JEV_HUMAN_NAME` | prompt | Skip the name prompt |
| `JEV_CHAT_FILE` | `jev_chat.json` | Default save path |
| `JEV_DEBUG` | off | Print each token as it is chosen |

## Notes

- Vocabulary is fixed (~255 tokens: words, `A`–`Z`, digits, punctuation, `END`). Jev cannot invent tokens outside that list; he can spell with letters.
- Replies are sequential API calls (one decision per token). Longer answers cost more and take longer.

## License

None.
