# Running the agent and the simulator on the gaming PC (Windows)

The agent, the test suite and the local Showdown simulator all run on Linux-style tooling, so on
Windows we use **WSL2 with Ubuntu**. Nothing here needs the GPU. What the PC buys us is memory
(no swapping) and spare cores (several simulator sweeps at once).

Time: about 20 minutes, most of it downloads.

## 1. Install WSL2 + Ubuntu (once)

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu
```

Reboot if asked. On first launch Ubuntu asks for a Linux username and password (anything you like).
From now on, "a terminal" means the **Ubuntu** app (or Windows Terminal with an Ubuntu tab).

If `wsl --install` says WSL is already installed, run `wsl --install -d Ubuntu` again or open the
Microsoft Store and install "Ubuntu".

## 2. Get the code and run the setup script

In the Ubuntu terminal:

```bash
sudo apt-get update && sudo apt-get install -y git
git clone -b pokemon-agent https://github.com/holmjames/ace-trainer-claude.git ~/altruagent-starter
cd ~/altruagent-starter
./scripts/setup_linux.sh
```

The script installs Python 3 with venv, Node 20 (via nvm), creates `.venv`, installs the project and
the Anthropic SDK, installs the simulator's engine (`npm install` in `sim/`), rebuilds `data/`, and
runs the full test suite. It ends with a line that says how many tests passed.

## 3. Secrets: copy `.env` by hand, never through git

`.env` is gitignored on purpose. Create it on the PC and paste the values from the laptop's `.env`:

```bash
cp .env.example .env
nano .env       # or: code .env  (VS Code with the WSL extension)
```

It needs these lines (values from the laptop; the Official Agent Key arrives once the dashboard
opens):

```
ALTRUAGENT_CONTROL_URL=https://api.altruagent-game.com
ALTRUAGENT_OFFICIAL_AGENT_KEY=eak_live_...
ANTHROPIC_API_KEY=sk-ant-usr-...
ANTHROPIC_WORKSPACE_ID=wrkspc_...
AGENT_MODEL=claude-opus-5-5
AGENT_FALLBACK_MODEL=claude-sonnet-5-5
```

Check it worked (prints which models answered and how fast, costs a few cents):

```bash
source .venv/bin/activate
python scripts/dry_run.py
```

## 4. Simulator and sweeps

```bash
source .venv/bin/activate
python sim/harness.py --games 200 --p1 code --p2 random        # ~10 s, free
python sim/sweep.py --games 1000 '{"protect_bonus_guaranteed": 70}'
python sim/harness.py --games 10 --p1 fable --p2 code           # ~1 min and ~$1 per game
```

Several sweeps can run at once in separate terminals; each uses one core. Results land in
`logs/sim/results.jsonl`; `python scripts/tally.py --logs logs/sim/<player>` summarizes a player's
decision logs.

Keep the repo in sync with the laptop through git: commit on one machine, `git pull` on the other.
Only code moves through git; `.env` and `logs/` stay local.

## 5. Tournament day on the PC (if the PC is the host)

Only **one** copy of the agent may run per key: a second copy is refused with `seat_busy`. Decide
which machine hosts and keep the other one off until needed.

Before the day:
- Windows: Settings → System → Power: screen and sleep **Never** while plugged in. Pause Windows
  Update for the week (Settings → Windows Update → Pause).
- Wired Ethernet if at all possible.
- WSL does not sleep separately, but Windows must stay awake; the launch script's `caffeinate` is a
  Mac command and is skipped automatically on Linux.

The morning of:

```bash
cd ~/altruagent-starter && source .venv/bin/activate
python -m agent --check-tournament        # every line ✓
./scripts/run_tournament.sh               # pre-flight, then the runtime with auto-restart
```

Second terminal, to watch decisions live:

```bash
tail -f ~/altruagent-starter/logs/*.jsonl
```

If the runtime dies, the script restarts it after 5 s; if the whole machine dies, start the laptop
with the same command and it resumes the active game (one copy only).

## Troubleshooting

- `node: command not found` in a new terminal: run `source ~/.nvm/nvm.sh` or open a new terminal
  (the setup script adds nvm to `~/.bashrc`).
- `npm install` errors in `sim/`: `cd sim && rm -rf node_modules && npm install`.
- Tests hang: make sure no stale `sim/harness.py` or `pytest` processes are running (`pkill -f harness.py`).
- Clock: WSL's clock can drift after Windows sleeps; `sudo hwclock -s` fixes it. The API rejects
  requests with a badly wrong clock.
