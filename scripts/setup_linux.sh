#!/usr/bin/env bash
# One-shot setup for Ubuntu / WSL2 (see docs/GAMING-PC-SETUP.md). Safe to rerun.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git curl build-essential >/dev/null

echo "== Node 20 via nvm"
if ! command -v node >/dev/null 2>&1 || [ "$(node -v | cut -c2-3)" -lt 20 ]; then
  if [ ! -d "$HOME/.nvm" ]; then
    curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
  fi
  # shellcheck disable=SC1091
  source "$HOME/.nvm/nvm.sh"
  nvm install 20 >/dev/null
  nvm alias default 20 >/dev/null
fi
# shellcheck disable=SC1091
[ -s "$HOME/.nvm/nvm.sh" ] && source "$HOME/.nvm/nvm.sh"
echo "node $(node -v), npm $(npm -v)"

echo "== Python venv + project"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e ".[dev]" anthropic
echo "python $(.venv/bin/python --version)"

echo "== simulator engine"
(cd sim && npm install --silent)

echo "== data tables"
.venv/bin/python scripts/build_dex.py

echo "== .env"
if [ ! -f .env ]; then
  cp .env.example .env
  printf '\n# Added for our agent (see PLAN.md and docs/GAMING-PC-SETUP.md). Fill in, never commit.\nANTHROPIC_API_KEY=\nANTHROPIC_WORKSPACE_ID=\nAGENT_MODEL=claude-opus-5-5\nAGENT_FALLBACK_MODEL=claude-sonnet-5-5\n' >> .env
  echo "created .env from the template: paste the keys from the laptop into it (nano .env)"
else
  echo ".env already present (left untouched)"
fi

echo "== tests"
.venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -1

echo
echo "Done. Next: fill in .env, then:  source .venv/bin/activate && python scripts/dry_run.py"
