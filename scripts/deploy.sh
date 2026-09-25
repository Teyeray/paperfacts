#!/usr/bin/env bash
# Linux workstation deployment for PaperFacts — the counterpart of the macOS-only
# scripts/dev_up.sh (see its header: that one clears a macOS-only .venv flag and starts an
# Apple-silicon mlx-vlm server, neither of which exists on this host).
#
# This machine already runs everything under `systemctl --user`:
#   paperfacts.service        127.0.0.1:8000   web UI + API + the single job worker
#   pf-vllm-paddle.service    127.0.0.1:8110   vLLM serving PaddleOCR-VL (the paddle lane)
#   paperfacts-tunnel.service                  cloudflared → https://paperfacts.yangruiming.org
# The venv is an editable install, so "deploying" is not a copy step: pull, sync only if the
# lock moved, run the tests, restart the unit, verify the running process. Nothing is installed
# anywhere else.
#
# Usage:
#   scripts/deploy.sh                    # deploy the working tree: preflight, tests, restart, verify
#   scripts/deploy.sh --check            # report only; exit 1 when the running service is stale
#   scripts/deploy.sh --pull             # git fetch + fast-forward the branch first (clean tree)
#   scripts/deploy.sh --all-units        # also restart the paddle lane and the tunnel
#   scripts/deploy.sh --force            # restart even with jobs queued/running
#   scripts/deploy.sh --skip-tests       # skip pytest (not recommended)
#   scripts/deploy.sh --public           # also verify the public HTTPS endpoint
#   scripts/deploy.sh --install-units    # (re)write the three unit files, then deploy
#   scripts/deploy.sh --rerun            # after deploying, re-run every document the new keys displaced
#   scripts/deploy.sh --rerun-only       # just the re-run, no deploy (skips tests and restarts)
#   scripts/deploy.sh --rerun-timeout=N  # seconds to wait for those jobs (default 1800, 0 = queue and return)
#
# PAPERFACTS_DEPLOY_URL overrides http://127.0.0.1:8000 (staging servers, tests).
#
# Exit codes: 0 deployed/verified · 1 stale (--check) or a failed step · 2 restart refused
#             (jobs queued or running) · 3 deployed but the paddle lane is down
set -euo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$ROOT"

USER_UNIT_DIR="$HOME/.config/systemd/user"
LOCAL_URL="${PAPERFACTS_DEPLOY_URL:-http://127.0.0.1:8000}"   # paperfacts.service's ExecStart port
PUBLIC_URL="https://paperfacts.yangruiming.org"
UA="Mozilla/5.0"                            # Cloudflare answers 403 to Python-urllib, even authenticated
STATE_FILE="$ROOT/data/deploy_state.json"   # data/ is gitignored; records what this tree deployed
HEALTH_TIMEOUT=90
RERUN_TIMEOUT=1800

MODE=deploy DO_PULL=0 ALL_UNITS=0 FORCE=0 SKIP_TESTS=0 PUBLIC=0 INSTALL_UNITS=0 RERUN=0

usage() { sed -n '/^# Usage:/,/^# Exit codes/p' "$0" | sed 's/^# \{0,1\}//'; }
info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
ok()    { printf '\033[1;32m ok \033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mwarn\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31mfail\033[0m %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
    case "$arg" in
        --check)         MODE=check ;;
        --pull)          DO_PULL=1 ;;
        --all-units)     ALL_UNITS=1 ;;
        --force)         FORCE=1 ;;
        --skip-tests)    SKIP_TESTS=1 ;;
        --public)        PUBLIC=1 ;;
        --install-units) INSTALL_UNITS=1 ;;
        --rerun)         RERUN=1 ;;
        --rerun-only)    MODE=rerun_only ;;
        --rerun-timeout=*) RERUN_TIMEOUT="${arg#*=}" ;;
        -h|--help)       usage; exit 0 ;;
        *)               usage >&2; die "unknown argument: $arg" ;;
    esac
done

# ---------------------------------------------------------------- preflight
[ "$(uname -s)" = "Linux" ] || die "this deploys the Linux workstation; on a Mac use scripts/dev_up.sh"
command -v uv >/dev/null || die "uv is not in PATH (expected ~/.local/bin/uv)"
command -v python3 >/dev/null || die "python3 is not in PATH"
systemctl --user show-environment >/dev/null 2>&1 \
    || die "systemd --user is not available for $USER (is this a login session with linger?)"
[ -f .env ] || die ".env is missing — copy .env.example and fill PAPERFACTS_LLM_API_KEY and PAPERFACTS_WEB_PASSWORD"
[ -f config.json ] || die "config.json is missing — the deployment cannot start without it"

unit_state() { systemctl --user show "$1" -p ActiveState --value 2>/dev/null || echo unknown; }
unit_started() {  # epoch of the running process, 0 when the unit is down
    # The human stamp systemd prints ("Mon 2026-09-21 09:57:53 HKT") is not something GNU date parses:
    # the bare zone abbreviation makes `date -d` fail, which silently turned this check into "always 0".
    # The monotonic property (microseconds since boot) is unambiguous, so convert it against /proc/uptime.
    local usec uptime_s
    usec="$(systemctl --user show "$1" -p ExecMainStartTimestampMonotonic --value 2>/dev/null || true)"
    case "$usec" in ''|0|*[!0-9]*) echo 0; return ;; esac
    uptime_s="$(cut -d. -f1 /proc/uptime)"
    echo $(( $(date +%s) - uptime_s + usec / 1000000 ))
}

env_value() {  # $1 = key: its value in .env, parsed the way the service parses it
    # The service reads .env with python-dotenv, so the same parser reads it here when the venv has it:
    # quotes are the value's delimiters, not part of it, and spaces or quotes inside a quoted value are kept.
    local py=python3
    if [ -x .venv/bin/python ] && .venv/bin/python -c 'import dotenv' 2>/dev/null; then py=.venv/bin/python; fi
    "$py" - "$1" <<'PY'
import sys

key = sys.argv[1]
try:
    from dotenv import dotenv_values

    value = dotenv_values(".env").get(key)
except ImportError:  # no venv yet: the common forms, the last assignment winning as in dotenv
    value = None
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, raw = line.partition("=")
        if not sep or name.strip() != key:
            continue
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            value = raw[1:-1]
            if raw[0] == '"':  # dotenv undoes backslash escapes inside double quotes
                value = value.replace('\\"', '"').replace("\\\\", "\\")
        else:
            value = raw.split(" #", 1)[0].rstrip()
sys.stdout.write(value or "")
PY
}

WEB_PASSWORD="$(env_value PAPERFACTS_WEB_PASSWORD)"
[ -n "$WEB_PASSWORD" ] || die "PAPERFACTS_WEB_PASSWORD is empty in .env"
WEB_USERNAME="$(python3 -c 'import json;print(json.load(open("config.json"))["web"]["username"])')"
# curl reads the credentials from a config on stdin (-K -), never from its argv: on a shared GPU host every
# user can read every process's command line through ps. printf is a shell builtin, so it has no argv either.
# Inside a quoted curl config value only backslash and double quote need escaping.
CURL_USER="$WEB_USERNAME:$WEB_PASSWORD"
CURL_USER="${CURL_USER//\\/\\\\}"
CURL_USER="${CURL_USER//\"/\\\"}"
curl_auth() {  # curl with the web credentials; takes curl's own arguments
    printf 'user = "%s"\n' "$CURL_USER" | curl -K - "$@"
}

# ---------------------------------------------------------------- re-run displaced documents
# keys.py names every stored extraction/comparison after a fingerprint of the package's own source, so a
# commit that touches the schema, the prompts or the comparison rules leaves each document's old artifacts
# unreachable: the library then reports `compared: false` and /report answers 404 "No comparison report yet".
# POST /api/documents/run-all queues exactly those documents (it skips the ones already compared) and returns
# at once; the single job worker then re-parses from the parse cache and re-compares.
rerun_pending_documents() {
    local tmp
    tmp="$(mktemp -d)"
    info "queueing every document the new keys displaced (POST /api/documents/run-all)"
    if ! curl_auth -sS --max-time 30 -A "$UA" -X POST "$LOCAL_URL/api/documents/run-all" \
        -o "$tmp/runall.json"; then
        rm -rf "$tmp"
        warn "run-all request failed — is paperfacts.service up?"
        return 1
    fi
    if ! python3 - "$tmp/runall.json" "$tmp/ids" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
submitted, skipped = body.get("submitted") or [], body.get("skipped") or []
with open(sys.argv[2], "w") as fh:
    for job in submitted:
        fh.write(job["job_id"] + "\n")
print(f"    queued {len(submitted)}, skipped {len(skipped)}")
for job in submitted:
    print(f"      + job {job['job_id']}  {job['document_id']}")
for item in skipped:
    print(f"      = {item.get('name')} — {item.get('reason')}")
PY
    then
        rm -rf "$tmp"
        warn "unexpected run-all response — see $LOCAL_URL/api/documents"
        return 1
    fi

    if [ ! -s "$tmp/ids" ]; then
        rm -rf "$tmp"
        ok "nothing to re-run: every document already has a comparison under these keys"
        return 0
    fi

    local total deadline done_count failed_count states
    total="$(wc -l <"$tmp/ids" | tr -d ' ')"
    deadline=$((SECONDS + RERUN_TIMEOUT))
    info "waiting for $total document(s), budget ${RERUN_TIMEOUT}s (the service finishes them with or without this script)"
    while [ "$SECONDS" -lt "$deadline" ]; do
        if curl_auth -sS --max-time 20 -A "$UA" "$LOCAL_URL/api/jobs" -o "$tmp/jobs.json" 2>/dev/null; then
            IFS='|' read -r done_count failed_count states < <(python3 - "$tmp/jobs.json" "$tmp/ids" <<'PY'
import json, sys
ids = [line.strip() for line in open(sys.argv[2]) if line.strip()]
try:
    jobs = {job["job_id"]: job for job in json.load(open(sys.argv[1]))}
except Exception:
    jobs = {}
done = failed = 0
parts = []
for job_id in ids:
    job = jobs.get(job_id) or {}
    status = job.get("status", "unknown")
    done += status == "done"
    failed += status == "failed"
    stage = next((s for s in job.get("stages", []) if s.get("status") == "running"), None)
    parts.append(f"{status}:{stage['name'] if stage else ''}")
print(f"{done}|{failed}|{' '.join(parts)}")
PY
)
            printf '\r     %s done, %s failed of %s  [%s]   ' \
                "${done_count:-0}" "${failed_count:-0}" "$total" "${states:0:90}"
            if [ "${done_count:-0}" -ge "$total" ]; then
                printf '\n'
                rm -rf "$tmp"
                ok "all $total document(s) re-ran under the new keys"
                return 0
            fi
            if [ "$((${done_count:-0} + ${failed_count:-0}))" -ge "$total" ]; then
                printf '\n'
                rm -rf "$tmp"
                warn "${failed_count:-0} job(s) failed — journalctl --user -u paperfacts.service -n 50"
                return 1
            fi
        fi
        sleep 10
    done
    printf '\n'
    rm -rf "$tmp"
    warn "budget of ${RERUN_TIMEOUT}s spent; the jobs keep running inside paperfacts.service — check $LOCAL_URL/api/jobs"
    return 0
}

if [ "$MODE" = rerun_only ]; then
    if rerun_pending_documents; then exit 0; else exit 1; fi
fi

# ---------------------------------------------------------------- optional pull
if [ "$DO_PULL" = 1 ]; then
    [ -z "$(git status --porcelain)" ] || die "--pull needs a clean tree (commit or stash first)"
    info "fetching origin"
    git fetch --quiet origin
    UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
    if [ -z "$UPSTREAM" ]; then
        warn "$(git rev-parse --abbrev-ref HEAD) has no upstream — skipping the fast-forward"
    else
        BEHIND="$(git rev-list --count "HEAD..$UPSTREAM")"
        if [ "$BEHIND" -gt 0 ]; then
            git merge --ff-only "$UPSTREAM"
            info "fast-forwarded ${BEHIND} commit(s) to $(git rev-parse --short HEAD)"
        else
            info "already level with $UPSTREAM"
        fi
    fi
fi

# ---------------------------------------------------------------- what is deployed
HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
HEAD_DESC="$(git log -1 --format='%h %s' 2>/dev/null || echo unknown)"
DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
LOCK_HASH="$(cat pyproject.toml uv.lock | sha256sum | cut -d' ' -f1)"

read_state() {  # $1 = key
    [ -f "$STATE_FILE" ] || { echo ""; return; }
    python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],"") or "")' \
        "$STATE_FILE" "$1" 2>/dev/null || echo ""
}
DEPLOYED_HEAD="$(read_state head)"
DEPLOYED_LOCK="$(read_state lock_hash)"

APP_STATE="$(unit_state paperfacts.service)"
APP_START="$(unit_started paperfacts.service)"
NEWEST_SRC="$(find src config.json -type f ! -path '*__pycache__*' -printf '%T@\n' 2>/dev/null \
    | sort -n | tail -1 | cut -d. -f1)"
NEWEST_SRC="${NEWEST_SRC:-0}"

NEED_RESTART=0 RESTART_WHY=""
if [ "$APP_STATE" != active ]; then
    NEED_RESTART=1 RESTART_WHY="paperfacts.service is $APP_STATE"
elif [ -z "$DEPLOYED_HEAD" ]; then
    NEED_RESTART=1 RESTART_WHY="no data/deploy_state.json yet, so what is running is unknown"
elif [ "$DEPLOYED_HEAD" != "$HEAD_SHA" ]; then
    NEED_RESTART=1 RESTART_WHY="running process predates $(git rev-parse --short "$DEPLOYED_HEAD") (HEAD is $(git rev-parse --short "$HEAD_SHA"))"
elif [ "$NEWEST_SRC" -gt "$APP_START" ]; then
    NEED_RESTART=1 RESTART_WHY="src/ or config.json is newer than the running process (uncommitted edit?)"
fi

NEED_SYNC=0
if [ ! -d .venv ]; then
    NEED_SYNC=1
elif [ -z "$DEPLOYED_LOCK" ] || [ "$DEPLOYED_LOCK" != "$LOCK_HASH" ]; then
    NEED_SYNC=1
fi

info "working tree"
printf '    branch      %s\n' "$BRANCH"
printf '    HEAD        %s (%s)\n' "$HEAD_DESC" "${HEAD_SHA:0:12}"
printf '    uncommitted %s file(s)\n' "$DIRTY"
printf '    deployed    %s\n' "${DEPLOYED_HEAD:0:12}${DEPLOYED_HEAD:+ (per data/deploy_state.json)}"
printf '    restart     %s\n' "$([ "$NEED_RESTART" = 1 ] && echo "needed — $RESTART_WHY" || echo "not needed")"
printf '    uv sync     %s\n' "$([ "$NEED_SYNC" = 1 ] && echo "needed (pyproject.toml/uv.lock moved or .venv missing)" || echo "not needed")"
if [ "$DIRTY" != 0 ]; then
    warn "the tree has uncommitted changes — the editable venv serves them as-is"
fi

if [ "$MODE" = check ]; then
    if [ "$NEED_RESTART" = 1 ]; then
        warn "stale: run scripts/deploy.sh to restart the service"
        exit 1
    fi
    ok "the running service is current"
    exit 0
fi

# ---------------------------------------------------------------- dependencies
if [ "$NEED_SYNC" = 1 ]; then
    info "uv sync --locked (the lock moved; on the CATL guest network export UV_HTTP_TIMEOUT=1800 first)"
    UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1800}" uv sync --locked
    ok "venv synced"
else
    ok "venv already matches pyproject.toml/uv.lock"
fi

# ---------------------------------------------------------------- tests
if [ "$SKIP_TESTS" = 1 ]; then
    warn "--skip-tests: deploying without running the suite"
else
    info "uv run --locked pytest -q"
    uv run --locked pytest -q 2>&1 | tail -n 3
    ok "tests green"
fi

# ---------------------------------------------------------------- in-flight work
# Jobs live only in the service's memory: a restart cancels everything queued and kills the
# running job (its stage artifacts stay on disk, so a re-queue resumes from the cache). Only a job
# queued or running is at stake. A document that is merely not compared -- never run, or failed for
# good -- loses nothing to a restart and must not block a deploy.
PENDING=""
JOBS_JSON="$(curl_auth -sS --max-time 10 -A "$UA" "$LOCAL_URL/api/jobs" 2>/dev/null || true)"
if [ -n "$JOBS_JSON" ]; then
    PENDING="$(printf '%s' "$JOBS_JSON" | python3 -c '
import json, sys
try:
    jobs = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for job in jobs if isinstance(jobs, list) else []:
    if job.get("status") in ("queued", "running"):
        stage = next((s["name"] for s in job.get("stages", []) if s.get("status") == "running"), "")
        line = "      - job %s doc %s %s %s" % (job.get("job_id"), job.get("document_id"), job["status"], stage)
        print(line.rstrip())
' 2>/dev/null || true)"
fi

if [ -n "$PENDING" ]; then
    warn "jobs queued or running — a restart drops them:"
    printf '%s\n' "$PENDING" >&2
    if [ "$FORCE" != 1 ]; then
        cat >&2 <<EOF
  Re-queue them after the restart with:
      curl -sS -u \$PASS -A "$UA" -X POST "$LOCAL_URL/api/documents/<document_id>/run"
  or re-run this script with --force to restart anyway.
EOF
        exit 2
    fi
    warn "--force: restarting anyway"
fi

# ---------------------------------------------------------------- units
if [ "$INSTALL_UNITS" = 1 ]; then
    info "installing the three unit files into $USER_UNIT_DIR"
    mkdir -p "$USER_UNIT_DIR"
    cat >"$USER_UNIT_DIR/paperfacts.service" <<EOF
[Unit]
Description=PaperFacts web UI (127.0.0.1:8000)
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$ROOT
ExecStart=$ROOT/.venv/bin/paperfacts serve
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
    cat >"$USER_UNIT_DIR/pf-vllm-paddle.service" <<EOF
[Unit]
Description=vLLM serving PaddleOCR-VL-1.6-0.9B for paperfacts paddle lane (127.0.0.1:8110)
After=network-online.target

[Service]
Type=simple
# flashinfer JIT needs nvcc, which this host does not have: without this the unit dies at startup.
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
ExecStart=$HOME/miniconda3/envs/vllm/bin/vllm serve $HOME/.paddlex/official_models/PaddleOCR-VL-1.6 --served-model-name PaddleOCR-VL-1.6-0.9B PaddleOCR-VL-1.6 --port 8110 --gpu-memory-utilization 0.30 --max-model-len 16384 --trust-remote-code
Restart=on-failure
RestartSec=15
TimeoutStartSec=900

[Install]
WantedBy=default.target
EOF
    cat >"$USER_UNIT_DIR/paperfacts-tunnel.service" <<EOF
[Unit]
Description=cloudflared tunnel for paperfacts.yangruiming.org
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/local/bin/cloudflared tunnel --config $HOME/.cloudflared/paperfacts.yml run paperfacts
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    loginctl enable-linger "$USER" 2>/dev/null || warn "could not enable linger (units stop at logout)"
    ok "unit files written and reloaded"
fi

for unit in paperfacts.service pf-vllm-paddle.service paperfacts-tunnel.service; do
    case "$unit" in
        paperfacts.service) ;;
        *) [ "$ALL_UNITS" = 1 ] || [ "$(unit_state "$unit")" != active ] || continue ;;
    esac
    if [ "$(unit_state "$unit")" = active ]; then
        info "systemctl --user restart $unit"
        systemctl --user restart "$unit"
    else
        info "systemctl --user start $unit"
        systemctl --user start "$unit"
    fi
done

# ---------------------------------------------------------------- verify
info "waiting for $LOCAL_URL/api/health (up to ${HEALTH_TIMEOUT}s)"
HEALTH=""
for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
    HEALTH="$(curl_auth -sS --max-time 5 -A "$UA" "$LOCAL_URL/api/health" 2>/dev/null || true)"
    case "$HEALTH" in *'"status"'*) break ;; esac
    if [ "$(unit_state paperfacts.service)" != active ]; then
        journalctl --user -u paperfacts.service -n 30 --no-pager >&2 || true
        die "paperfacts.service died during startup (journal above)"
    fi
    sleep 1
done
case "$HEALTH" in *'"status"'*) ok "health: $HEALTH" ;;
    *) journalctl --user -u paperfacts.service -n 30 --no-pager >&2 || true
       die "no healthy /api/health after ${HEALTH_TIMEOUT}s" ;;
esac

CODE_NOAUTH="$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' -A "$UA" "$LOCAL_URL/api/documents" || echo 000)"
[ "$CODE_NOAUTH" = 401 ] && ok "Basic gate still on (401 without credentials)" \
    || warn "expected 401 without credentials, got $CODE_NOAUTH"

CODE_UI="$(curl_auth -sS --max-time 10 -o /dev/null -w '%{http_code}' -A "$UA" "$LOCAL_URL/" || echo 000)"
[ "$CODE_UI" = 200 ] && ok "web UI 200" || warn "web UI returned $CODE_UI"

LANE_DOWN=0
PADDLE_BACKEND="$(sed -n 's/^PAPERFACTS_PADDLE_VL_BACKEND=//p' .env | head -n1)"
PADDLE_URL="$(sed -n 's/^PAPERFACTS_PADDLE_VL_SERVER_URL=//p' .env | head -n1)"
PADDLE_MODEL="$(sed -n 's/^PAPERFACTS_PADDLE_VL_MODEL_NAME=//p' .env | head -n1)"
if [ "$PADDLE_BACKEND" = "vllm-server" ]; then
    MODELS="$(curl -sS --max-time 10 "${PADDLE_URL%/}/models" 2>/dev/null || true)"
    case "$MODELS" in
        *"$PADDLE_MODEL"*) ok "paddle lane: $PADDLE_MODEL at $PADDLE_URL" ;;
        *) LANE_DOWN=1
           warn "paddle lane not serving $PADDLE_MODEL at $PADDLE_URL — parses would run an in-process VLM, which hangs on this host"
           warn "fix: systemctl --user restart pf-vllm-paddle.service   (or re-run with --all-units; first start loads weights, ~1 min)" ;;
    esac
fi

if [ "$PUBLIC" = 1 ]; then
    PUB="$(curl_auth -sS --max-time 20 -A "$UA" "$PUBLIC_URL/api/health" 2>/dev/null || true)"
    case "$PUB" in *'"status"'*) ok "public: $PUBLIC_URL/api/health" ;; *) warn "public endpoint did not answer: $PUB" ;; esac
fi
ok "service started at $(systemctl --user show paperfacts.service -p ExecMainStartTimestamp --value)"

# ---------------------------------------------------------------- record
python3 - "$STATE_FILE" "$HEAD_SHA" "$LOCK_HASH" <<'PY'
import datetime, json, os, sys
path, head, lock = sys.argv[1:4]
state = {
    "head": head,
    "lock_hash": lock,
    "deployed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "branch": os.popen("git rev-parse --abbrev-ref HEAD").read().strip(),
}
with open(path, "w") as fh:
    json.dump(state, fh, indent=2)
PY
ok "recorded in data/deploy_state.json: ${HEAD_SHA:0:12}"

if [ "$RERUN" = 1 ]; then
    if [ "$LANE_DOWN" = 1 ]; then
        warn "skipping --rerun: the paddle lane is down and an in-process VLM hangs on this host"
    else
        rerun_pending_documents || exit 1
    fi
fi

if [ "$LANE_DOWN" = 1 ]; then
    warn "deployed, but the paddle lane is down (see above)"
    exit 3
fi
info "deployed $(git rev-parse --short HEAD) — UI at $LOCAL_URL, public at $PUBLIC_URL"