# Multi-Agent Student Analyzer — Docker Setup

## Files
- `Dockerfile` — builds the Streamlit app image
- `requirements.txt` — pinned Python dependencies (verified to install cleanly together)
- `docker-compose.yml` — one-command run with healthcheck
- `.dockerignore` — keeps build context small

## Before you build
Save your app code as **`app.py`** in this same folder (the Dockerfile expects that exact filename). If your file has a different name, either rename it or edit the `COPY app.py .` line in the `Dockerfile`.

## Build & run

### Option A — Docker Compose (recommended)
```bash
docker compose up --build
```
Visit http://localhost:8501

Stop with `docker compose down`.

### Option B — Plain Docker
```bash
docker build -t multi-agent-student-analyzer .
docker run -p 8501:8501 multi-agent-student-analyzer
```

## Publishing to Docker Hub

### 1. Create a Docker Hub account/repo (if you haven't)
Sign up at https://hub.docker.com and note your username. You don't need to pre-create the repo — pushing creates it automatically (as a public repo by default).

### 2. Build, tag, and push
The included `push.sh` script does this for you, including a multi-arch build (amd64 + arm64) so it works on both Intel/AMD and Apple Silicon/ARM machines:

```bash
chmod +x push.sh
./push.sh <your-dockerhub-username> latest
```

It will prompt you to `docker login` if you aren't already, then build and push `your-dockerhub-username/multi-agent-student-analyzer:latest`.

If you'd rather do it manually:
```bash
docker login
docker build -t <your-dockerhub-username>/multi-agent-student-analyzer:latest .
docker push <your-dockerhub-username>/multi-agent-student-analyzer:latest
```

### 3. Give the other user access
- **Public repo** (default): they just need the image name — no login required on their end.
- **Private repo**: they'll need to run `docker login` with credentials you grant (either their own Docker Hub account added as a collaborator, or shared credentials), before they can `docker pull`.

## Running it as another user (on a different machine/account)

They don't need your source code or Dockerfile — just the published image.

**Simplest — plain Docker:**
```bash
docker pull <your-dockerhub-username>/multi-agent-student-analyzer:latest
docker run -d -p 8501:8501 --name student-analyzer <your-dockerhub-username>/multi-agent-student-analyzer:latest
```
Then visit `http://localhost:8501` (or `http://<server-ip>:8501` if running on a remote machine).

**With Compose:** share `docker-compose.pull.yml` with them, have them replace `YOUR_DOCKERHUB_USERNAME` with your actual username, then:
```bash
docker compose -f docker-compose.pull.yml up -d
```

**Updating to a newer version later:**
```bash
docker compose -f docker-compose.pull.yml pull
docker compose -f docker-compose.pull.yml up -d
```

## Notes
- **HF token / Gmail credentials** are entered directly in the app's sidebar at runtime — nothing is baked into the image. If you'd rather inject them via environment variables, you'd need to adapt `app.py` to read `os.environ.get("HF_TOKEN")` etc. and default the sidebar fields to those values.
- The image runs as a **non-root user** (`appuser`) for basic container hardening.
- A **healthcheck** hits Streamlit's `/_stcore/health` endpoint every 30s.
- `matplotlib`, `pandas`, and `numpy` occasionally need a few system shared libraries on slim Debian images (`libfreetype6`, `libpng16-16`, `libgomp1`) — these are already installed in the Dockerfile.
- Uploaded Excel files are handled entirely in-browser via Streamlit's file uploader; no persistent volume is required unless you want one (the compose file includes an optional `./data` mount you can remove if unused).
- If you deploy this behind a reverse proxy (nginx, Traefik, etc.), make sure WebSocket upgrade headers are forwarded — Streamlit relies on WebSockets for live updates.
