# Production Deployment Guide

This guide deploys:

- Frontend: Vercel at `https://interview-ai-agent.vercel.app`
- Backend API: Render at `https://interview-ai-agent-kpc8.onrender.com`
- Database: Supabase Postgres
- LiveKit server: Docker-based self-hosting, with an important production warning below
- AI interview agent: Fly.io, with the event-driven direction you want

Do not commit real secrets into GitHub. Keep `.env` local only. In production dashboards, paste secrets into Vercel, Render, Supabase, and Fly.io settings.

## Very Important Architecture Note

Render is good for the FastAPI backend.

Render is not a good public host for self-hosted LiveKit media. Render web services expose one public HTTP port. LiveKit needs public WebSocket plus WebRTC media ports, commonly UDP port ranges. Render docs say public traffic is forwarded to one HTTP port, while additional ports are only available over private networking. LiveKit self-hosting docs recommend production networking with SSL/TURN/media port configuration and say Docker host networking is best for performance.

Because of that, my production recommendation is:

1. Best: use LiveKit Cloud.
2. Good self-hosted option: run LiveKit on a VPS/Fly.io/VM where you can expose TCP and UDP ports.
3. Risky: LiveKit Docker on Render. It may connect by WebSocket but fail audio/video media in real browsers.

Official docs:

- Vercel env vars: https://vercel.com/docs/projects/environment-variables
- Render web services and port binding: https://render.com/docs/web-services
- Render env vars: https://render.com/docs/configure-environment-variables/
- Supabase Postgres connection strings/poolers: https://supabase.com/docs/guides/database/connecting-to-postgres/serverless-drivers
- LiveKit self-hosting: https://docs.livekit.io/transport/self-hosting/
- LiveKit deployment config: https://docs.livekit.io/home/self-hosting/deployment
- Fly Machines API: https://fly.io/docs/machines/api/machines-resource/

## Production URLs

Use these exact public URLs:

```text
FRONTEND_URL=https://interview-ai-agent.vercel.app
BACKEND_URL=https://interview-ai-agent-kpc8.onrender.com
```

For LiveKit, choose one:

```text
# Recommended
LIVEKIT_URL=wss://YOUR-LIVEKIT-CLOUD-OR-VPS-DOMAIN

# Risky if you still try Render LiveKit Docker
LIVEKIT_URL=wss://YOUR-LIVEKIT-RENDER-SERVICE.onrender.com
```

## What Changed In The Codebase For Production

I made these production-safety changes:

- `api_server.py` now respects Render's `PORT` env var.
- `dashboard/next.config.ts` no longer injects LiveKit API key/secret into the frontend build.
- `dashboard/app/interviews/[id]/page.tsx` now fetches interview data and LiveKit tokens from the Render backend.
- `dashboard/app/interview/[roomId]/InterviewRoom.tsx` now requests tokens from the backend.
- `dashboard/app/api/livekit/token/route.ts` now proxies to the backend instead of generating tokens with Vercel-held LiveKit secrets.

This means Vercel does not need `LIVEKIT_API_SECRET`.

## Deployment Order

Do this in this order:

1. Create Supabase database.
2. Deploy Render backend API.
3. Deploy LiveKit server.
4. Deploy Fly.io agent.
5. Deploy Vercel frontend.
6. Run one full interview test.

Do not deploy the frontend first. It needs the backend URL.

## Step 1: Supabase Postgres

1. Open Supabase.
2. Create a new project.
3. Go to Project Settings.
4. Go to Database.
5. Click Connect.
6. Use the Session Pooler connection string for Render.

Use Session Pooler because Render is a persistent server and often needs IPv4-compatible pooled access.

The string usually looks like:

```text
postgresql://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@aws-0-YOUR_REGION.pooler.supabase.com:5432/postgres
```

If Supabase gives `postgres://`, that is okay. The backend already converts it to `postgresql://`.

Set this in Render:

```text
DATABASE_URL=postgresql://postgres.YOUR_PROJECT_REF:YOUR_PASSWORD@aws-0-YOUR_REGION.pooler.supabase.com:5432/postgres
```

Do not use your local SQLite database in production.

## Step 2: Render Backend API

Create a Render Web Service.

Render settings:

```text
Name: interview-ai-api
Runtime: Python
Root Directory: leave empty
Build Command: pip install -r requirements.txt
Start Command: python api_server.py
Public URL: https://interview-ai-agent-kpc8.onrender.com
```

Important: this repo has `runtime.txt` with:

```text
python-3.11.11
```

It also has `.python-version` with:

```text
3.11.11
```

Render must use Python 3.11 for this backend. To force it, also add this Render environment variable:

```text
PYTHON_VERSION=3.11.11
```

If Render uses Python 3.14, dependency installation can fail at `pydantic-core` with:

```text
error: metadata-generation-failed
Encountered error while generating package metadata.
pydantic-core
```

That error is not your app code. It means Render selected a Python version that is too new for the pinned dependency set. Commit and push `runtime.txt` and `.python-version`, add `PYTHON_VERSION=3.11.11` in Render, then redeploy with **Clear build cache & deploy**.

The backend now reads Render's `PORT`, so this works:

```python
port = int(os.getenv("PORT") or os.getenv("API_PORT", 8001))
```

### Render Backend Environment Variables

Add these in Render service settings.

Use this exact table:

```text
ENV=production

API_SECRET_TOKEN=<generate a new production secret>
ALLOWED_ORIGINS=https://interview-ai-agent.vercel.app,http://localhost:3000,http://127.0.0.1:3000
PYTHON_VERSION=3.11.11

DATABASE_URL=<your Supabase Session Pooler connection string>

LIVEKIT_URL=<your public LiveKit URL, must be wss:// in production>
LIVEKIT_API_KEY=<your production LiveKit API key>
LIVEKIT_API_SECRET=<your production LiveKit API secret>

GROQ_API_KEY=<same provider key type as local, but preferably a production key>
DEEPGRAM_API_KEY=<same provider key type as local, but preferably a production key>

API_HOST=0.0.0.0
```

Generate `API_SECRET_TOKEN`:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Do not reuse the local dev token if this will be public.

### Backend Health Check

After Render deploys:

```text
https://interview-ai-agent-kpc8.onrender.com/docs
```

If `/docs` opens, the API process is running.

Then test:

```text
https://interview-ai-agent-kpc8.onrender.com/api/candidates
```

If you see `[]` or candidate JSON, the API and DB are alive.

## Step 3: LiveKit Production

### Recommended Option: LiveKit Cloud

This is the simplest and most reliable for production.

Use the values LiveKit Cloud gives you:

```text
LIVEKIT_URL=wss://YOUR_PROJECT.livekit.cloud
LIVEKIT_API_KEY=<LiveKit Cloud key>
LIVEKIT_API_SECRET=<LiveKit Cloud secret>
```

Put those same three values in:

- Render backend
- Fly.io agent

Do not put `LIVEKIT_API_SECRET` in Vercel.

### Self-Hosted Option

If you self-host, production LiveKit must have:

- HTTPS/WSS domain
- Public WebSocket access
- Public WebRTC media ports
- TURN setup if users are behind strict NAT
- Redis
- Production API key and secret, not `devkey` and not the local `secret123...`

Your local `config/livekit.yaml` has dev values:

```yaml
keys:
  devkey: secret1234567890123456789012345678901234567890
```

Do not use those in production.

Generate production LiveKit credentials:

```powershell
python -c "import secrets; print('key_' + secrets.token_urlsafe(12)); print(secrets.token_urlsafe(48))"
```

Example shape:

```text
LIVEKIT_API_KEY=key_xxxxxxxxxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### About LiveKit Docker On Render

I do not recommend Render for public LiveKit media.

Render docs say only one public HTTP port is forwarded to a web service. LiveKit needs more than that for real WebRTC media. If you still deploy it on Render, expect possible problems like:

- agent joins but audio does not flow
- browser connects then disconnects
- candidate can see room but cannot hear AI
- media works locally but not in production

If you still want to try it anyway, create a separate Render Docker web service for LiveKit and use `LIVEKIT_CONFIG` env var from the LiveKit docs, but treat it as experimental.

## Step 4: Fly.io Agent

The agent should not run on Vercel. It must be a long-running process or an event-started machine.

For first production launch, run the agent on Fly.io as a small machine. Once the normal flow works, add event-driven start/stop.

### Install Fly CLI

```powershell
winget install Fly.Flyctl
fly auth login
```

### Create Agent App

From the repo root:

```powershell
fly launch --name interview-ai-agent-worker --dockerfile agent/Dockerfile --no-deploy
```

Choose a nearby region.

### Set Fly Secrets

```powershell
fly secrets set LIVEKIT_URL="wss://YOUR-LIVEKIT-DOMAIN"
fly secrets set LIVEKIT_API_KEY="YOUR_PRODUCTION_LIVEKIT_KEY"
fly secrets set LIVEKIT_API_SECRET="YOUR_PRODUCTION_LIVEKIT_SECRET"
fly secrets set API_URL="https://interview-ai-agent-kpc8.onrender.com"
fly secrets set API_SECRET_TOKEN="YOUR_RENDER_API_SECRET_TOKEN"
fly secrets set GROQ_API_KEY="YOUR_GROQ_KEY"
fly secrets set DEEPGRAM_API_KEY="YOUR_DEEPGRAM_KEY"
fly secrets set TTS_PROVIDER="deepgram"
fly secrets set DEEPGRAM_TTS_MODEL="aura-2-andromeda-en"
```

Deploy:

```powershell
fly deploy
```

Check logs:

```powershell
fly logs
```

You want to see:

```text
Interview agent running.
plugin registered livekit.plugins.groq
plugin registered livekit.plugins.deepgram
plugin registered livekit.plugins.silero
registered worker
```

### Event-Driven Agent Plan

The event-driven version should work like this:

1. Candidate clicks start interview.
2. Render backend starts/resumes the Fly Machine using Fly Machines API.
3. Agent registers with LiveKit.
4. Candidate joins the LiveKit room.
5. Interview ends.
6. Agent posts evaluation to Render.
7. Render or the agent stops/suspends the Fly Machine.

For first production, do not start here. First prove the always-on Fly agent works. After that, add Fly Machines API integration.

The Fly Machines API supports starting and stopping machines. You need:

```text
FLY_API_TOKEN=<Fly token>
FLY_APP_NAME=interview-ai-agent-worker
FLY_MACHINE_ID=<machine id>
```

Keep those in Render only.

## Step 5: Vercel Frontend

Create Vercel project:

Recommended with the repo-level `vercel.json`:

```text
Framework: Next.js
Root Directory: leave empty / repository root
Install Command: npm --prefix dashboard install
Build Command: npm --prefix dashboard run build
Output Directory: dashboard/.next
Production URL: https://interview-ai-agent.vercel.app
```

Alternative without using root `vercel.json`:

```text
Framework: Next.js
Root Directory: dashboard
Install Command: npm install
Build Command: npm run build
Output Directory: leave default
```

If Vercel says `No Output Directory named "public" found`, your Vercel project is configured as a static site or has `public` set as the output directory. Change the Framework Preset to Next.js and remove `public` from Output Directory.

### Vercel Environment Variables

Only add this:

```text
NEXT_PUBLIC_API_URL=https://interview-ai-agent-kpc8.onrender.com
```

Optional for rewrites only:

```text
API_URL=https://interview-ai-agent-kpc8.onrender.com
```

Do not add these to Vercel:

```text
LIVEKIT_API_SECRET
LIVEKIT_API_KEY
GROQ_API_KEY
DEEPGRAM_API_KEY
DATABASE_URL
API_SECRET_TOKEN
```

Vercel is public frontend/serverless. Keep the important secrets on Render and Fly.

## Step 6: Production Test Checklist

Open:

```text
https://interview-ai-agent.vercel.app
```

Then:

1. Upload a resume.
2. Confirm candidate/interview is created.
3. Start interview.
4. Browser asks for mic permission.
5. AI joins the room.
6. AI speaks the greeting.
7. Candidate answers.
8. Transcript shows candidate answer.
9. AI status changes to thinking.
10. AI speaks the next question.
11. End the interview.
12. Evaluation saves to backend.

If any step fails, check in this order:

1. Render backend logs.
2. Fly agent logs.
3. LiveKit logs/dashboard.
4. Vercel browser console.

## Common Problems And Fixes

### Frontend says backend unreachable

Check Vercel:

```text
NEXT_PUBLIC_API_URL=https://interview-ai-agent-kpc8.onrender.com
```

Check Render CORS:

```text
ALLOWED_ORIGINS=https://interview-ai-agent.vercel.app,http://localhost:3000,http://127.0.0.1:3000
```

Redeploy Render after changing CORS.

### LiveKit token rejected

The same `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` must be used by:

- LiveKit server
- Render backend
- Fly agent

Vercel should not generate tokens.

### AI joins but no audio

Check:

```text
DEEPGRAM_API_KEY is set on Fly
LIVEKIT_URL is reachable from Fly
Only one agent worker is running
LiveKit media ports work in production
```

If LiveKit is on Render, this is likely the Render public UDP/media limitation.

### AI hears candidate but does not answer

Check Fly logs for:

```text
TypeError: 'coroutine' object does not support the asynchronous context manager protocol
```

That bug was fixed in this codebase. If it appears, redeploy the latest code to Fly.

### Database errors

Check Render:

```text
DATABASE_URL=<Supabase Session Pooler URL>
```

Make sure the Supabase password is URL encoded if it contains special characters like `@`, `#`, `/`, `?`, or `:`.

### Render service starts then exits

Check Start Command:

```text
python api_server.py
```

Check env:

```text
PORT is provided automatically by Render
API_HOST=0.0.0.0
```

### Agent starts locally by mistake

Do not run this in production:

```powershell
wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/Alexa/OneDrive/Desktop/Interview-AI-FINAL && python3 agent/run_all.py start"
```

That is local development only.

## Local Development Commands

Use these only on your PC:

Terminal 1:

```powershell
docker compose up -d livekit redis agent
docker compose logs -f agent
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe api_server.py
```

Terminal 3:

```powershell
npm --prefix dashboard run dev
```

Open:

```text
http://localhost:3000
```

## Final Production Env Summary

### Vercel

```text
NEXT_PUBLIC_API_URL=https://interview-ai-agent-kpc8.onrender.com
API_URL=https://interview-ai-agent-kpc8.onrender.com
```

### Render Backend

```text
ENV=production
API_HOST=0.0.0.0
API_SECRET_TOKEN=<new generated production token>
ALLOWED_ORIGINS=https://interview-ai-agent.vercel.app,http://localhost:3000,http://127.0.0.1:3000
DATABASE_URL=<Supabase Session Pooler URL>
LIVEKIT_URL=<wss:// production LiveKit URL>
LIVEKIT_API_KEY=<production LiveKit key>
LIVEKIT_API_SECRET=<production LiveKit secret>
GROQ_API_KEY=<production Groq key>
DEEPGRAM_API_KEY=<production Deepgram key>
```

### Fly Agent

```text
LIVEKIT_URL=<same production LiveKit URL as Render>
LIVEKIT_API_KEY=<same production LiveKit key as Render>
LIVEKIT_API_SECRET=<same production LiveKit secret as Render>
API_URL=https://interview-ai-agent-kpc8.onrender.com
API_SECRET_TOKEN=<same token as Render>
GROQ_API_KEY=<production Groq key>
DEEPGRAM_API_KEY=<production Deepgram key>
TTS_PROVIDER=deepgram
DEEPGRAM_TTS_MODEL=aura-2-andromeda-en
```

## Final Advice

For the least painful production launch:

1. Use Supabase Postgres.
2. Use Render for FastAPI.
3. Use Vercel for dashboard.
4. Use Fly.io for the agent.
5. Use LiveKit Cloud or a VPS/Fly-hosted LiveKit server, not Render, for real production audio/video.
