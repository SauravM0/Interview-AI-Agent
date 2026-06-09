# Interview AI Agent — 2-Day Deploy Guide

> Goal: live URL you can show at job interviews in 48 hours, using only free services.

---

## Architecture (after cleanup)

```
Browser (Vercel)  →  FastAPI on Render  →  SQLite or Supabase
                  →  LiveKit Cloud (free) or local Docker
                  →  Agent worker (local or Fly.io)
```

---

## Step 0 — Prerequisites (30 min)

Create free accounts (no credit card required for any of these):

| Service | URL | What for |
|---|---|---|
| Google AI Studio | aistudio.google.com | Gemini API key (LLM) |
| Deepgram | console.deepgram.com | STT key ($200 free credit) |
| LiveKit Cloud | cloud.livekit.io | WebRTC + agent runner |
| Supabase | supabase.com | PostgreSQL database |
| Vercel | vercel.com | Frontend hosting |
| Render | render.com | API backend hosting |

---

## Step 1 — Fix the code (Day 1 AM, ~1 hour)

Apply the patches from the fixed files. Three critical changes:

### 1a. Change the LLM model name

In `agent/main.py` and `.env`:
```bash
# .env
GEMINI_MODEL=gemini-2.5-flash
```

### 1b. Install Kokoro TTS (replaces paid Deepgram TTS)

```bash
cd agent
pip install livekit-plugins-kokoro
```

The agent already has the Kokoro swap — it imports `kokoro.TTS()` automatically if available.

### 1c. Slim the root requirements.txt

```bash
# Remove these from requirements.txt (never used):
# langchain, langchain-community, langchain-google-genai
# chromadb, sentence-transformers, docx2txt, langchain-text-splitters, av
pip install -r requirements.txt  # now ~10 packages vs ~20
```

### 1d. Delete dead code

```bash
rm -rf interviews/
rm -rf database/chroma_db/
rm rag_processor.py
rm agent/agent_logic.py
rm agent/transcript_collector.py
rm frontend_scaffold/
```

### 1e. Test locally

```bash
# Terminal 1: LiveKit (Docker)
docker compose up livekit redis

# Terminal 2: API server
cp .env.example .env
# Edit .env with your keys
python api_server.py

# Terminal 3: Agent
cd agent && python main.py dev

# Terminal 4: Frontend
cd dashboard && npm install && npm run dev
# Open http://localhost:3000
```

Upload a resume PDF, click Start Interview, verify the agent speaks to you.

---

## Step 2 — Set up Supabase (Day 1 PM, ~20 min)

Supabase gives you a real PostgreSQL database — no ephemeral filesystem issues.

1. Go to [supabase.com](https://supabase.com) → New project
2. Choose a region close to you
3. Wait ~2 minutes for provisioning
4. Go to **Settings → Database → Connection string → URI**
5. Copy the connection string — it looks like:
   ```
   postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
   ```
6. Add to your `.env`:
   ```bash
   DATABASE_URL=postgresql://postgres.[ref]:[password]@...
   ```
7. Test:
   ```bash
   python -c "from scheduler.database import init_db; init_db()"
   # Should print: Database ready.
   ```

The app will auto-create all tables on first startup. No migrations needed.

---

## Step 3 — Deploy frontend to Vercel (Day 2 AM, ~20 min)

1. Push your code to GitHub (or fork the repo)

2. Go to [vercel.com](https://vercel.com) → New Project → Import Git Repo

3. Set **Root Directory** to `dashboard`

4. Add environment variables in Vercel dashboard:
   ```
   NEXT_PUBLIC_API_URL      = https://your-api.onrender.com
   NEXT_PUBLIC_LIVEKIT_URL  = wss://your-project.livekit.cloud
   LIVEKIT_API_KEY          = your_livekit_api_key
   LIVEKIT_API_SECRET       = your_livekit_api_secret
   ```

5. Click Deploy → live URL appears in ~2 minutes

---

## Step 4 — Deploy API to Render (Day 2 PM, ~30 min)

1. Go to [render.com](https://render.com) → New Web Service

2. Connect your GitHub repo

3. Configuration:
   ```
   Name:          interview-ai-api
   Root Directory: (leave empty — root of repo)
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn api_server:app --host 0.0.0.0 --port $PORT
   Instance Type: Free (or $7/mo for no cold starts)
   ```

4. Add all environment variables from `.env`:
   ```
   API_SECRET_TOKEN    = your_secret_token
   LIVEKIT_URL         = wss://your-project.livekit.cloud
   LIVEKIT_API_KEY     = your_key
   LIVEKIT_API_SECRET  = your_secret
   GOOGLE_API_KEY      = your_google_key
   GEMINI_API_KEY      = your_google_key
   GEMINI_MODEL        = gemini-2.5-flash
   DEEPGRAM_API_KEY    = your_deepgram_key
   DATABASE_URL        = your_supabase_connection_string
   ALLOWED_ORIGINS     = https://your-app.vercel.app
   ```

5. Deploy → your API URL is `https://interview-ai-api.onrender.com`

6. Update `NEXT_PUBLIC_API_URL` in Vercel to point to this URL, then redeploy Vercel.

---

## Step 5 — Run the agent (for demos)

For portfolio demos, the simplest approach is to run the agent on your **local machine**.
This avoids any infra cost and has no minute limits.

```bash
# Make sure LiveKit Cloud keys are in .env
cd agent
python main.py dev
```

The agent connects to LiveKit Cloud over WebSocket — no open ports needed on your machine.

When showing the demo to an interviewer:
1. Keep this terminal running
2. Open your Vercel URL on your laptop
3. Upload a resume → interview starts immediately

### Optional: Deploy agent to Fly.io (free 3 VMs)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

cd agent
# Create fly.toml
fly launch --name interview-ai-agent --no-deploy
fly secrets set GOOGLE_API_KEY=... LIVEKIT_URL=wss://... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... DEEPGRAM_API_KEY=...
fly deploy
```

fly.toml:
```toml
app = "interview-ai-agent"
primary_region = "ord"

[build]
  dockerfile = "Dockerfile"

[processes]
  app = "python main.py start"

[env]
  API_URL = "https://your-api.onrender.com"
```

---

## Free tier summary

| Service | Free allowance | Notes |
|---|---|---|
| Vercel | Unlimited deploys, 100GB/mo | Frontend — genuinely free forever |
| Render | 750 hrs/mo | Spins down after 15 min idle — use keep-alive ping for demos |
| Supabase | 500MB, 2 projects | Pauses after 7 days inactive, resumes instantly on request |
| LiveKit Cloud Build | 1,000 agent min/mo | ~33 x 30-min interviews/mo — plenty for demos |
| Gemini 2.5 Flash | 10 RPM, 250 RPD | Enough for demos; upgrade to paid if needed |
| Deepgram STT | $200 credit | ~400 hrs transcription — lasts months |
| Kokoro TTS | Free forever | No account, no key, runs locally in agent process |

**Total monthly cost for portfolio demos: $0**

---

## Troubleshooting

### Agent doesn't respond
- Check `GEMINI_MODEL=gemini-2.5-flash` in `.env` (not gemini-3-flash-preview)
- Check `GOOGLE_API_KEY` is set and valid
- Run agent with `python main.py dev` and watch the terminal for self-test result

### Interview settings are generic (no role/focus)
- The interview settings form must be submitted BEFORE clicking Start Interview
- Check that settings are saved: `GET /api/interviews/settings?room_name=YOUR_ROOM`

### End Turn button doesn't work
- This is fixed in the patched `agent/main.py` (data_received signature fix)
- Ensure you're using the patched version

### Render cold start (30-60s delay)
- Free tier spins down after 15 min idle
- For demos: visit your Render URL 2 min before to warm it up
- Or upgrade to $7/mo Render Starter for always-on

### Supabase paused
- Go to supabase.com dashboard → resume project (takes ~30s)
- Or add a keep-alive ping (free UptimeRobot → ping your API health endpoint)
