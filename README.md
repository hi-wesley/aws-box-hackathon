# aws-box-hackathon

Local UI + Bedrock-backed chat over two files:
- `long-term-sales-data.csv`
- `Online Garment Tracking and Management System.pdf`

## Quick start
1. Install Node 18+ and Python (for a static file server).
2. Install deps: `npm install` (adds `pdf-parse` for summarizing the PDF).
3. Set AWS creds/region (use a profile or env vars): `AWS_REGION=us-east-1` plus `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_SESSION_TOKEN` if needed.
4. Start the Bedrock proxy: `node server.js` (defaults: `PORT=3001`, `BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0`, `MAX_TOKENS=500`).
5. Serve the UI: `python -m http.server 8000` and open `http://localhost:8000/` (or use `npx live-server --port=8000` for auto-reload). The UI is now React (via CDN + Babel) and auto-reloads on refresh, no build step required.
6. The UI shows the CSV/PDF on the left, pulls Purpose/Status from Bedrock on the top-right, and lets you chat on the bottom-right.

## Endpoints
- `POST /chat` body:`{ "prompt": "..." }` -> `{ "answer": "..." }`
- `GET /summaries` -> `{ purpose, status }` (Bedrock summarizes both files; trims the CSV + PDF text)
- `GET /health` -> `{ ok: true, model, region }`

Keep the CSV/PDF next to `index.html` so the preview/download buttons work. If your Bedrock model access differs, set `BEDROCK_MODEL_ID` before starting the server. Switch ports by changing `PORT` and updating `CHAT_API` in `index.html` if needed.
