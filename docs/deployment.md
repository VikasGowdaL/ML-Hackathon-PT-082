# Two-Laptop Deployment (Final Evaluation)

| Laptop | Runs | Needs |
|---|---|---|
| **Laptop 1 (API)** | FastAPI backend: `python scripts/run_api.py` | Repo + `.venv`, trip data in `data/raw/` (incl. `data/raw/evaluator_test/`), model artifacts in `artifacts/models/` |
| **Laptop 2 (UI)** | React UI: `npm run dev` (or `npm run preview`) in `frontend/` | Repo + Node.js ≥ 20. No data and no model files |

Both laptops must be on the **same network** (same Wi-Fi or a phone hotspot). Many university / guest Wi-Fi networks block
traffic between devices ("client isolation"); a phone hotspot is the safest fallback.

## Laptop 1 – API

```powershell
cd ML-Hackathon-PT-082
.venv\Scripts\Activate.ps1
python scripts/run_api.py            # binds 0.0.0.0:8000
```

It prints the addresses to use, for example:

```
  from the LAN: http://10.20.205.133:8000/api/health   -> on Laptop 2 set VITE_API_URL=http://10.20.205.133:8000
```

Use the address of the **Wi-Fi / Ethernet** adapter. Ignore virtual adapters such as `192.168.56.1` (VirtualBox / Hyper-V).
Check with `ipconfig` (Windows) or `ipconfig getifaddr en0` (macOS).

### Firewall (Windows, once, PowerShell as Administrator)

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private,Public
```

When Windows shows "Allow Python to communicate on these networks", tick both private and public. On macOS, allow incoming
connections for Python when prompted.

### CORS (who may call the API)

By default the API accepts requests from `localhost` and from any **private network address** (`10.x.x.x`,
`192.168.x.x`, `172.16–31.x.x`) on any port, so no change is needed for a LAN demo. To restrict it to the UI laptop only:

```powershell
$env:PATHSNAP_CORS_ORIGINS = "http://10.20.205.140:5173"   # Laptop 2's address
python scripts/run_api.py
```

`PATHSNAP_CORS_ORIGINS="*"` allows every origin (debugging only).

## Laptop 2 – UI

```powershell
cd ML-Hackathon-PT-082\frontend
copy .env.example .env.local        # then edit: VITE_API_URL=http://<Laptop 1 IP>:8000
npm ci
npm run dev                          # http://localhost:5173
```

Open `http://localhost:5173` on Laptop 2.

* **Changing the API address without rebuilding:** open `http://localhost:5173/?api=http://<Laptop 1 IP>:8000`. The value is
  remembered in this browser. `VITE_API_URL` in `.env.local` takes precedence over a remembered value; a `?api=` in the URL
  always wins.
* If nothing is configured, the UI calls port 8000 on the same host it was loaded from.
* Production build instead of the dev server: `npm run build` then `npm run preview` (same port 5173).

## Pre-demo checks (in this order)

1. **Laptop 1:** `http://127.0.0.1:8000/api/health` in a browser → `{"status":"ok", ...}`.
2. **Laptop 2:** `http://<Laptop 1 IP>:8000/api/health` in a browser. If this fails, it is the network or firewall,
   not the code.
3. **Laptop 2:** `http://<Laptop 1 IP>:8000/docs` shows the Swagger UI.
4. **Laptop 2:** open the UI; the trip list loads. The UI's error message shows which API address it tried.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Step 2 times out | Firewall rule missing, wrong IP (virtual adapter), or the Wi-Fi isolates clients → use a phone hotspot |
| Step 2 works, UI says "Unable to connect … at http://127.0.0.1:8000" | `VITE_API_URL` not set on Laptop 2 → set `.env.local` and restart `npm run dev`, or use `?api=` |
| Browser console: "blocked by CORS policy" | UI opened from a non-private address → set `PATHSNAP_CORS_ORIGINS` on Laptop 1 |
| Laptop 1's IP changed after reconnecting | Restart `run_api.py` to see the new address and update Laptop 2 |
