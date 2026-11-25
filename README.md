# Fasal Doctor

Photo-based crop diagnosis for farmers: big-button, voice-first, in English, Hindi, Tamil and Telugu.
(c) 2026 Stormchasers Team. All rights reserved.

## What is in this folder
- `index.html`, `manifest.json`, `icon.svg` - the app screens (all in the top folder so they upload easily)
- `netlify/functions/api.mjs` - the backend (AI diagnosis, owner login, photo storage using Netlify Blobs)
- `netlify.toml`, `package.json`
- `tools/train.py` - trains a small model from your photos

## Put it on Netlify
1. Put this folder on GitHub (upload all files; do not upload any file with passwords).
2. On netlify.com: Add new site > Import an existing project > pick the GitHub repo. Leave the build settings as they are (netlify.toml sets them).
3. Site configuration > Environment variables. Add:
   - `ANTHROPIC_API_KEY` = your key from console.anthropic.com (this is paid per photo)
   - `OWNER_PASSWORD` = the owner password (8+ characters)
4. Deploy (Deploys > Trigger deploy). Your site gets a public web address; you can add your own domain under Domain management.

## Owner password
Set ONLY as `OWNER_PASSWORD` in Netlify settings. It is never in the code. There is no sign-up and no change-password button; to change it, edit that setting and redeploy. Protect your Netlify and GitHub accounts with two-step verification.

## Owner tools (bottom of the app)
- Upload reference photos with the disease name: pick many photos and type the label, or (on a computer) upload a folder with one subfolder per disease. The AI sees a couple of these as examples when diagnosing (`REF_COUNT`, default 2).
- See and delete all stored photos, including farmers' photos.

## Farmer photos and consent
Photos are saved only when the farmer taps the "Yes, share" button. Saved: photo, crop, diagnosis, season, weather, district. Not saved: name, phone, GPS.

## Training a model
`tools/train.py` downloads your photos and trains a small phone-friendly model (saved as `fasal_model.tflite`). Run it in free Google Colab:
`pip install tensorflow` then `FASAL_URL=https://your-site OWNER_PASSWORD=... python train.py`
About 30 photos per disease is the minimum. The model is not connected to the app yet.

## Known limits
- Netlify functions stop after a short time. If diagnosis is slow it can time out; keep `REF_COUNT` low and leave web search off.
- `ENABLE_WEB_SEARCH=true` lets the AI cross-check on websites. It is slower, costs extra, and is untested.
- Login and request limits use stored counters and are approximate.
