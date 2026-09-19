---
title: TrackSense
emoji: 🚆
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# TrackSense — POC

Railway track inspection dashboard with an AI defect-classification demo
(YOLOv8n-cls, ~87% validation accuracy).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 7860
```

Then open http://localhost:7860 and go to **AI Detection** in the sidebar.

## Deploy (Hugging Face Spaces, free)

1. Create a free account at https://huggingface.co/join
2. Click "New Space" → give it a name → SDK = **Docker** → Hardware = **CPU basic (free)** → Create.
3. On the new Space's page, use "Add file" → "Upload files" (or `git push`, see below)
   and upload every file in this folder, **including `best.pt`**.
4. The Space will build automatically (first build takes ~5-10 min since it
   installs PyTorch). When it says "Running", your public URL is:
   `https://huggingface.co/spaces/<your-username>/<space-name>`
5. Open that URL on your phone at the site visit — go to **AI Detection**,
   tap the upload box, take a photo, tap Analyze.

### Alternative: push with git instead of the upload UI

```bash
git clone https://huggingface.co/spaces/<your-username>/<space-name>
cp -r * <space-name>/
cd <space-name>
git add .
git commit -m "TrackSense POC"
git push
```

## Notes

- The model is a **classifier**, not an object detector — it returns a class
  label + confidence for the whole photo, not bounding boxes.
- Free HF Spaces sleep after ~48h of inactivity and take ~30-60s to wake up
  on the next visit — fine for a demo, just don't be alarmed by the first
  load being slow.
- SQLite data (inspection history) resets whenever the Space restarts/sleeps
  on the free tier, since the filesystem isn't persistent. Fine for a POC;
  if you need it to persist, Spaces support a paid persistent-storage add-on.
