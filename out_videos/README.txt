Annotated videos and per-frame JSON are written under your chosen --output_path.

Recommended layout:
  out_videos\objectdet\   — FCOS + LPRNet (--output_path "out_videos\objectdet")
  out_videos\semanticseg\ — HRNet segmentation + LPRNet (--output_path "out_videos\semanticseg")

Each run creates:
  videos\   — annotated video (*_annotated.mp4, or .avi fallback on Windows)
  jsons\    — one JSON per input video with frame index -> detections
  plots\    — used when processing images/folders of images

Large generated files are normally not committed to git.
