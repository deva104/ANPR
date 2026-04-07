import os
from src.semantic_segmentation.models.hrnet import hrnet
from src.semantic_segmentation.utils.util import (
    get_warped_plates,
    plate_locate,
    get_score_and_class_from_prediction,
    preprocess_image,
    upsample_coordinates,
    convert_coordinates_to_bbox,
)
from src.infer_video_utils import (
    annotated_video_path,
    is_video_file,
    run_video_phase1,
    safe_fps,
    video_writer,
)

from torch import nn as nn
import argparse
import cv2
import torch

from src.License_Plate_Recognition.model.LPRNet import build_lprnet
from src.License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference

import numpy as np
import json
from tqdm import tqdm


def run_single_frame(semantic_model, lprnet, image, conf_thresh):
    """[summary]

    Args:
        semantic_model ([type]): [description]
        lprnet ([type]): [description]
        image ([type]): [description]

    Returns:
        [type]: [description]
    """
    original_image = image.copy()
    image = preprocess_image(image)
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        out = semantic_model(image, (image.shape[2], image.shape[3]))
        prediction_softmax = nn.Softmax(dim=1)(out["output"])
        out = (
            torch.argmax(out["output"], dim=1)
            .detach()
            .cpu()
            .squeeze(dim=0)
            .numpy()
            .astype(np.uint8)
        )
        coordinates, _ = plate_locate(out)
        scores = get_score_and_class_from_prediction(out,prediction_softmax, coordinates)
        pred_boxes = convert_coordinates_to_bbox(coordinates)

        pred_boxes_new = []
        coordinates_new = []
        for box, score, c in zip(pred_boxes, scores, coordinates):

            if score[0] > conf_thresh:
                pred_boxes_new.append(box)
                coordinates_new.append(c)

        coordinates, boxes = upsample_coordinates(
            coordinates_new, out.shape, original_image.shape
        )

    if len(boxes) == 0:
        return {0: {"coordinates": [], "boxes": [], "label": ""}}

    plate_images = get_warped_plates(original_image, coordinates)
    plate_images_tensor = []
    for plate_image in plate_images:
        im = cv2.resize(plate_image, (94, 24)).astype("float32")
        im -= 127.5
        im *= 0.0078125
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        plate_images_tensor.append(im)

    plate_labels = Greedy_Decode_inference(lprnet, torch.stack(plate_images_tensor, 0))
    out_dict = {}

    for idx, (box_4pnt, box, label) in enumerate(zip(coordinates, boxes, plate_labels)):
        out_dict.update({idx: {"coordinates": box_4pnt, "boxes": box, "label": label}})

    return out_dict


def plot_single_frame_from_out_dict(im, out_dict,line_thickness=3,color = (255,0,0)):
    if out_dict:
      for _, v in out_dict.items():
        box, label = v["boxes"], v["label"]
        
        if len(box) < 4:
            continue

        tl = (
            line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
        )  # line/font thickness
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        mask = im.copy()
        mask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = color
        im = cv2.addWeighted(im, 0.7, mask, 0.3, 0)
        # cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                im,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
    return im


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def process_directory(args, semantic_model, lprnet):

    for i in tqdm(os.listdir(args.source)):
        p = os.path.join(args.source, i)
        ext = os.path.splitext(i)[1].lower()

        if is_video_file(p):
            process_video(
                p, semantic_model, lprnet, args.output_path, args.conf_thresh
            )
            continue

        if ext in IMAGE_EXTENSIONS:
            image = cv2.imread(p)
            out_dict = run_single_frame(semantic_model, lprnet, image, args.conf_thresh)

            plotted_image = plot_single_frame_from_out_dict(image, out_dict)

            cv2.imwrite(
                os.path.join(args.output_path, "plots", i),
                plotted_image,
            )

            with open(
                os.path.join(
                    args.output_path,
                    "jsons",
                    i.replace("jpg", "json").replace("png", "json"),
                ),
                "w",
            ) as outfile:
                json.dump({i: out_dict}, outfile, default=str)

    return


def process_video(video_path, semantic_model, lprnet, output_dir, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fps = safe_fps(cap)
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        print(f"No frames read from: {video_path}")
        return

    out_target = annotated_video_path(video_path, output_dir)
    writer, actual_out = video_writer(
        out_target, fps, (frame.shape[1], frame.shape[0])
    )
    final_dict = {}
    base = os.path.splitext(os.path.basename(video_path))[0]
    print(f"processing video -> {actual_out}")

    idx = 0
    pbar = tqdm(desc=base, unit="fr")
    while ret:
        out_dict = run_single_frame(semantic_model, lprnet, frame, conf_thresh)
        out_frame = plot_single_frame_from_out_dict(frame, out_dict)
        final_dict[idx] = out_dict
        writer.write(out_frame)
        pbar.update(1)
        ret, frame = cap.read()
        idx += 1
    pbar.close()
    cap.release()
    writer.release()

    json_name = base + ".json"
    with open(os.path.join(output_dir, "jsons", json_name), "w") as outfile:
        json.dump(final_dict, outfile, default=str)
    return


def process_txt(args, semantic_model, lprnet):
    txt_path = os.path.abspath(args.source)
    base_dir = os.path.dirname(txt_path)

    with open(txt_path, "r") as txt_file:
        for raw in txt_file:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            full = line if os.path.isabs(line) else os.path.join(base_dir, line)
            if not os.path.isfile(full):
                print(f"skip (not found): {full}")
                continue
            ext = os.path.splitext(full)[1].lower()

            if is_video_file(full):
                process_video(
                    full,
                    semantic_model,
                    lprnet,
                    args.output_path,
                    args.conf_thresh,
                )
            elif ext in IMAGE_EXTENSIONS:
                image = cv2.imread(full)
                if image is None:
                    print(f"skip (unreadable image): {full}")
                    continue
                name = os.path.basename(full)
                out_dict = run_single_frame(
                    semantic_model, lprnet, image, args.conf_thresh
                )

                plotted_image = plot_single_frame_from_out_dict(image, out_dict)

                cv2.imwrite(
                    os.path.join(args.output_path, "plots", name),
                    plotted_image,
                )

                stem = os.path.splitext(name)[0] + ".json"
                with open(
                    os.path.join(args.output_path, "jsons", stem),
                    "w",
                ) as outfile:
                    json.dump({name: out_dict}, outfile, default=str)

    return


if __name__ == "__main__":
    _ROOT = os.path.dirname(os.path.abspath(__file__))
    _WEIGHTS = os.path.join(_ROOT, "weights")

    parser = argparse.ArgumentParser()

    # add more formats based on what is supported by opencv
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image, folder, video, folder of videos, or .txt list. Videos: .mp4 .avi .mov .mkv. Images: .jpg .jpeg .png",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="confidence threshold for plate segmentation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Output root: plots/, jsons/, videos/ (annotated *_annotated.mp4)",
    )
    parser.add_argument("--detect_every", type=int, default=5)
    parser.add_argument("--detect_scale", type=float, default=0.67)
    parser.add_argument("--ocr_min_width", type=int, default=100)
    parser.add_argument("--ocr_min_height", type=int, default=28)
    parser.add_argument("--ocr_sharpness", type=float, default=80.0)
    parser.add_argument("--ocr_force_every", type=int, default=10)
    parser.add_argument("--vote_buffer", type=int, default=7)
    parser.add_argument("--live_preview", action="store_true")

    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "jsons"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "videos"), exist_ok=True)

    # load object detection model
    if torch.cuda.is_available():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    semantic_model = hrnet().eval()
    if torch.cuda.is_available():
        semantic_model = nn.SyncBatchNorm.convert_sync_batchnorm(semantic_model)
    semantic_model = nn.DataParallel(semantic_model)
    semantic_model.load_state_dict(
        torch.load(
            os.path.join(_WEIGHTS, "best_semantic.pth"),
            map_location=torch.device("cpu"),
        )["state_dict"]
    )

    # load ocr

    lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
    lprnet.load_state_dict(
        torch.load(
            os.path.join(_WEIGHTS, "best_lprnet.pth"),
            map_location=torch.device("cpu"),
        )
    )

    if torch.cuda.is_available():
        semantic_model = semantic_model.cuda()
        lprnet = lprnet.cuda()

    if os.path.isdir(args.source):
        print("source is directory, might need time to process")
        process_directory(args, semantic_model, lprnet)

    else:

        ext = os.path.splitext(args.source)[1].lower()

        if ext in IMAGE_EXTENSIONS:
            print("source is image")
            image = cv2.imread(args.source)
            out_dict = run_single_frame(semantic_model, lprnet, image, args.conf_thresh)
            if out_dict:
                plotted_image = plot_single_frame_from_out_dict(image, out_dict)

                cv2.imwrite(
                    os.path.join(args.output_path, "plots", "plotted_image.png"),
                    plotted_image,
                )

                with open(
                    os.path.join(args.output_path, "jsons", "output.json"), "w"
                ) as outfile:
                    json.dump(
                        {os.path.basename(args.source.rstrip("/\\")): out_dict},
                        outfile,
                        default=str,
                    )

        if is_video_file(args.source):
            print("source is video")
            def detector_fn(frame_bgr):
                image = preprocess_image(frame_bgr)
                if torch.cuda.is_available():
                    image = image.cuda()
                with torch.no_grad():
                    out = semantic_model(image, (image.shape[2], image.shape[3]))
                    prediction_softmax = nn.Softmax(dim=1)(out["output"])
                    pred = (
                        torch.argmax(out["output"], dim=1)
                        .detach()
                        .cpu()
                        .squeeze(dim=0)
                        .numpy()
                        .astype(np.uint8)
                    )
                    coordinates, _ = plate_locate(pred)
                    scores = get_score_and_class_from_prediction(
                        pred, prediction_softmax, coordinates
                    )
                    pred_boxes = convert_coordinates_to_bbox(coordinates)
                    coordinates_new = []
                    for score, c in zip(scores, coordinates):
                        if score[0] > args.conf_thresh:
                            coordinates_new.append(c)
                    _, boxes = upsample_coordinates(
                        coordinates_new, pred.shape, frame_bgr.shape
                    )
                out_boxes = []
                for b in boxes:
                    out_boxes.append(
                        [int(b[0]), int(b[1]), int(b[2]), int(b[3]), 1.0]
                    )
                return out_boxes

            def ocr_fn(crop_bgr):
                if crop_bgr is None or crop_bgr.size == 0:
                    return ""
                im = cv2.resize(crop_bgr, (94, 24)).astype("float32")
                im -= 127.5
                im *= 0.0078125
                im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
                labels = Greedy_Decode_inference(lprnet, torch.stack([im], 0))
                if not labels:
                    return ""
                return labels[0]

            run_video_phase1(
                video_path=args.source,
                output_path=args.output_path,
                detector_fn=detector_fn,
                ocr_fn=ocr_fn,
                detect_every=args.detect_every,
                detect_scale=args.detect_scale,
                ocr_min_width=args.ocr_min_width,
                ocr_min_height=args.ocr_min_height,
                ocr_sharpness=args.ocr_sharpness,
                ocr_force_every=args.ocr_force_every,
                vote_buffer=args.vote_buffer,
                live_preview=args.live_preview,
                conf_thresh=args.conf_thresh,
            )

        if ext == ".txt":
            print("source is txt, might need time to process")
            process_txt(args, semantic_model, lprnet)

    print("processing done")
