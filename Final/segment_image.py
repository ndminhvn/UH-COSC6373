import torchvision
import cv2
import torch
import argparse
import time
import segmentation_utils


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_model(name, device):
    if name == "deeplabv3":
        print("USING DEEPLABV3 WITH MOBILENETV3 BACKBONE")
        model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
            pretrained=True
        )
    else:
        print("USING LITE R-ASPP WITH MOBILENETV3 BACKBONE")
        model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            pretrained=True
        )

    model = model.eval().to(device)
    return model


def segment_image(input_path, model, device, output_path):
    # Read and convert to RGB
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Inference
    start = time.time()
    with torch.no_grad():
        outputs = segmentation_utils.get_segment_labels(img_rgb, model, device)
    end = time.time()
    fps = 1.0 / (end - start)
    print(f"Inference time: {end - start:.3f}s ({fps:.1f} FPS)")

    # Build overlay
    seg_map = segmentation_utils.draw_segmentation_map(outputs["out"])
    overlaid = segmentation_utils.image_overlay(img_rgb, seg_map)

    # Convert back to BGR for saving with OpenCV
    overlaid_bgr = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlaid_bgr)
    print(f"Saved segmented image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Image semantic segmentation")
    parser.add_argument(
        "-i", "--input", required=True, help="path to input image (jpg/png/etc.)"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=["deeplabv3", "lraspp"],
        help="which pretrained model to use",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="path to save the output image (defaults to input_seg.png)",
    )
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.model, device)

    # Determine output filename
    if args.output:
        out_path = args.output
    else:
        base = args.input.rsplit(".", 1)[0]
        out_path = f"{base}_{args.model}_seg.png"

    segment_image(args.input, model, device, out_path)


if __name__ == "__main__":
    main()
