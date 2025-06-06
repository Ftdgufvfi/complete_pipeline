import argparse
import re
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from model import DCENet
import torchvision
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDir', type=str, required=True, help='path to test images')
    parser.add_argument('--ckpt', type=str, required=True, help='path to *_best_model.pth')
    parser.add_argument('--outDir', type=str, required=True, help='path to save output')
    parser.add_argument('--imFormat', type=str, default='jpg',
                        help='file extension, which will be considered, default to image file named *.jpg')
    args = parser.parse_args()
    return args


def make_grid(nrow, ncol, h, w, hspace, wspace):
    grid = np.ones(
        (nrow * h + hspace * (nrow - 1), ncol * w + (ncol - 1) * wspace, 3),
        dtype=np.float32
    )
    return grid


def read_image(fp, h, w):
    fp = str(fp)
    img = cv2.imread(fp)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = cv2.resize(img, (w, h))
    return img


def putText(im, text, pos, color, size=1, scale=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    im = cv2.putText(im, text, pos, font, size, color, scale)
    return im


def row_arrange(wspace, images):
    n = len(images)
    h, w, c = images[0].shape
    row = np.ones((h, (n - 1) * wspace + n * w, c))
    curr_w = 0
    for image in images:
        row[:, curr_w:curr_w + w, :] = image
        curr_w += w + wspace
    return row


def main():
    args = parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DCENet(return_results=[4, 8])
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    inPath = Path(args.testDir)
    outPath = Path(args.outDir)
    outPath.mkdir(parents=True, exist_ok=True)

    r = re.compile(args.imFormat, re.IGNORECASE)  # e.g., jpg or png

    num_images = 0
    for file in inPath.glob('*'):
        if r.search(str(file)):
            img = Image.open(file)

            img = torch.from_numpy(np.array(img))
            img = img.float().div(255)
            img = img.permute((2, 0, 1)).contiguous()
            img = img.unsqueeze(0).to(device)

            with torch.no_grad():
                results, Astack = model(img)
            enhanced_image = results[1]
            torchvision.utils.save_image(enhanced_image, outPath.joinpath('enhanced_' + file.name))

            num_images += 1

    # Visualization grid
    h, w = 384, 512
    hspace, wspace = 8, 4
    ncol = 4
    nrow = (num_images + ncol - 1) // ncol
    grid = make_grid(nrow, ncol, h, w, hspace=hspace, wspace=wspace)
    curr_h = 0
    curr_row = 0
    curr_col = 0

    for file in inPath.glob('*'):
        if r.search(str(file)):
            ori = read_image(str(file), h, w)
            enh_our = read_image(outPath.joinpath('enhanced_' + file.name), h, w)
            gamma_fixed = utils.gamma_correction(ori, 0.4)
            gamma_alike = utils.gamma_like(ori, enh_our)

            pos = (20, 42)
            color = (1.0, 1.0, 1.0)
            ori = putText(ori, 'Input', pos, color=color)
            enh_our = putText(enh_our, 'ZeroDCE', pos, color=color)
            gamma_fixed = putText(gamma_fixed, 'Gamma(0.4)', pos, color=color)
            gamma_alike = putText(gamma_alike, 'Gamma(adaptive)', pos, color=color)

            row = row_arrange(wspace, [ori, enh_our, gamma_fixed, gamma_alike])

            # Calculate grid position
            row_idx = curr_row * (h + hspace)
            col_idx = 0
            grid[row_idx:row_idx + h, col_idx:col_idx + row.shape[1], :] = row

            curr_row += 1

    grid_image = Image.fromarray((grid * 255).astype(np.uint8), mode='RGB')
    grid_image.save(outPath.joinpath('comparison.pdf'))

    print(f"Processed {num_images} images. Results saved to {outPath}")


if __name__ == "__main__":
    main()
