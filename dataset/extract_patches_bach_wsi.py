import argparse 
import numpy as np
import os
from PIL import Image
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import tifffile
import xml.etree.ElementTree as ET


def parse_xml_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotated_regions = []
    n = 0
    for annotation in root.findall(".//Annotation"):
        for region in annotation.find("Regions").findall("Region"):
            n += 1

            # Extract label
            label = "Unknown"
            if region.attrib['Text'] is not None: 
                label = region.attrib['Text'].replace(" ", "_")
            attributes = region.find("Attributes")
            if attributes is not None:
                for attr in attributes.findall("Attribute"):
                    if 'Value' in attr.attrib:
                        label = attr.attrib['Value'].replace(" ", "_")

            # Extract coordinates
            vertices = region.find("Vertices")
            coords = [(int(float(vertex.attrib["X"])), int(float(vertex.attrib["Y"])))
                      for vertex in vertices.findall("Vertex")]

            if coords:
                poly = Polygon(coords)
                annotated_regions.append({'label': label, 'polygon': poly})

    return annotated_regions


def is_background(patch_img, thresh, threshold_ratio=0.5):
    """
    Determine if a patch is mostly background using Otsu thresholding.
    patch_img: PIL Image
    Returns True if background ratio > threshold_ratio
    """
    otsu = False
    if otsu:
        # Convert to grayscale (float image in [0,1])
        gray = rgb2gray(np.array(patch_img))

        # Pixels brighter than threshold are considered background
        background_mask = gray > thresh
        background_ratio = np.mean(background_mask)

        return background_ratio > 0.5 #threshold_ratio
    else: 
        # Convert to grayscale (float image in [0,1])
        gray = rgb2gray(np.array(patch_img))

        std = np.std(gray)

        return std < 0.025


def extract_patches_from_tiff(tiff_path, xml_path, output_dir,
                              patch_size=256, overlap=0):
    """
    Extract patches from TIFF + XML using tifffile (no OpenSlide).
    """
    os.makedirs(output_dir, exist_ok=True)
    image = tifffile.imread(tiff_path)
    if image.ndim == 2:  # grayscale
        image = np.stack([image]*3, axis=-1)
    elif image.shape[0] in [3, 4]:  # (C, H, W) -> (H, W, C)
        image = np.transpose(image, (1, 2, 0))

    height, width = image.shape[:2]

    # Convert to grayscale (float image in [0,1])
    gray = rgb2gray(np.array(image))
    # Otsu threshold
    thresh = threshold_otsu(gray)

    # Load annotation polygons
    regions = parse_xml_annotations(xml_path)
    polygons = [r['polygon'] for r in regions]
    labels = [r['label'] for r in regions]
    tree = STRtree(polygons)

    count = 0
    for x in range(0, width - patch_size + 1, patch_size - overlap):
        for y in range(0, height - patch_size + 1, patch_size - overlap):
            patch_box = Polygon([
                                    (x, y),
                                    (x + patch_size, y),
                                    (x + patch_size, y + patch_size),
                                    (x, y + patch_size),
                                    (x, y)  # close the polygon
                                ])

            matched = False
            for poly in polygons:
                if patch_box.intersects(poly):
                    idx = polygons.index(poly)
                    label = labels[idx]
                    matched = True
                    break

            patch = image[y:y+patch_size, x:x+patch_size, :3]
            patch_img = Image.fromarray(patch.astype(np.uint8))

            if not matched:
                background = is_background(patch_img, thresh, threshold_ratio=0.5)
                
                if not background: 
                    label = 'Healthy'
                else:
                    label = 'Background'

            svs_name = tiff_path.split(os.sep)[-1].split(".svs")[0]
            fname = f"tile_{svs_name}_{x}_{y}_label_{label}.png"

            patch_img.save(os.path.join(output_dir, fname))
            count += 1

    print(f"Saved {count} patches from {os.path.basename(tiff_path)}")


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/auto/globalscratch/users/t/g/tgodelai/')
    parser.add_argument('--dataset', type=str, default='bach', choices=['bach'])
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--overlap', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parser()

    images_dir = os.path.join(args.data_dir, 'thunder', 'datasets', 'bach', 'ICIAR2018_BACH_Challenge', 'WSI')

    svs_files = [f for f in os.listdir(images_dir) if f.endswith('.svs')]
    out_dir = os.path.join(args.data_dir, 'thunder', 'datasets', 'bach', 'ICIAR2018_BACH_Challenge', f'patches_{args.patch_size}')

    files_to_extract = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10']

    for fname in svs_files: 
        if fname.split('.svs')[-1] in files_to_extract: 
            tiff_path = os.path.join(images_dir, fname)
            xmlname = fname.replace('.svs', '.xml')
            xml_path = os.path.join(images_dir, xmlname)
            if not os.path.exists(xml_path):
                print(f"No XML file for {fname}, skipping.")
                continue
             
            extract_patches_from_tiff(tiff_path, xml_path, out_dir,
                                    patch_size=args.tile_size, overlap=args.overlap)