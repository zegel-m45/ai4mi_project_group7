import nibabel as nib
import numpy as np

from scipy import ndimage
from pathlib import Path


ESOPHAGUS = 1
HEART = 2
TRACHEA = 3
AORTA = 4

# CHANGE
input_path = (
    "/home/kim/Documents/GitHub/ai4mi_project_group7/"
    "data/segthor_part1/train/Patient_03/GT.nii.gz"
)
output_path = (
    "/home/kim/Documents/GitHub/ai4mi_project_group7/"
    "data/segthor_part1/train/Patient_03/GT_voronoi.nii.gz"
)

img = nib.load(input_path)
data = img.get_fdata().astype(np.int16)
print("Shape:", data.shape) # (512, 512, 147)


def get_blobs_per_slice(merged_data):
    """
    Return the segments/blobs per slice, sorted by size.
    """
    # Connected-component labeling
    components, num = ndimage.label(merged_data)
    blobs = []

    for component_id in range(1, num + 1):

        # Retrieve an individual blob
        current_blob = components == component_id
        area = current_blob.sum()

        # Get coordinates of all pixels
        coords = np.argwhere(current_blob)
        center_y, center_x = coords.mean(axis=0)

        blobs.append({
            "blob": current_blob,
            "area": area,
            "center": (center_x, center_y),
        })

    # Sort by y coordinate (up to down for the 1st view in 3DSlicer)
    blobs.sort(key=lambda b: b["center"][1])

    return blobs

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_center(blob):
    coords = np.argwhere(blob)
    y, x = coords.mean(axis=0)
    return (x, y)

def get_bounding_box_heart(data):
    heart = data == HEART
    coords = np.argwhere(heart)
    y_min, x_min, z_min = coords.min(axis=0)
    y_max, x_max, z_max = coords.max(axis=0)

    return {
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
        "z_min": int(z_min),
        "z_max": int(z_max),
    }

def get_location_wrt_heart(z, heart_bbox):
    if z < heart_bbox["z_min"]:
        return "below_heart"
    if z > heart_bbox["z_max"]:
        return "above_heart"
    return "heart"


def voronoi_split(
    merged_blob,
    aorta_center,
    esophagus_center
):
    yy, xx = np.indices(merged_blob.shape)
    xa, ya = aorta_center
    xe, ye = esophagus_center

    dist_aorta = ((xx - xa) ** 2 + (yy - ya) ** 2)
    dist_esophagus = ((xx - xe) ** 2 + (yy - ye) ** 2)

    aorta_part = (merged_blob & (dist_aorta <= dist_esophagus))
    esophagus_part = (merged_blob & (dist_esophagus < dist_aorta))

    return aorta_part, esophagus_part


output = data.copy()
heart_bbox = get_bounding_box_heart(data)
previous_aorta_center = None
previous_esophagus_center = None

for z in range(data.shape[2]):
    phase = get_location_wrt_heart(z,heart_bbox)
    merged_data = data[:, :, z] == 1
    blobs = get_blobs_per_slice(merged_data)

    print(
        f"\nSlice {z}: "
        f"{len(blobs)} blobs"
        f"phase={phase}"
    )

    for i, blob in enumerate(blobs):
        ys, xs = np.where(blob["blob"])
        print(
            f"Blob {i}: "
            f"area={blob['area']}, "
            f"center={blob['center']}"
            f"x={xs.min()}..{xs.max()}, "
            f"y={ys.min()}..{ys.max()}, "
        )
    print("------")

    # TODO: Check for cases (other patients) if this rule-based method also holds
    if len(blobs) == 0:
        continue

    elif (phase == "below_heart") and len(blobs) == 1:
        blob = blobs[0]
        output[:, :, z][blob["blob"]] = AORTA
        previous_aorta_center = blob["center"]

    elif (phase == "below_heart" or phase == "heart") and len(blobs) == 2:
        top, bottom = blobs
        output[:, :, z][top["blob"]] = ESOPHAGUS
        output[:, :, z][bottom["blob"]] = AORTA

        previous_aorta_center = bottom["center"]
        previous_esophagus_center = top["center"]
        
    elif phase == "above_heart" and len(blobs) == 3:
        top, middle, bottom = blobs
        output[:, :, z][top["blob"]] = AORTA
        output[:, :, z][middle["blob"]] = ESOPHAGUS
        output[:, :, z][bottom["blob"]] = AORTA

        previous_aorta_center = (
            (top["center"][0] + bottom["center"][0]) / 2,
            (top["center"][1] + bottom["center"][1]) / 2,
        )
        previous_esophagus_center = middle["center"]

    elif phase == "above_heart" and len(blobs) == 2:
        largest = max(blobs, key=lambda b: b["area"])
        smallest = min(blobs, key=lambda b: b["area"])
        output[:, :, z][largest["blob"]] = AORTA
        output[:, :, z][smallest["blob"]] = ESOPHAGUS

        previous_aorta_center = largest["center"]
        previous_esophagus_center = smallest["center"]

    elif phase == "above_heart" and len(blobs) == 1:
        blob = blobs[0]
        merged_blob = blob["blob"]
        area = blob["area"]
        ESOPHAGUS_AREA_THRESHOLD = 300

        if area <= ESOPHAGUS_AREA_THRESHOLD:
            output[:, :, z][merged_blob] = ESOPHAGUS
            previous_esophagus_center = blob["center"]

        elif previous_aorta_center is not None and previous_esophagus_center is not None:
            aorta_part, esophagus_part = voronoi_split(
                merged_blob,
                previous_aorta_center,
                previous_esophagus_center
            )
            output[:, :, z][aorta_part] = AORTA
            output[:, :, z][esophagus_part] = ESOPHAGUS

            previous_aorta_center = get_center(aorta_part)
            previous_esophagus_center = get_center(esophagus_part)

    else:
        print("Other case happenin.......")


output_img = nib.Nifti1Image(
    output,
    img.affine,
    img.header
)
nib.save(output_img, output_path)
print(f"Saved output to: {output_path}")


