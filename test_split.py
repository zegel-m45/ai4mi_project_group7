import nibabel as nib
import numpy as np
import os
from scipy import ndimage

# Labels
ESOPHAGUS = 1
HEART = 2
TRACHEA = 3
AORTA = 4

MIN_BLOB_AREA = 20        # Minimal area to consider it a blob
ESOPHAGUS_MAX_SIZE = 400  # Maximal size used for classification of esophagus

PATIENT = "Patient_01"
BASE_PATH = os.path.expandvars("$HOME/ai4mi_project_group7/data/segthor_part1/train")
INPUT_PATH = f"{BASE_PATH}/{PATIENT}/GT.nii.gz"
OUTPUT_PATH = f"{BASE_PATH}/{PATIENT}/GT_split.nii.gz"


class Blob:
    def __init__(self, blob):
        self.blob = blob
        coords = np.argwhere(blob)
        self.center_y, self.center_x = coords.mean(axis=0)
        self.center = (self.center_x, self.center_y)
        self.area = blob.sum()
        self.circularity = get_circularity(blob)


class Organ:
    def __init__(self, name):
        self.name = name
        self.center = None

    def update(self, blob):
        self.center = blob.center


def get_blobs_per_slice(slice):
    # Assign an integer label to each connected component (vertical, horizontal, diagonal)
    components, num = ndimage.label(slice)
    blobs = []

    for component_id in range(1, num + 1):
        current_blob = components == component_id 
        area = current_blob.sum()

        if area < MIN_BLOB_AREA:
            continue

        blobs.append(Blob(current_blob))

    # Sort by coordinate: up to down for the 1st view in 3DSlicer
    blobs.sort(key=lambda blob: blob.center[0])

    return blobs


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_circularity(blob):
    area = blob.sum()
    eroded = ndimage.binary_erosion(blob)
    boundary = blob ^ eroded # XOR
    perimeter = boundary.sum()

    if perimeter == 0:
        return 0.0

    # Isoperimetric quotient​: https://www.bohrium.com/en/sciencepedia/feynman/keyword/isoperimetric_quotient
    # Number that tells us how "circular" a shape is: as aorta is often more circular than esophagus
    return 4 * np.pi * area / (perimeter ** 2)


def approximately_same_size(area1, area2, tolerance=0.55):
    larger = max(area1, area2)
    smaller = min(area1, area2)
    return smaller / larger >= (1 - tolerance)


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


def erosion_and_split(merged_blob, 
                    aorta_center, 
                    esophagus_center, 
                    max_erosion_iterations=30, 
                    min_island_area=1
    ):
    eroded = None
    erosion_iterations = None

    for i in range(1, max_erosion_iterations + 1):
        candidate = ndimage.binary_erosion(merged_blob, structure=np.ones((3, 3)), iterations=i)
        labeled, num = ndimage.label(candidate)

        # The binary erosion did not lead to a separate island: apply erosion again
        if num < 2:
            continue

        islands = []

        for island_id in range(1, num + 1):
            current_island = labeled == island_id
            area = current_island.sum()

            if area >= min_island_area:
                islands.append(Blob(current_island))

        if len(islands) >= 2:
            eroded = candidate
            erosion_iterations = i
            break

    if eroded is None:
        return None, None

    print(f"Split after {erosion_iterations} erosion iterations, "f"{len(islands)} islands")

    aorta_island = min(islands, key=lambda island: distance(island.center, aorta_center))
    remaining = [island for island in islands if island is not aorta_island]

    if len(remaining) == 0:
        return None, None

    esophagus_island = min(remaining, key=lambda island: distance(island.center, esophagus_center))

    aorta_grown = ndimage.binary_dilation(aorta_island.blob, structure=np.ones((3, 3)), iterations=erosion_iterations)
    esophagus_grown = ndimage.binary_dilation(esophagus_island.blob, structure=np.ones((3, 3)), iterations=erosion_iterations)

    # Never grow outside the original segmentation
    aorta_grown &= merged_blob
    esophagus_grown &= merged_blob

    # Resolve overlap
    overlap = aorta_grown & esophagus_grown

    if np.any(overlap):
        y, x = np.indices(merged_blob.shape)

        dist_to_aorta = np.sqrt((x - aorta_island.center[0]) ** 2 + (y - aorta_island.center[1]) ** 2)
        dist_to_esophagus = np.sqrt((x - esophagus_island.center[0]) ** 2 + (y - esophagus_island.center[1]) ** 2)

        aorta_overlap = overlap & (dist_to_aorta <= dist_to_esophagus)
        esophagus_overlap = overlap & ~aorta_overlap

        aorta_grown[overlap] = False
        esophagus_grown[overlap] = False
        aorta_grown[aorta_overlap] = True
        esophagus_grown[esophagus_overlap] = True

    return aorta_grown, esophagus_grown


def below_heart_cases(blobs, output, z, aorta_1, aorta_2, esophagus):
    # Lower aorta appears first, solo
    if len(blobs) == 1:
        blob = blobs[0]
        output[:, :, z][blob.blob] = AORTA
        aorta_2.update(blob)
    
    # Lower aorta and esophaghus are not merged: split on location
    elif len(blobs) == 2:
        top, bottom = blobs
        output[:, :, z][top.blob] = ESOPHAGUS
        output[:, :, z][bottom.blob] = AORTA

        esophagus.update(top)
        aorta_2.update(bottom)

    # The upper aorta always shows up above the heart only.
    # The minimum blob size makes sure this does not occur (or haven't seen it yet)
    else:
        print(">= 3 blobs not handled yet.")


def during_heart_cases(blobs, output, z, aorta_1, aorta_2, esophagus):
    if len(blobs) == 1:
        blob = blobs[0]

        # Likely the aorta only
        if blob.circularity > 0.95: 
            output[:, :, z][blob.blob] = AORTA
            aorta_2.update(blob)

        # Probably merged aorta and esophaghus
        else:
            aorta_part, esophagus_part = erosion_and_split(
                merged_blob=blob.blob,
                aorta_center=aorta_2.center,
                esophagus_center=esophagus.center,
            )

            if aorta_part is not None:
                output[:, :, z][aorta_part] = AORTA
                aorta_2.update(Blob(aorta_part))

            if esophagus_part is not None:
                output[:, :, z][esophagus_part] = ESOPHAGUS
                esophagus.update(Blob(esophagus_part))
    
    # Lower aorta and esophaghus are not merged: split on location
    elif len(blobs) == 2:
        top, bottom = blobs

        output[:, :, z][top.blob] = ESOPHAGUS
        output[:, :, z][bottom.blob] = AORTA

        esophagus.update(top)
        aorta_2.update(bottom)
    
    # For one case only seen in patient 11 - slice 80.
    # I think then we have 2 esophaghus classifications.
    elif len(blobs) == 3:
        # Aorta is largest blob
        aorta_blob = max(blobs, key=lambda b: b.area)
        esophagus_blobs = [blob for blob in blobs if blob is not aorta_blob]

        output[:, :, z][aorta_blob.blob] = AORTA
        aorta_2.update(aorta_blob)

        for blob in esophagus_blobs:
            output[:, :, z][blob.blob] = ESOPHAGUS
            # Skip updating the center for this case, not sure if correct.
       
    else: 
        print(">= 4 blobs not handled yet.")


def above_heart_cases(blobs, output, z, aorta_1, aorta_2, esophagus):
    if len(blobs) == 1:
        blob = blobs[0]

        # Hardcoding, only for the patient with the most incorrect splits.
        if PATIENT == "Patient_05":
            threshold = 650
        else:
            threshold = ESOPHAGUS_MAX_SIZE

        # Only esophaghus (usually near top).
        if blob.area <= threshold:
            output[:, :, z][blob.blob] = ESOPHAGUS
            esophagus.update(blob)
        
        # All three (aorta_1, aorta_2 and esophagus) are merged
        else:
            if aorta_1.center is not None and aorta_2.center is not None:
                aorta_center = (
                    (aorta_1.center[0] + aorta_2.center[0]) / 2,
                    (aorta_1.center[1] + aorta_2.center[1]) / 2,
                )
            else:
                aorta_center = aorta_2.center

            aorta_part, esophagus_part = erosion_and_split(
                merged_blob=blob.blob,
                aorta_center=aorta_center,
                esophagus_center=esophagus.center,
            )

            if aorta_part is not None:
                output[:, :, z][aorta_part] = AORTA
                aorta_1.update(Blob(aorta_part))
                aorta_2.update(Blob(aorta_part))

            if esophagus_part is not None:
                output[:, :, z][esophagus_part] = ESOPHAGUS
                esophagus.update(Blob(esophagus_part))

            if aorta_part is None or esophagus_part is None: 
                print( f"1 BLOB - NO EROSION SPLIT - slice {z}" )

    elif len(blobs) == 2:
        first, second = blobs
        
        # Both small, split on location
        if first.area < 450 and second.area < 450:
            left, right = sorted(blobs, key=lambda b: b.center[1])
            output[:, :, z][left.blob] = ESOPHAGUS
            output[:, :, z][right.blob] = AORTA

            esophagus.update(left)
            aorta_1.update(right) # Aorta's are the same now
            aorta_2.update(right)

        # Both ~same size, probably (upper aorta) + (esophagus merged with lower aorta)
        elif approximately_same_size(first.area, second.area):
            # Upper aorta
            output[:, :, z][first.blob] = AORTA
            aorta_1.update(first)
            
            aorta_part, esophagus_part = erosion_and_split(
                merged_blob=second.blob, 
                aorta_center=aorta_2.center,
                esophagus_center=esophagus.center,
                )

            if aorta_part is not None:
                output[:, :, z][aorta_part] = AORTA
                aorta_2.update(Blob(aorta_part))
            
            if esophagus_part is not None:
                output[:, :, z][esophagus_part] = ESOPHAGUS
                esophagus.update(Blob(esophagus_part))

            if aorta_part is None or esophagus_part is None: 
                print( f"APPROX SAME SIZE - NO EROSION SPLIT - slice {z}" )
                            
        # Smaller = esophagus, larger = aorta
        else:
            esophagus_blob = min(blobs, key=lambda b: b.area)
            aorta_blob = max(blobs, key=lambda b: b.area)

            output[:, :, z][esophagus_blob.blob] = ESOPHAGUS
            output[:, :, z][aorta_blob.blob] = AORTA

            esophagus.update(esophagus_blob)
            aorta_1.update(aorta_blob)
            aorta_2.update(aorta_blob)

    # All three blobs are segmented: assign based on location 
    elif len(blobs) == 3:
        top, middle, bottom = blobs
        output[:, :, z][top.blob] = AORTA
        output[:, :, z][middle.blob] = ESOPHAGUS
        output[:, :, z][bottom.blob] = AORTA

        esophagus.update(middle)
        aorta_1.update(top)
        aorta_2.update(bottom)

    else:
        print(">= 4 blobs not handled yet.")

# Added to process multiple patients
def process_patient(patient_id):
    # For patient_05
    global PATIENT
    PATIENT = patient_id
    BASE_PATH = os.path.expandvars("$HOME/ai4mi_project_group7/data/segthor_part1/train")
    INPUT_PATH = f"{BASE_PATH}/{PATIENT}/GT.nii.gz"
    OUTPUT_PATH = f"{BASE_PATH}/{PATIENT}/GT_split.nii.gz"

    print(f"Processing {PATIENT}")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Skipping {PATIENT}: File not found at {INPUT_PATH}")
        return

    img = nib.load(INPUT_PATH)
    data = img.get_fdata().astype(np.int16)
    print("Shape CT scan:", data.shape) # Differs per patient
    output = data.copy()
    heart_bbox = get_bounding_box_heart(data)

    aorta_1 = Organ("aorta_1") # Upper aorta (that comes above the heart)
    aorta_2 = Organ("aorta_2") # Lower aorta, closer to spine
    esophagus = Organ("esophagus")

    for z in range(data.shape[2]):
        phase = get_location_wrt_heart(z, heart_bbox)
        current_slice = data[:, :, z] == 1
        blobs = get_blobs_per_slice(current_slice)
        
        print(f"\nSlice {z}: " f"{len(blobs)} blobs" f"phase={phase}")

        for i, blob in enumerate(blobs):
            ys, xs = np.where(blob.blob)
            print(
                f"Blob {i}: "
                f"area={blob.area}, "
                f"center={blob.center}, "
                f"x={xs.min()}..{xs.max()}, "
                f"y={ys.min()}..{ys.max()}, "
                f"circularity={blob.circularity}"
            )
        print("------")

        # No aorta or esophaghus
        if len(blobs) == 0:
            continue

        if phase == "below_heart":
            below_heart_cases(blobs, output, z, aorta_1, aorta_2, esophagus)
        elif phase == "heart":
            during_heart_cases(blobs, output, z, aorta_1, aorta_2, esophagus)
        elif phase == "above_heart":
            above_heart_cases(blobs, output, z, aorta_1, aorta_2, esophagus)
        else:
            print("Other case happening..? Not implemented")


    output_img = nib.Nifti1Image(
        output, 
        img.affine, 
        img.header)
    nib.save(output_img, OUTPUT_PATH)
    print(f"Saved output to: {OUTPUT_PATH}")

if __name__ == "__main__":
    patient_list = [f"Patient_{i:02d}" for i in range(1, 21)]
    for patient in patient_list:
        process_patient(patient)
        
    print("\nAll patients processed!")


