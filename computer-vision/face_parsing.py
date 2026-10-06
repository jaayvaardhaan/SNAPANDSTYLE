import cv2
import numpy as np
import onnxruntime as ort
import os


MODEL_PATH = "models/resnet18.onnx"


LABELS = {

    0: "background",
    1: "skin",
    2: "left_eyebrow",
    3: "right_eyebrow",
    4: "left_eye",
    5: "right_eye",
    6: "eyeglass",
    7: "left_ear",
    8: "right_ear",
    9: "earring",
    10: "nose",
    11: "mouth",
    12: "upper_lip",
    13: "lower_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat"

}


SKIN_CLASS = 1
HAIR_CLASS = 17
CLOTH_CLASS = 16


INPUT_SIZE = 512


def parse_face(
    image_path,
    output_dir="output"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )


    image = cv2.imread(
        image_path
    )


    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )


    original_height, original_width = (
        image.shape[:2]
    )


    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    resized = cv2.resize(
        rgb,
        (
            INPUT_SIZE,
            INPUT_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )


    resized = (
        resized.astype(
            np.float32
        )
        / 255.0
    )


    mean = np.array(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=np.float32
    )


    std = np.array(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=np.float32
    )


    normalized = (
        resized -
        mean
    ) / std


    tensor = np.transpose(
        normalized,
        (2, 0, 1)
    )


    tensor = np.expand_dims(
        tensor,
        axis=0
    ).astype(
        np.float32
    )


    session = ort.InferenceSession(

        MODEL_PATH,

        providers=[
            "CPUExecutionProvider"
        ]

    )


    input_name = (
        session
        .get_inputs()[0]
        .name
    )


    output_names = [

        output.name

        for output in
        session.get_outputs()

    ]


    outputs = session.run(

        output_names,

        {
            input_name:
                tensor
        }

    )


    prediction = outputs[0]


    prediction = prediction.squeeze(
        0
    )


    class_mask = np.argmax(

        prediction,

        axis=0

    ).astype(
        np.uint8
    )


    restored_mask = cv2.resize(

        class_mask,

        (
            original_width,
            original_height
        ),

        interpolation=cv2.INTER_NEAREST

    )


    raw_label_path = os.path.join(

        output_dir,

        "face_parsing_labels.png"

    )


    skin_mask_path = os.path.join(

        output_dir,

        "skin_mask.png"

    )


    hair_mask_path = os.path.join(

        output_dir,

        "hair_mask.png"

    )


    cloth_mask_path = os.path.join(

        output_dir,

        "cloth_mask.png"

    )


    visualization_path = os.path.join(

        output_dir,

        "face_parsing_mask.png"

    )


    cv2.imwrite(

        raw_label_path,

        restored_mask

    )


    skin_mask = np.where(

        restored_mask == SKIN_CLASS,

        255,

        0

    ).astype(
        np.uint8
    )


    hair_mask = np.where(

        restored_mask == HAIR_CLASS,

        255,

        0

    ).astype(
        np.uint8
    )


    cloth_mask = np.where(

        restored_mask == CLOTH_CLASS,

        255,

        0

    ).astype(
        np.uint8
    )


    cv2.imwrite(

        skin_mask_path,

        skin_mask

    )


    cv2.imwrite(

        hair_mask_path,

        hair_mask

    )


    cv2.imwrite(

        cloth_mask_path,

        cloth_mask

    )


    colors = {

        0: (0, 0, 0),

        1: (120, 180, 255),

        2: (100, 255, 100),

        3: (100, 220, 100),

        4: (0, 255, 0),

        5: (0, 200, 0),

        6: (255, 0, 255),

        7: (255, 180, 80),

        8: (255, 150, 60),

        9: (255, 200, 100),

        10: (180, 120, 80),

        11: (80, 80, 255),

        12: (100, 100, 255),

        13: (120, 120, 255),

        14: (80, 180, 180),

        15: (100, 200, 200),

        16: (180, 180, 180),

        17: (80, 50, 200),

        18: (255, 255, 0)

    }


    color_mask = np.zeros_like(
        image
    )


    for class_id, color in colors.items():

        color_mask[
            restored_mask == class_id
        ] = color


    overlay = cv2.addWeighted(

        image,

        0.60,

        color_mask,

        0.40,

        0

    )


    cv2.imwrite(

        visualization_path,

        overlay

    )


    return {

        "skin_mask": skin_mask_path,

        "hair_mask": hair_mask_path,

        "cloth_mask": cloth_mask_path,

        "label_mask": raw_label_path,

        "statistics": {

            "skin_pixels": int(
                np.count_nonzero(
                    skin_mask
                )
            ),

            "hair_pixels": int(
                np.count_nonzero(
                    hair_mask
                )
            ),

            "cloth_pixels": int(
                np.count_nonzero(
                    cloth_mask
                )
            )

        }

    }