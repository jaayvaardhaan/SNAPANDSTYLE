import cv2
import numpy as np
import os
import math


HAIR_COLORS = [

    "black",
    "dark_brown",
    "brown",
    "light_brown",
    "blonde",
    "red_or_auburn",
    "gray_or_white"

]


def _softmax(values):

    values = np.array(
        values,
        dtype=np.float64
    )

    values -= np.max(
        values
    )

    exp_values = np.exp(
        values
    )

    return (
        exp_values /
        np.sum(
            exp_values
        )
    )


def analyze_hair(
    image_path,
    hair_mask_path,
    output_path=None
):

    image = cv2.imread(
        image_path
    )


    hair_mask = cv2.imread(

        hair_mask_path,

        cv2.IMREAD_GRAYSCALE

    )


    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )


    if hair_mask is None:

        raise FileNotFoundError(
            f"Could not load hair mask: {hair_mask_path}"
        )


    height, width = image.shape[:2]


    if hair_mask.shape != (
        height,
        width
    ):

        hair_mask = cv2.resize(

            hair_mask,

            (width, height),

            interpolation=cv2.INTER_NEAREST

        )


    hair_mask = np.where(

        hair_mask > 0,

        255,

        0

    ).astype(
        np.uint8
    )


    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (3, 3)

    )


    hair_mask = cv2.morphologyEx(

        hair_mask,

        cv2.MORPH_OPEN,

        kernel

    )


    hair_mask = cv2.morphologyEx(

        hair_mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    ys, xs = np.where(

        hair_mask > 0

    )


    if len(xs) < 150:

        raise ValueError(
            "Not enough hair pixels."
        )


    hair_pixels = image[
        ys,
        xs
    ]


    hair_lab = cv2.cvtColor(

        hair_pixels.reshape(
            -1,
            1,
            3
        ),

        cv2.COLOR_BGR2LAB

    ).reshape(

        -1,
        3

    ).astype(
        np.float32
    )


    hair_hsv = cv2.cvtColor(

        hair_pixels.reshape(
            -1,
            1,
            3
        ),

        cv2.COLOR_BGR2HSV

    ).reshape(

        -1,
        3

    ).astype(
        np.float32
    )


    L_values = hair_lab[:, 0]


    low = np.percentile(
        L_values,
        5
    )


    high = np.percentile(
        L_values,
        95
    )


    stable_indices = (

        (L_values >= low)

        &

        (L_values <= high)

    )


    stable_lab = hair_lab[
        stable_indices
    ]


    stable_pixels = hair_pixels[
        stable_indices
    ]


    if len(stable_lab) < 150:

        stable_lab = hair_lab

        stable_pixels = hair_pixels


    median_lab = np.median(

        stable_lab,

        axis=0

    )


    L = float(
        median_lab[0]
    )


    a = float(
        median_lab[1]
    )


    b = float(
        median_lab[2]
    )


    lightness = (

        L /
        255.0

    ) * 100.0


    a_centered = (
        a -
        128.0
    )


    b_centered = (
        b -
        128.0
    )


    chroma = math.sqrt(

        a_centered ** 2

        +

        b_centered ** 2

    )


    median_hsv = np.median(

        hair_hsv[
            stable_indices
        ],

        axis=0

    )


    if len(
        median_hsv
    ) == 0:

        median_hsv = np.median(
            hair_hsv,
            axis=0
        )


    hue = float(
        median_hsv[0]
    )


    saturation = float(
        median_hsv[1]
    )


    value = float(
        median_hsv[2]
    )


    scores = {

        "black": 0.0,

        "dark_brown": 0.0,

        "brown": 0.0,

        "light_brown": 0.0,

        "blonde": 0.0,

        "red_or_auburn": 0.0,

        "gray_or_white": 0.0

    }


    if (
        lightness < 24
        and
        chroma < 9
    ):

        scores["black"] += 3.0


    if (
        lightness < 40
        and
        chroma >= 8
        and
        b_centered >= 5
    ):

        scores["dark_brown"] += 3.0


    if (
        35 <= lightness < 58
        and
        chroma >= 8
        and
        b_centered >= 5
    ):

        scores["brown"] += 3.0


    if (
        50 <= lightness < 72
        and
        chroma >= 7
        and
        b_centered >= 7
    ):

        scores["light_brown"] += 3.0


    if (
        lightness >= 65
        and
        chroma >= 8
        and
        b_centered >= 5
    ):

        scores["blonde"] += 3.0


    if (
        a_centered > 10
        and
        b_centered > 5
        and
        a_centered > b_centered
        and
        chroma >= 10
    ):

        scores["red_or_auburn"] += 3.0


    if (
        lightness >= 55
        and
        chroma < 10
        and
        saturation < 50
    ):

        scores["gray_or_white"] += 3.0


    scores["black"] += max(

        0.0,

        2.0 -
        abs(
            lightness -
            18.0
        ) / 10.0

    )


    scores["dark_brown"] += max(

        0.0,

        2.0 -
        abs(
            lightness -
            30.0
        ) / 15.0

    )


    scores["brown"] += max(

        0.0,

        2.0 -
        abs(
            lightness -
            46.0
        ) / 15.0

    )


    scores["light_brown"] += max(

        0.0,

        2.0 -
        abs(
            lightness -
            62.0
        ) / 15.0

    )


    scores["blonde"] += max(

        0.0,

        2.0 -
        abs(
            lightness -
            78.0
        ) / 18.0

    )


    hue_strength = max(

        0.0,

        min(
            1.0,
            chroma / 25.0
        )

    )


    if (
        a_centered >
        b_centered
    ):

        scores["red_or_auburn"] += (

            hue_strength *
            1.5

        )


    scores_array = [

        scores[name]

        for name in HAIR_COLORS

    ]


    probabilities = _softmax(
        scores_array
    )


    best_index = int(
        np.argmax(
            probabilities
        )
    )


    hair_color = HAIR_COLORS[
        best_index
    ]


    confidence = float(
        probabilities[
            best_index
        ]
    )


    if output_path is not None:

        visualization = image.copy()


        for x, y in zip(
            xs,
            ys
        ):

            visualization[
                y,
                x
            ] = (

                0,
                255,
                0

            )


        cv2.putText(

            visualization,

            f"Hair: {hair_color}",

            (30, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (0, 255, 0),

            2

        )


        cv2.putText(

            visualization,

            (
                f"Confidence: "
                f"{confidence:.2f}"
            ),

            (30, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2

        )


        cv2.imwrite(

            output_path,

            visualization

        )


    return {

        "hair_color": hair_color,

        "confidence": confidence,

        "measurements": {

            "lab_lightness": round(
                lightness,
                2
            ),

            "lab_a": round(
                a_centered,
                2
            ),

            "lab_b": round(
                b_centered,
                2
            ),

            "chroma": round(
                chroma,
                2
            ),

            "hsv_hue": round(
                hue,
                2
            ),

            "hsv_saturation": round(
                saturation,
                2
            ),

            "hsv_value": round(
                value,
                2
            )

        }

    }