import cv2
import numpy as np
import math
import os


SKIN_TONES = [

    "deep",
    "medium-deep",
    "medium",
    "medium-light",
    "light"

]


UNDERTONES = [

    "warm",
    "cool",
    "neutral"

]


MST_RGB = np.array(

    [

        [246, 237, 228],

        [243, 231, 219],

        [247, 234, 208],

        [234, 218, 186],

        [215, 189, 150],

        [160, 126, 86],

        [130, 92, 67],

        [96, 65, 52],

        [58, 49, 42],

        [41, 36, 32]

    ],

    dtype=np.uint8

)


MST_TO_SKIN_TONE = {

    1: "light",
    2: "light",
    3: "medium-light",
    4: "medium-light",
    5: "medium",
    6: "medium",
    7: "medium-deep",
    8: "medium-deep",
    9: "deep",
    10: "deep"

}


def _ciede2000(
    lab1,
    lab2
):

    L1, a1, b1 = map(
        float,
        lab1
    )

    L2, a2, b2 = map(
        float,
        lab2
    )


    C1 = math.sqrt(
        a1 * a1 +
        b1 * b1
    )


    C2 = math.sqrt(
        a2 * a2 +
        b2 * b2
    )


    C_bar = (
        C1 +
        C2
    ) / 2.0


    G = 0.5 * (

        1.0 -

        math.sqrt(

            (
                C_bar ** 7
            )
            /
            (
                C_bar ** 7 +
                25.0 ** 7
            )

        )

    )


    a1p = (
        1.0 +
        G
    ) * a1


    a2p = (
        1.0 +
        G
    ) * a2


    C1p = math.sqrt(
        a1p ** 2 +
        b1 ** 2
    )


    C2p = math.sqrt(
        a2p ** 2 +
        b2 ** 2
    )


    h1p = (
        math.degrees(
            math.atan2(
                b1,
                a1p
            )
        )
        % 360.0
    )


    h2p = (
        math.degrees(
            math.atan2(
                b2,
                a2p
            )
        )
        % 360.0
    )


    dLp = (
        L2 -
        L1
    )


    dCp = (
        C2p -
        C1p
    )


    dh = (
        h2p -
        h1p
    )


    if abs(dh) <= 180:

        dhp = dh

    elif dh > 180:

        dhp = dh - 360

    else:

        dhp = dh + 360


    dHp = (

        2.0 *

        math.sqrt(
            C1p *
            C2p
        ) *

        math.sin(
            math.radians(
                dhp / 2.0
            )
        )

    )


    Lbp = (
        L1 +
        L2
    ) / 2.0


    Cbp = (
        C1p +
        C2p
    ) / 2.0


    if (
        abs(
            h1p -
            h2p
        ) <= 180
    ):

        hbp = (
            h1p +
            h2p
        ) / 2.0

    elif (
        h1p +
        h2p
        < 360
    ):

        hbp = (
            h1p +
            h2p +
            360
        ) / 2.0

    else:

        hbp = (
            h1p +
            h2p -
            360
        ) / 2.0


    T = (

        1.0

        - 0.17 *
        math.cos(
            math.radians(
                hbp - 30
            )
        )

        + 0.24 *
        math.cos(
            math.radians(
                2 * hbp
            )
        )

        + 0.32 *
        math.cos(
            math.radians(
                3 * hbp + 6
            )
        )

        - 0.20 *
        math.cos(
            math.radians(
                4 * hbp - 63
            )
        )

    )


    dtheta = (

        30.0 *

        math.exp(

            -(
                (
                    hbp -
                    275
                )
                /
                25
            ) ** 2

        )

    )


    RC = (

        2.0 *

        math.sqrt(

            (
                Cbp ** 7
            )
            /
            (
                Cbp ** 7 +
                25.0 ** 7
            )

        )

    )


    SL = (

        1.0 +

        (
            0.015 *
            (Lbp - 50) ** 2
        )

        /

        math.sqrt(
            20 +
            (Lbp - 50) ** 2
        )

    )


    SC = (
        1.0 +
        0.045 *
        Cbp
    )


    SH = (

        1.0 +

        0.015 *
        Cbp *
        T

    )


    RT = (

        -math.sin(

            math.radians(
                2 * dtheta
            )

        )
        *
        RC

    )


    return math.sqrt(

        (
            dLp / SL
        ) ** 2

        +

        (
            dCp / SC
        ) ** 2

        +

        (
            dHp / SH
        ) ** 2

        +

        (
            RT *
            (dCp / SC) *
            (dHp / SH)

        )

    )


def _rgb_to_lab(rgb):

    pixel = np.uint8(
        [[
            [
                int(rgb[0]),
                int(rgb[1]),
                int(rgb[2])
            ]
        ]]
    )


    converted = cv2.cvtColor(

        pixel,

        cv2.COLOR_RGB2LAB

    )[0][0]


    return np.array(

        [

            (

                float(
                    converted[0]
                )
                /
                255.0
            )
            *
            100.0,

            float(
                converted[1]
            )
            -
            128.0,

            float(
                converted[2]
            )
            -
            128.0

        ]

    )


def analyze_skin(
    image_path,
    skin_mask_path,
    output_path=None
):

    image = cv2.imread(
        image_path
    )


    skin_mask = cv2.imread(

        skin_mask_path,

        cv2.IMREAD_GRAYSCALE

    )


    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )


    if skin_mask is None:

        raise FileNotFoundError(
            f"Could not load skin mask: {skin_mask_path}"
        )


    height, width = image.shape[:2]


    if skin_mask.shape != (
        height,
        width
    ):

        skin_mask = cv2.resize(

            skin_mask,

            (width, height),

            interpolation=cv2.INTER_NEAREST

        )


    skin_mask = np.where(

        skin_mask > 0,

        255,

        0

    ).astype(
        np.uint8
    )


    kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (3, 3)

    )


    skin_mask = cv2.morphologyEx(

        skin_mask,

        cv2.MORPH_OPEN,

        kernel

    )


    skin_mask = cv2.morphologyEx(

        skin_mask,

        cv2.MORPH_CLOSE,

        kernel

    )


    sampling_kernel = cv2.getStructuringElement(

        cv2.MORPH_ELLIPSE,

        (5, 5)

    )


    sampling_mask = cv2.erode(

        skin_mask,

        sampling_kernel,

        iterations=1

    )


    ys, xs = np.where(

        sampling_mask > 0

    )


    if len(xs) < 200:

        raise ValueError(
            "Not enough skin pixels."
        )


    pixels = image[
        ys,
        xs
    ]


    lab_cv = cv2.cvtColor(

        pixels.reshape(
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


    rgb = cv2.cvtColor(

        pixels.reshape(
            -1,
            1,
            3
        ),

        cv2.COLOR_BGR2RGB

    ).reshape(
        -1,
        3
    ).astype(
        np.float32
    )


    L = lab_cv[:, 0]


    low = np.percentile(
        L,
        5
    )


    high = np.percentile(
        L,
        95
    )


    stable_indices = (

        (L >= low)

        &

        (L <= high)

    )


    stable_lab_cv = lab_cv[
        stable_indices
    ]


    stable_rgb = rgb[
        stable_indices
    ]


    if len(
        stable_lab_cv
    ) < 200:

        stable_lab_cv = lab_cv

        stable_rgb = rgb


    lab_L = (

        stable_lab_cv[:, 0]
        /
        255.0
    ) * 100.0


    lab_a = (
        stable_lab_cv[:, 1]
        -
        128.0
    )


    lab_b = (
        stable_lab_cv[:, 2]
        -
        128.0
    )


    skin_lab = np.column_stack(

        [
            lab_L,
            lab_a,
            lab_b
        ]

    )


    representative = np.median(

        skin_lab,

        axis=0

    )


    rep_L = float(
        representative[0]
    )

    rep_a = float(
        representative[1]
    )

    rep_b = float(
        representative[2]
    )


    mst_lab = np.array(

        [
            _rgb_to_lab(rgb)

            for rgb in MST_RGB

        ]

    )


    distances = np.array(

        [

            _ciede2000(

                representative,

                reference

            )

            for reference in mst_lab

        ]

    )


    temperature = 5.0


    logits = (

        -distances /

        temperature

    )


    logits -= np.max(
        logits
    )


    probabilities = np.exp(
        logits
    )


    probabilities /= np.sum(
        probabilities
    )


    tone_probabilities = {

        "light": (
            probabilities[0] +
            probabilities[1]
        ),

        "medium-light": (
            probabilities[2] +
            probabilities[3]
        ),

        "medium": (
            probabilities[4] +
            probabilities[5]
        ),

        "medium-deep": (
            probabilities[6] +
            probabilities[7]
        ),

        "deep": (
            probabilities[8] +
            probabilities[9]
        )

    }


    skin_tone = max(

        tone_probabilities,

        key=tone_probabilities.get

    )


    tone_confidence = float(

        tone_probabilities[
            skin_tone
        ]

    )


    chroma = math.sqrt(

        rep_a ** 2 +

        rep_b ** 2

    )


    hue_angle = (

        math.degrees(

            math.atan2(

                rep_b,

                rep_a

            )

        )

        %

        360

    )


    warm_score = 0.0
    cool_score = 0.0
    neutral_score = 0.0


    if rep_b > 15:

        warm_score += 0.45


    if rep_b >= rep_a * 0.85:

        warm_score += 0.30


    if (
        rep_a > 20
        and
        rep_a > rep_b * 1.15
    ):

        cool_score += 0.50


    if rep_b < 12:

        cool_score += 0.25


    if chroma < 14:

        neutral_score += 0.60


    if (
        abs(
            rep_a -
            rep_b
        ) <= 8
        and
        chroma >= 14
    ):

        neutral_score += 0.35


    undertone_scores = {

        "warm": warm_score,

        "cool": cool_score,

        "neutral": neutral_score

    }


    undertone = max(

        undertone_scores,

        key=undertone_scores.get

    )


    sorted_scores = sorted(

        undertone_scores.values(),

        reverse=True

    )


    margin = (

        sorted_scores[0] -
        sorted_scores[1]

    )


    if margin >= 0.40:

        undertone_confidence = 1.0

    elif margin >= 0.20:

        undertone_confidence = 0.75

    elif margin >= 0.10:

        undertone_confidence = 0.55

    else:

        undertone_confidence = 0.35


    L_std = float(

        np.std(
            skin_lab[:, 0]
        )

    )


    a_std = float(

        np.std(
            skin_lab[:, 1]
        )

    )


    b_std = float(

        np.std(
            skin_lab[:, 2]
        )

    )


    consistency = max(

        0.0,

        min(

            1.0,

            1.0 -

            L_std /
            35.0

        )

    )


    tone_confidence = (

        tone_confidence *

        (
            0.70 +
            0.30 *
            consistency
        )

    )


    tone_confidence = max(

        0.0,

        min(

            1.0,

            tone_confidence

        )

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

            f"Tone: {skin_tone}",

            (30, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (0, 255, 0),

            2

        )


        cv2.putText(

            visualization,

            (
                f"Undertone: "
                f"{undertone}"
            ),

            (30, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 0),

            2

        )


        cv2.imwrite(

            output_path,

            visualization

        )


    return {

        "skin_tone": skin_tone,

        "skin_undertone": undertone,

        "confidence": {

            "skin_tone":
                float(
                    tone_confidence
                ),

            "skin_undertone":
                float(
                    undertone_confidence
                )

        },

        "measurements": {

            "lab_L":
                round(
                    rep_L,
                    2
                ),

            "lab_a":
                round(
                    rep_a,
                    2
                ),

            "lab_b":
                round(
                    rep_b,
                    2
                ),

            "chroma":
                round(
                    chroma,
                    2
                ),

            "hue_angle":
                round(
                    hue_angle,
                    2
                )

        }

    }