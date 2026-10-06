import os
import json
from datetime import datetime

from face_landmarks import analyze_face_shape
from face_parsing import parse_face
from hair_analysis import analyze_hair
from skin_analysis import analyze_skin


INPUT_DIR = "input"
OUTPUT_DIR = "output"


SUPPORTED_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"

}


def find_input_image():

    if not os.path.exists(INPUT_DIR):

        raise FileNotFoundError(
            f"Input folder not found: {INPUT_DIR}"
        )


    files = []

    for filename in os.listdir(
        INPUT_DIR
    ):

        path = os.path.join(
            INPUT_DIR,
            filename
        )


        if not os.path.isfile(
            path
        ):

            continue


        extension = (
            os.path.splitext(
                filename
            )[1]
            .lower()
        )


        if extension in SUPPORTED_EXTENSIONS:

            files.append(
                path
            )


    if not files:

        raise FileNotFoundError(

            "No supported image found "
            "inside the input folder."

        )


    preferred = os.path.join(

        INPUT_DIR,

        "test.jpg"

    )


    if os.path.exists(
        preferred
    ):

        return preferred


    files.sort(

        key=os.path.getmtime,

        reverse=True

    )


    return files[0]


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    image_path = find_input_image()


    print()
    print(
        "================================"
    )

    print(
        "      FASHION CV PIPELINE"
    )

    print(
        "================================"
    )

    print()

    print(
        "Input:",
        image_path
    )


    # ========================================================
    # 1. Face shape
    # ========================================================

    print()
    print(
        "[1/4] Face shape..."
    )


    face_shape_result = analyze_face_shape(

        image_path,

        os.path.join(
            OUTPUT_DIR,
            "face_shape.jpg"
        )

    )


    print(
        "Face shape:",
        face_shape_result[
            "face_shape"
        ]
    )


    # ========================================================
    # 2. Face parsing
    # ========================================================

    print()
    print(
        "[2/4] Face parsing..."
    )


    parsing_result = parse_face(

        image_path,

        OUTPUT_DIR

    )


    print(
        "Skin mask:",
        parsing_result[
            "skin_mask"
        ]
    )


    print(
        "Hair mask:",
        parsing_result[
            "hair_mask"
        ]
    )


    # ========================================================
    # 3. Hair analysis
    # ========================================================

    print()
    print(
        "[3/4] Hair analysis..."
    )


    hair_result = analyze_hair(

        image_path,

        parsing_result[
            "hair_mask"
        ],

        os.path.join(
            OUTPUT_DIR,
            "hair_analysis.jpg"
        )

    )


    print(
        "Hair color:",
        hair_result[
            "hair_color"
        ]
    )


    # ========================================================
    # 4. Skin analysis
    # ========================================================

    print()
    print(
        "[4/4] Skin analysis..."
    )


    skin_result = analyze_skin(

        image_path,

        parsing_result[
            "skin_mask"
        ],

        os.path.join(
            OUTPUT_DIR,
            "skin_analysis.jpg"
        )

    )


    print(
        "Skin tone:",
        skin_result[
            "skin_tone"
        ]
    )


    print(
        "Undertone:",
        skin_result[
            "skin_undertone"
        ]
    )


    # ========================================================
    # Final profile
    # ========================================================

    appearance_profile = {

        "face_shape":
            face_shape_result[
                "face_shape"
            ],

        "skin_tone":
            skin_result[
                "skin_tone"
            ],

        "skin_undertone":
            skin_result[
                "skin_undertone"
            ],

        "hair_color":
            hair_result[
                "hair_color"
            ],

        "confidence": {

            "face_shape":
                float(
                    face_shape_result[
                        "confidence"
                    ]
                ),

            "skin_tone":
                float(
                    skin_result[
                        "confidence"
                    ][
                        "skin_tone"
                    ]
                ),

            "skin_undertone":
                float(
                    skin_result[
                        "confidence"
                    ][
                        "skin_undertone"
                    ]
                ),

            "hair_color":
                float(
                    hair_result[
                        "confidence"
                    ]
                )

        }

    }


    # ========================================================
    # Save final JSON
    # ========================================================

    profile_path = os.path.join(

        OUTPUT_DIR,

        "appearance_profile.json"

    )


    with open(

        profile_path,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            appearance_profile,

            file,

            indent=4

        )


    # ========================================================
    # Final output
    # ========================================================

    print()
    print(
        "================================"
    )

    print(
        "       FINAL APPEARANCE"
    )

    print(
        "================================"
    )


    print()

    print(

        json.dumps(

            appearance_profile,

            indent=4

        )

    )


    print()

    print(
        "Saved:",
        profile_path
    )


if __name__ == "__main__":

    main()