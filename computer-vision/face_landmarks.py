import cv2
import mediapipe as mp
import numpy as np


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections


MODEL_PATH = "models/face_landmarker.task"


def _interpolate_boundary(points, target_y):

    if len(points) < 2:
        return None

    for i in range(len(points) - 1):

        y1, x1 = points[i]
        y2, x2 = points[i + 1]

        if y1 <= target_y <= y2:

            if y2 == y1:
                return x1

            ratio = (
                target_y - y1
            ) / (
                y2 - y1
            )

            return x1 + (
                x2 - x1
            ) * ratio

    return None


def _softmax(values):

    values = np.array(
        values,
        dtype=np.float64
    )

    values -= np.max(values)

    exp_values = np.exp(values)

    return exp_values / np.sum(exp_values)


def analyze_face_shape(
    image_path,
    output_path=None
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )


    height, width = image.shape[:2]


    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=VisionRunningMode.IMAGE
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )


    with FaceLandmarker.create_from_options(
        options
    ) as landmarker:

        result = landmarker.detect(
            mp_image
        )


    if not result.face_landmarks:

        raise ValueError(
            "No face detected."
        )


    face = result.face_landmarks[0]


    oval_indices = set()


    for connection in (
        FaceLandmarksConnections
        .FACE_LANDMARKS_FACE_OVAL
    ):

        oval_indices.add(
            connection.start
        )

        oval_indices.add(
            connection.end
        )


    oval_points = [

        (
            face[index].x,
            face[index].y,
            index
        )

        for index in oval_indices

    ]


    leftmost = min(
        oval_points,
        key=lambda p: p[0]
    )

    rightmost = max(
        oval_points,
        key=lambda p: p[0]
    )

    topmost = min(
        oval_points,
        key=lambda p: p[1]
    )

    bottommost = max(
        oval_points,
        key=lambda p: p[1]
    )


    face_left = leftmost[0]
    face_right = rightmost[0]

    face_top = topmost[1]
    face_bottom = bottommost[1]


    face_width = (
        face_right -
        face_left
    )

    face_height = (
        face_bottom -
        face_top
    )


    if face_width <= 0:

        raise ValueError(
            "Invalid face width."
        )


    center_x = (
        face_left +
        face_right
    ) / 2.0


    left_boundary = []
    right_boundary = []


    for x, y, _ in oval_points:

        if x < center_x:

            left_boundary.append(
                (y, x)
            )

        else:

            right_boundary.append(
                (y, x)
            )


    left_boundary.sort()
    right_boundary.sort()


    levels = {

        "upper": 0.25,
        "upper_middle": 0.35,
        "middle": 0.50,
        "lower_middle": 0.65,
        "lower": 0.75

    }


    measurements = {}


    for name, ratio in levels.items():

        target_y = (

            face_top +

            ratio *
            face_height

        )


        left_x = _interpolate_boundary(
            left_boundary,
            target_y
        )

        right_x = _interpolate_boundary(
            right_boundary,
            target_y
        )


        if (
            left_x is None
            or
            right_x is None
        ):

            measurements[name] = None

        else:

            measurements[name] = (
                right_x -
                left_x
            )


    valid_widths = [

        value

        for value in measurements.values()

        if value is not None

    ]


    if not valid_widths:

        raise ValueError(
            "Could not calculate face width profile."
        )


    maximum_width = max(
        valid_widths
    )


    normalized = {

        name: (

            value /
            maximum_width

        )

        if value is not None
        else None

        for name, value in measurements.items()

    }


    length_width_ratio = (
        face_height /
        maximum_width
    )


    upper = normalized[
        "upper"
    ]

    upper_middle = normalized[
        "upper_middle"
    ]

    middle = normalized[
        "middle"
    ]

    lower_middle = normalized[
        "lower_middle"
    ]

    lower = normalized[
        "lower"
    ]


    jaw_taper = 0.0


    if (
        middle is not None
        and
        lower is not None
    ):

        jaw_taper = (
            middle -
            lower
        )


    scores = {

        "round": 0.0,

        "oval": 0.0,

        "square": 0.0,

        "oblong": 0.0,

        "heart": 0.0,

        "diamond": 0.0

    }


    if upper is not None:

        scores["round"] += (
            max(
                0.0,
                1.0 -
                abs(
                    length_width_ratio -
                    1.0
                ) / 0.35
            )
        )


        scores["oval"] += (
            max(
                0.0,
                1.0 -
                abs(
                    length_width_ratio -
                    1.25
                ) / 0.45
            )
        )


        scores["oblong"] += (
            max(
                0.0,
                (
                    length_width_ratio -
                    1.20
                ) / 0.50
            )
        )


    if lower is not None:

        if lower >= 0.90:

            scores["square"] += 1.0


        if lower < 0.90:

            scores["round"] += 0.45


        if lower < 0.82:

            scores["heart"] += 0.65


    if (
        upper is not None
        and
        lower is not None
    ):

        if (
            upper >= 0.92
            and
            lower < 0.82
        ):

            scores["heart"] += 1.0


        if (
            upper >= 0.90
            and
            lower >= 0.88
            and
            jaw_taper < 0.08
        ):

            scores["square"] += 1.0


    if (
        middle is not None
        and
        upper is not None
        and
        lower is not None
    ):

        if (
            middle >
            upper + 0.04
            and
            middle >
            lower + 0.04
        ):

            scores["diamond"] += 1.5


    if (
        length_width_ratio >= 1.35
    ):

        scores["oblong"] += 1.0


    if (
        1.05 <= length_width_ratio < 1.35
        and
        lower is not None
        and
        lower < 0.88
    ):

        scores["oval"] += 0.7


    if (
        length_width_ratio <= 1.08
        and
        lower is not None
        and
        lower < 0.90
    ):

        scores["round"] += 0.8


    score_values = list(
        scores.values()
    )


    probabilities = _softmax(
        score_values
    )


    class_names = list(
        scores.keys()
    )


    best_index = int(
        np.argmax(
            probabilities
        )
    )


    face_shape = class_names[
        best_index
    ]


    confidence = float(
        probabilities[
            best_index
        ]
    )


    if output_path is not None:

        visualization = image.copy()


        for connection in (
            FaceLandmarksConnections
            .FACE_LANDMARKS_FACE_OVAL
        ):

            start = face[
                connection.start
            ]

            end = face[
                connection.end
            ]


            start_point = (
                int(
                    start.x *
                    width
                ),
                int(
                    start.y *
                    height
                )
            )


            end_point = (
                int(
                    end.x *
                    width
                ),
                int(
                    end.y *
                    height
                )
            )


            cv2.line(

                visualization,

                start_point,

                end_point,

                (0, 0, 255),

                2

            )


        for name, ratio in levels.items():

            value = measurements[name]


            if value is None:
                continue


            y = int(

                (

                    face_top +

                    ratio *
                    face_height

                )

                *

                height

            )


            left_x = _interpolate_boundary(
                left_boundary,
                face_top +
                ratio *
                face_height
            )

            right_x = _interpolate_boundary(
                right_boundary,
                face_top +
                ratio *
                face_height
            )


            if (
                left_x is None
                or
                right_x is None
            ):
                continue


            cv2.line(

                visualization,

                (
                    int(
                        left_x *
                        width
                    ),
                    y
                ),

                (
                    int(
                        right_x *
                        width
                    ),
                    y
                ),

                (255, 0, 0),

                1

            )


        cv2.putText(

            visualization,

            (
                f"Face: "
                f"{face_shape}"
            ),

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

        "face_shape": face_shape,

        "confidence": confidence,

        "features": {

            "length_width_ratio":
                float(
                    length_width_ratio
                ),

            "upper_width_ratio":
                float(
                    upper
                )
                if upper is not None
                else None,

            "upper_middle_width_ratio":
                float(
                    upper_middle
                )
                if upper_middle is not None
                else None,

            "middle_width_ratio":
                float(
                    middle
                )
                if middle is not None
                else None,

            "lower_middle_width_ratio":
                float(
                    lower_middle
                )
                if lower_middle is not None
                else None,

            "lower_width_ratio":
                float(
                    lower
                )
                if lower is not None
                else None,

            "jaw_taper":
                float(
                    jaw_taper
                )

        }

    }