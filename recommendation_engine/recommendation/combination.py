import pandas as pd

from filtering import load_dataset, filter_items
from candidate_scoring import select_all_candidates


MAX_CANDIDATES = 10


def generate_normal_outfits(
    topwear,
    bottomwear,
    footwear
):

    combinations = []

    for _, top in topwear.iterrows():

        for _, bottom in bottomwear.iterrows():

            for _, shoe in footwear.iterrows():

                combinations.append({
                    "topwear": top.to_dict(),
                    "bottomwear": bottom.to_dict(),
                    "footwear": shoe.to_dict()
                })

    return combinations


def generate_one_piece_outfits(
    one_piece,
    footwear
):

    combinations = []

    for _, dress in one_piece.iterrows():

        for _, shoe in footwear.iterrows():

            combinations.append({
                "one_piece": dress.to_dict(),
                "footwear": shoe.to_dict()
            })

    return combinations


def generate_outfits(candidates):

    outfits = []

    if (
        "topwear" in candidates
        and "bottomwear" in candidates
        and "footwear" in candidates
    ):

        outfits.extend(
            generate_normal_outfits(
                candidates["topwear"],
                candidates["bottomwear"],
                candidates["footwear"]
            )
        )

    if (
        "one_piece" in candidates
        and "footwear" in candidates
    ):

        outfits.extend(
            generate_one_piece_outfits(
                candidates["one_piece"],
                candidates["footwear"]
            )
        )

    return outfits


def display_outfit(outfit):

    print("-" * 60)

    if "one_piece" in outfit:

        item = outfit["one_piece"]

        print(
            f"One-piece: "
            f"{item['articleType']} "
            f"({item['baseColour']}) "
            f"[ID: {item['id']}]"
        )

        item = outfit["footwear"]

        print(
            f"Footwear: "
            f"{item['articleType']} "
            f"({item['baseColour']}) "
            f"[ID: {item['id']}]"
        )

    else:

        item = outfit["topwear"]

        print(
            f"Topwear: "
            f"{item['articleType']} "
            f"({item['baseColour']}) "
            f"[ID: {item['id']}]"
        )

        item = outfit["bottomwear"]

        print(
            f"Bottomwear: "
            f"{item['articleType']} "
            f"({item['baseColour']}) "
            f"[ID: {item['id']}]"
        )

        item = outfit["footwear"]

        print(
            f"Footwear: "
            f"{item['articleType']} "
            f"({item['baseColour']}) "
            f"[ID: {item['id']}]"
        )


if __name__ == "__main__":

    print("Loading SnapStyle dataset...")

    df = load_dataset()

    gender = "Men"
    occasion = "interview"
    season = "Winter"

    print()
    print("User requirements:")
    print(f"Gender: {gender}")
    print(f"Occasion: {occasion}")
    print(f"Season: {season}")

    # ----------------------------------------------
    # Hard filtering
    # ----------------------------------------------

    candidates = {

        "topwear": filter_items(
            df,
            gender=gender,
            occasion=occasion,
            category="topwear"
        ),

        "bottomwear": filter_items(
            df,
            gender=gender,
            occasion=occasion,
            category="bottomwear"
        ),

        "one_piece": filter_items(
            df,
            gender=gender,
            occasion=occasion,
            category="one_piece"
        ),

        "footwear": filter_items(
            df,
            gender=gender,
            occasion=occasion,
            category="footwear"
        )
    }

    print()
    print("After hard filtering:")

    for category, items in candidates.items():

        print(
            f"{category}: {len(items)}"
        )

    # ----------------------------------------------
    # Preliminary candidate scoring
    # ----------------------------------------------

    candidates = select_all_candidates(
        candidates,
        requested_occasion=occasion,
        requested_season=season,
        limit=MAX_CANDIDATES
    )

    print()
    print("After candidate scoring:")

    for category, items in candidates.items():

        print(
            f"{category}: {len(items)}"
        )

    # ----------------------------------------------
    # Generate combinations
    # ----------------------------------------------

    outfits = generate_outfits(
        candidates
    )

    print()
    print(
        f"Total combinations generated: "
        f"{len(outfits)}"
    )

    # ----------------------------------------------
    # Display sample
    # ----------------------------------------------

    print()
    print("Sample outfits:")

    for outfit in outfits[:10]:

        display_outfit(
            outfit
        )