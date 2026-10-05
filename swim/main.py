"""Entry point: python -m swim.main input.mp4 -o output.mp4"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a swimming video.")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("-o", "--output", default="data/output/out.mp4", help="Path to output video")
    args = parser.parse_args()

    # TODO: read frames -> detect pose -> compute metrics -> draw -> write video
    print(f"Processing {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
