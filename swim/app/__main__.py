"""Start the local web app: python -m swim.app, then open http://127.0.0.1:8000 (see docs/app.md).

Run it from the repo root, like python -m swim.run: saved results record the video path relative to it.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m swim.app",
                                     description="Process videos and watch the results in the browser.")
    parser.add_argument("--input", default="data/input", help="Folder the Videos page opens first (default: data/input)")
    parser.add_argument("--output", default="data/output", help="Results root folder, as for swim.run (default: data/output)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    import uvicorn

    from swim.app.server import create_app

    print(f"swim-analytics: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    # Local only: listen on 127.0.0.1, not on the network
    uvicorn.run(create_app(args.output, args.input), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
