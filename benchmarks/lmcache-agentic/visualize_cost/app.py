import argparse
import csv
from pathlib import Path
from flask import Flask, jsonify, render_template, request
from prefix_cache_cost import compute_prefix_cache_curve
from compaction_cost import compute_compaction_curve


def create_app(csv_path: str) -> Flask:
    app = Flask(__name__)
    app.config["CSV_PATH"] = str(Path(csv_path).resolve())

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/session_ids")
    def session_ids():
        csv_path = app.config["CSV_PATH"]
        values = []
        seen = set()
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row["session_id"]
                if session_id not in seen:
                    seen.add(session_id)
                    values.append(session_id)
        values.sort()
        return jsonify({"session_ids": values})

    @app.get("/api/prefix_cache")
    def prefix_cache():
        session_id = request.args.get("session_id", "").strip()
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400
        try:
            result = compute_prefix_cache_curve(app.config["CSV_PATH"], session_id)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/compaction")
    def compaction():
        session_id = request.args.get("session_id", "").strip()
        ratio_text = request.args.get("ratio", "").strip()
        if not session_id:
            return jsonify({"error": "Missing session_id"}), 400
        if ratio_text == "":
            return jsonify({"error": "Missing ratio"}), 400
        try:
            ratio = float(ratio_text)
            if ratio > 1.0:
                ratio /= 100.0
            result = compute_compaction_curve(app.config["CSV_PATH"], session_id, ratio)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive cost visualizer")
    parser.add_argument("--csv", required=True, help="Path to the cost CSV file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.csv)
    app.run(host=args.host, port=args.port, debug=True)
