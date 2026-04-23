import argparse
import csv
from pathlib import Path
from flask import Flask, jsonify, render_template, request

from compaction_cost import compute_compaction_curve
from kv_compaction_growth import compute_kv_compaction_curve
from kv_prefix_growth import compute_kv_prefix_curve
from model_catalog import DEFAULT_MODEL, get_model_info, model_options
from gpu_catalog import DEFAULT_GPU, get_gpu_info, gpu_options
from overall_cost import DEFAULT_GAP_SECONDS, compute_overall_compaction_curve, compute_overall_prefix_curve
from cost_quality import (
    compute_cost_quality_compaction_curve,
    compute_cost_quality_prefix_curve,
    compute_avg_quality_compaction_point,
    compute_avg_quality_prefix_point,
)
from prefix_cache_cost import compute_prefix_cache_curve


def create_app(csv_path: str) -> Flask:
    app = Flask(__name__)
    app.config['CSV_PATH'] = str(Path(csv_path).resolve())
    app.config['DEFAULT_MODEL'] = DEFAULT_MODEL
    app.config['DEFAULT_GPU'] = DEFAULT_GPU
    app.config['DEFAULT_GAP_SECONDS'] = DEFAULT_GAP_SECONDS

    @app.get('/')
    def index():
        return render_template('index.html', default_model=app.config['DEFAULT_MODEL'], default_gap_seconds=app.config['DEFAULT_GAP_SECONDS'])



    def parse_gap_seconds() -> float:
        raw = request.args.get('gap_seconds', str(app.config['DEFAULT_GAP_SECONDS'])).strip()
        try:
            gap_seconds = float(raw)
        except ValueError as exc:
            raise ValueError('gap_seconds must be a number greater than 0') from exc
        if gap_seconds <= 0:
            raise ValueError('gap_seconds must be greater than 0')
        return gap_seconds

    @app.get('/api/session_ids')
    def session_ids():
        csv_path = app.config['CSV_PATH']
        values = []
        seen = set()
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                session_id = row['session_id']
                if session_id not in seen:
                    seen.add(session_id)
                    values.append(session_id)
        values.sort()
        return jsonify({'session_ids': values})

    @app.get('/api/models')
    def models():
        return jsonify({'default_model': app.config['DEFAULT_MODEL'], 'models': model_options()})


    @app.get('/api/gpus')
    def gpus():
        return jsonify({'default_gpu': app.config['DEFAULT_GPU'], 'gpus': gpu_options()})

    @app.get('/api/gpu_info')
    def gpu_info():
        gpu_name = request.args.get('gpu', app.config['DEFAULT_GPU']).strip() or app.config['DEFAULT_GPU']
        try:
            info = get_gpu_info(gpu_name)
            return jsonify(info)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/model_info')
    def model_info():
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        try:
            info = get_model_info(model_name)
            return jsonify(info)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/prefix_cache')
    def prefix_cache():
        session_id = request.args.get('session_id', '').strip()
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        try:
            result = compute_prefix_cache_curve(app.config['CSV_PATH'], session_id)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/compaction')
    def compaction():
        session_id = request.args.get('session_id', '').strip()
        ratio_text = request.args.get('ratio', '').strip()
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        if ratio_text == '':
            return jsonify({'error': 'Missing ratio'}), 400
        try:
            ratio = float(ratio_text)
            if ratio > 1.0:
                ratio /= 100.0
            result = compute_compaction_curve(app.config['CSV_PATH'], session_id, ratio)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/kv_prefix')
    def kv_prefix():
        session_id = request.args.get('session_id', '').strip()
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        try:
            result = compute_kv_prefix_curve(app.config['CSV_PATH'], session_id, model_name)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400


    @app.get('/api/overall_prefix')
    def overall_prefix():
        session_id = request.args.get('session_id', '').strip()
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        gpu_name = request.args.get('gpu', app.config['DEFAULT_GPU']).strip() or app.config['DEFAULT_GPU']
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        try:
            gap_seconds = parse_gap_seconds()
            result = compute_overall_prefix_curve(app.config['CSV_PATH'], session_id, model_name, gpu_name, gap_seconds=gap_seconds)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/overall_compaction')
    def overall_compaction():
        session_id = request.args.get('session_id', '').strip()
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        gpu_name = request.args.get('gpu', app.config['DEFAULT_GPU']).strip() or app.config['DEFAULT_GPU']
        ratio_text = request.args.get('ratio', '').strip()
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        if ratio_text == '':
            return jsonify({'error': 'Missing ratio'}), 400
        try:
            ratio = float(ratio_text)
            if ratio > 1.0:
                ratio /= 100.0
            gap_seconds = parse_gap_seconds()
            result = compute_overall_compaction_curve(app.config['CSV_PATH'], session_id, ratio, model_name, gpu_name, gap_seconds=gap_seconds)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400


    @app.get('/api/cost_quality_prefix')
    def cost_quality_prefix():
        session_id = request.args.get('session_id', '').strip()
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        gpu_name = request.args.get('gpu', app.config['DEFAULT_GPU']).strip() or app.config['DEFAULT_GPU']
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        try:
            gap_seconds = parse_gap_seconds()
            result = compute_cost_quality_prefix_curve(app.config['CSV_PATH'], session_id, model_name, gpu_name, gap_seconds=gap_seconds)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/cost_quality_compaction')
    def cost_quality_compaction():
        session_id = request.args.get('session_id', '').strip()
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        gpu_name = request.args.get('gpu', app.config['DEFAULT_GPU']).strip() or app.config['DEFAULT_GPU']
        ratio_text = request.args.get('ratio', '').strip()
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        if ratio_text == '':
            return jsonify({'error': 'Missing ratio'}), 400
        try:
            ratio = float(ratio_text)
            if ratio > 1.0:
                ratio /= 100.0
            gap_seconds = parse_gap_seconds()
            result = compute_cost_quality_compaction_curve(app.config['CSV_PATH'], session_id, ratio, model_name, gpu_name, gap_seconds=gap_seconds)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400


    @app.get('/api/avg_quality_prefix')
    def avg_quality_prefix():
        session_id = request.args.get('session_id', '').strip()
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        gpu_name = request.args.get('gpu', app.config['DEFAULT_GPU']).strip() or app.config['DEFAULT_GPU']
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        try:
            gap_seconds = parse_gap_seconds()
            result = compute_avg_quality_prefix_point(app.config['CSV_PATH'], session_id, model_name, gpu_name, gap_seconds=gap_seconds)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/avg_quality_compaction')
    def avg_quality_compaction():
        session_id = request.args.get('session_id', '').strip()
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        gpu_name = request.args.get('gpu', app.config['DEFAULT_GPU']).strip() or app.config['DEFAULT_GPU']
        ratio_text = request.args.get('ratio', '').strip()
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        if ratio_text == '':
            return jsonify({'error': 'Missing ratio'}), 400
        try:
            ratio = float(ratio_text)
            if ratio > 1.0:
                ratio /= 100.0
            gap_seconds = parse_gap_seconds()
            result = compute_avg_quality_compaction_point(app.config['CSV_PATH'], session_id, ratio, model_name, gpu_name, gap_seconds=gap_seconds)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    @app.get('/api/kv_compaction')
    def kv_compaction():
        session_id = request.args.get('session_id', '').strip()
        model_name = request.args.get('model', app.config['DEFAULT_MODEL']).strip() or app.config['DEFAULT_MODEL']
        ratio_text = request.args.get('ratio', '').strip()
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        if ratio_text == '':
            return jsonify({'error': 'Missing ratio'}), 400
        try:
            ratio = float(ratio_text)
            if ratio > 1.0:
                ratio /= 100.0
            result = compute_kv_compaction_curve(app.config['CSV_PATH'], session_id, ratio, model_name)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'error': str(exc)}), 400

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive trace visualizer')
    parser.add_argument('--csv', required=True, help='Path to the cost CSV file')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.csv)
    app.run(host=args.host, port=args.port, debug=True)
