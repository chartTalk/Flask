import os
from flask import Blueprint, request, jsonify, current_app
from app.handlers.infer import run_inference

bp = Blueprint('routes', __name__)

@bp.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400

    file = request.files['image']
    filename = file.filename
    upload_folder = current_app.config.get('UPLOAD_FOLDER', './uploads')
    os.makedirs(upload_folder, exist_ok=True)
    save_path = os.path.join(upload_folder, filename)
    file.save(save_path)

    try:
        result = run_inference(save_path)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
