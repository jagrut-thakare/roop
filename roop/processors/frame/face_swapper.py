from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    source_paths = roop.globals.source_path
    if isinstance(source_paths, list):
        images = [cv2.imread(p) for p in source_paths]
        # process each image as needed
    else:
        image = cv2.imread(source_paths)
        # process single image as needed

    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    if isinstance(roop.globals.source_path, list):
        for p in roop.globals.source_path:
            if not get_one_face(cv2.imread(p)):
                update_status(f'No face detected in source path: {p}', NAME)
                return False
    else:
        if not get_one_face(cv2.imread(roop.globals.source_path)):
            update_status('No face in source path detected.', NAME)
            return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_faces: List[Face], reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for i, target_face in enumerate(many_faces):
                # Use corresponding source face or fallback to first
                source_face = source_faces[i] if i < len(source_faces) else source_faces[0]
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_faces[0], target_face, temp_frame)
    return temp_frame


def process_frames(source_paths: List[str], temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_faces = [get_one_face(cv2.imread(path)) for path in source_paths]
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            update_status(f'Could not load frame: {temp_frame_path}', NAME)
            continue
        result = process_frame(source_faces, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    source_faces = [get_one_face(cv2.imread(path)) for path in source_paths]
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        update_status(f'Could not load target image: {target_path}', NAME)
        return
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_faces, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_paths, temp_frame_paths, process_frames)
