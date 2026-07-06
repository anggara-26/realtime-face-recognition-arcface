import json

import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods

from . import imaging
from .engine import ENGINE_CHOICES, cosine_similarity, get_engine, l2_normalize
from .models import Person


def _threshold_for(engine_name: str) -> float:
    return settings.FACE_MATCH_THRESHOLD.get(engine_name, 0.4)


@ensure_csrf_cookie
def index(request):
    context = {
        "engine_choices": ENGINE_CHOICES,
        "default_engine": settings.FACE_DEFAULT_ENGINE,
        "thresholds_json": json.dumps(settings.FACE_MATCH_THRESHOLD),
    }
    return render(request, "faceid/index.html", context)


@require_GET
def list_persons(request):
    engine_name = request.GET.get("engine", settings.FACE_DEFAULT_ENGINE)
    persons = Person.objects.filter(engine=engine_name).order_by("name")
    data = [
        {
            "id": p.id,
            "name": p.name,
            "engine": p.engine,
            "num_source_photos": p.num_source_photos,
            "det_score": p.det_score,
            "photo_url": p.photo.url if p.photo else None,
            "created_at": p.created_at.isoformat(),
        }
        for p in persons
    ]
    return _json({"ok": True, "persons": data})


@require_http_methods(["DELETE"])
def delete_person(request, person_id: int):
    deleted, _ = Person.objects.filter(id=person_id).delete()
    return _json({"ok": deleted > 0})


@require_http_methods(["POST"])
def enroll(request):
    """One-shot (or TTA / multi-photo) enrollment endpoint.

    multipart/form-data:
      name: str (required)
      engine: 'arcface' | 'dlib' (default: settings.FACE_DEFAULT_ENGINE)
      tta: 'true' | 'false' -- augment a single photo (BAB IV.4.4)
      images: one or more image files
    """
    name = (request.POST.get("name") or "").strip()
    engine_name = request.POST.get("engine", settings.FACE_DEFAULT_ENGINE)
    use_tta = request.POST.get("tta", "false").lower() == "true"
    files = request.FILES.getlist("images")

    if not name:
        return _json({"ok": False, "error": "Name is required."}, status=400)
    if engine_name not in ENGINE_CHOICES:
        return _json({"ok": False, "error": f"Unknown engine '{engine_name}'."}, status=400)
    if not files:
        return _json({"ok": False, "error": "At least one image is required."}, status=400)

    try:
        engine = get_engine(engine_name)
    except ImportError as exc:
        return _json({"ok": False, "error": str(exc)}, status=500)

    images = []
    for f in files:
        img = imaging.decode_upload(f.read())
        if img is not None:
            images.append(img)
    if not images:
        return _json({"ok": False, "error": "Could not decode any uploaded image."}, status=400)

    embeddings = []
    best_face = None
    if use_tta and len(images) == 1:
        variants = imaging.tta_variants(images[0])
        for variant in variants:
            face = engine.embed_single(variant)
            if face is not None:
                embeddings.append(face.embedding)
                if best_face is None:
                    best_face = face
        source_photo_count = 1
    else:
        for img in images:
            face = engine.embed_single(img)
            if face is not None:
                embeddings.append(face.embedding)
                if best_face is None:
                    best_face = face
        source_photo_count = len(images)

    if not embeddings:
        return _json({"ok": False, "error": "No face detected in the provided photo(s)."}, status=422)

    mean_embedding = l2_normalize(np.mean(np.stack(embeddings), axis=0))

    person = Person.objects.create(
        name=name,
        engine=engine_name,
        embedding=mean_embedding.tolist(),
        det_score=best_face.det_score if best_face else 0.0,
        num_source_photos=source_photo_count,
    )
    person.photo.save(f"{name}_{person.id}.jpg", ContentFile(imaging.encode_jpeg(images[0])), save=True)

    return _json(
        {
            "ok": True,
            "id": person.id,
            "name": person.name,
            "engine": person.engine,
            "bbox": best_face.bbox if best_face else None,
            "det_score": person.det_score,
            "num_source_photos": person.num_source_photos,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def recognize(request):
    """Real-time recognition endpoint: one webcam frame in, all detected
    faces + their gallery match (or "Tak Dikenal") out.

    CSRF-exempt: this is a stateless JSON API with no session/auth to
    protect, and is called by clients (e.g. over an ngrok tunnel) that
    never load the index page to pick up a CSRF cookie.
    """
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return _json({"ok": False, "error": "Invalid JSON body."}, status=400)

    data_url = payload.get("image")
    engine_name = payload.get("engine", settings.FACE_DEFAULT_ENGINE)
    threshold = payload.get("threshold")
    threshold = float(threshold) if threshold is not None else _threshold_for(engine_name)

    if not data_url:
        return _json({"ok": False, "error": "Missing 'image'."}, status=400)
    if engine_name not in ENGINE_CHOICES:
        return _json({"ok": False, "error": f"Unknown engine '{engine_name}'."}, status=400)

    img = imaging.decode_data_url(data_url)
    if img is None:
        return _json({"ok": False, "error": "Could not decode image."}, status=400)

    try:
        engine = get_engine(engine_name)
    except ImportError as exc:
        return _json({"ok": False, "error": str(exc)}, status=500)

    gallery = list(Person.objects.filter(engine=engine_name))
    gallery_matrix = (
        np.stack([np.asarray(p.embedding, dtype=np.float32) for p in gallery])
        if gallery
        else None
    )

    detections = []
    for face in engine.get_faces(img):
        label, score, matched_id = "Tak Dikenal", 0.0, None
        if gallery_matrix is not None:
            sims = gallery_matrix @ face.embedding
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= threshold:
                label, score, matched_id = gallery[best_idx].name, best_score, gallery[best_idx].id
            else:
                score = best_score
        detections.append(
            {
                "bbox": face.bbox,
                "det_score": face.det_score,
                "label": label,
                "score": score,
                "person_id": matched_id,
            }
        )

    return _json({"ok": True, "engine": engine_name, "threshold": threshold, "faces": detections})


def _json(data, status: int = 200):
    from django.http import JsonResponse

    return JsonResponse(data, status=status)
