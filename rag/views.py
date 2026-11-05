"""
HTTP layer. Views only handle requests and responses; the RAG logic
lives in engine.py so it can be tested and reused without Django.
"""
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from . import engine

MAX_QUESTION_LENGTH = 500


@require_GET
def home(request):
    return render(request, "rag/index.html")


# CSRF protection stops other websites from using a logged-in user's cookies to
# send requests. This app has no login or cookies, so there is nothing to protect,
# and exempting it lets tools like curl or Postman call the API directly.
@csrf_exempt
@require_POST
def ask(request):
    """POST /ask  with JSON {"question": "..."}  ->  {"answer", "mode", "sources"}"""
    try:
        question = json.loads(request.body).get("question", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return JsonResponse({"error": 'Send JSON like {"question": "..."}'}, status=400)

    if not question:
        return JsonResponse({"error": "Question is required."}, status=400)
    if len(question) > MAX_QUESTION_LENGTH:
        return JsonResponse({"error": f"Max {MAX_QUESTION_LENGTH} characters."}, status=400)

    try:
        return JsonResponse(engine.answer(question))
    except FileNotFoundError as e:  # index not built yet
        return JsonResponse({"error": str(e)}, status=503)
    except Exception as e:  # e.g. invalid API key, network error, quota exceeded
        return JsonResponse({"error": f"LLM request failed: {e}"}, status=502)
