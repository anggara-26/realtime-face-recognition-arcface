def no_index_middleware(get_response):
    """Blanket noindex header: this is a demo with enrolled faces + names,
    not something that should ever surface in search results."""

    def middleware(request):
        response = get_response(request)
        response["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        return response

    return middleware
