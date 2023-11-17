from urllib.parse import unquote


def remove_separators(payload: str) -> str:
    """Remove line separators from the payload"""
    payload = payload.replace("\\r\\n", "")
    return payload


def remove_url_encoding(payload: str) -> str:
    """Decode URL encoded characters from the payload"""

    # You need this loop to consider repetitive(mostly double) encoding
    while True:
        unquoted_payload = unquote(payload)
        if unquoted_payload == payload:
            break
        payload = unquoted_payload

    return payload

