"""The model client notebook 20 asks for pipeline specifications. An Ollama
server on this machine is the default and needs no key; any other host takes a
deliberate `Endpoint.allow_remote` from a caller that has shown the user the
host. Keys come from the environment, never from a .blend, which gets shared.
"""

import json
import os
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_ENDPOINT = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
TIMEOUT_SECONDS = 120

# Hosts treated as this machine. Everything else is remote.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")

SYSTEM_PROMPT = """You write SciGraphs pipeline specifications.

Answer with one JSON object and nothing else: no prose, no explanation, no code
fence. The object must follow the schema below.

Rules that matter more than they look:

* A dataset that arrives with real coordinates -- an OSMnx street network, a
  SuiteSparse matrix with a coordinate file -- must NOT be given a `layout`
  section. Those coordinates are the data; a force-directed layout would throw
  the geography away. Use `layout` only for graphs with no positions of their own.
* Centrality measures are heavy-tailed. Set `visual.color_norm` to `RANK` when
  coloring by one, or almost every node lands in the darkest color.
* Prefer `node_radius_rel` and `edge_radius_rel` to the absolute forms: they are
  a fraction of the graph's extent and survive any `layout.scale`.
* Set `render.view_transform` to `Standard` whenever color encodes a value.
  The default tone-maps the image and breaks the correspondence.
* Include a `render` section, or the specification produces no image.

SCHEMA
------
%s
"""


class EndpointError(RuntimeError):
    """Unreachable, refused, or answered with something unusable."""


class Endpoint:
    """Where to send a request, and whether that is allowed."""

    def __init__(self, url=DEFAULT_ENDPOINT, model=DEFAULT_MODEL,
                 allow_remote=False, api_key_env=""):
        self.url = (url or DEFAULT_ENDPOINT).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.allow_remote = bool(allow_remote)
        self.api_key_env = api_key_env or ""

    @property
    def host(self):
        return urllib.parse.urlparse(self.url).hostname or ""

    @property
    def is_local(self):
        return self.host in LOCAL_HOSTS

    def check(self):
        """Raise unless this endpoint may be contacted."""
        if not self.url.startswith(("http://", "https://")):
            raise EndpointError("Endpoint must be an http or https URL")
        if not self.is_local and not self.allow_remote:
            raise EndpointError(
                "%s is not this machine. Tick 'Allow remote endpoint' if you "
                "intend the prompt to leave your computer." % self.host
            )

    def api_key(self):
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


def generate(endpoint, request_text, digest, json_schema=None,
             temperature=0.0, seed=42):
    """Ask the endpoint for a specification, in Ollama's /api/generate shape.
    `json_schema` becomes the response format, so an endpoint honoring it cannot
    invent field names. The fixed `temperature` and `seed` keep one request
    reproducing one specification hash, which published figures depend on.
    """
    endpoint.check()

    payload = {
        "model": endpoint.model,
        "prompt": request_text,
        "system": SYSTEM_PROMPT % digest,
        "stream": False,
        "format": json_schema if json_schema else "json",
        "options": {"temperature": temperature, "seed": seed},
    }

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.url + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    key = endpoint.api_key()
    if key:
        request.add_header("Authorization", "Bearer " + key)

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise EndpointError(
            "Could not reach %s (%s). If you meant to use a local model, is "
            "Ollama running?" % (endpoint.url, reason)
        )
    except OSError as exc:
        raise EndpointError("Could not reach %s (%s)" % (endpoint.url, exc))

    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        raise EndpointError("Endpoint did not return JSON")

    text = envelope.get("response")
    if not isinstance(text, str) or not text.strip():
        error = envelope.get("error")
        raise EndpointError(error or "Endpoint returned an empty response")

    return _parse_object(text)


def _parse_object(text):
    """Pull one JSON object out of a reply that may be wrapped in prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start, depth = text.find("{"), 0
    if start >= 0:
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
    raise EndpointError("Could not find a JSON object in the reply")
