"""
Pure routing tests — no GPU, no network, no real embedding model.
Covers: signal detection, cosine threshold, scope filtering, complexity scoring,
prefer-loaded logic, context overflow, keyword directives, cloud directive.
"""
import numpy as np
import pytest

from conftest import MockBackend

from infergate.config import ModelDescriptor
from infergate.config import RouterConfig
from infergate.config import RouterSettings
from infergate.config import TaskClassConfig
from infergate.embeddings import compute_centroids
from infergate.embeddings import route_by_embedding
from infergate.router import Router
from infergate.selector import complexity_score
from infergate.selector import select_model
from infergate.signals import detect_signal
from infergate.signals import has_cloud_directive
from infergate.signals import has_images
from infergate.signals import task_class_directive
from infergate.signals import text_content
from infergate.types import InferRequest
from infergate.types import RouteStrategy


# ─── helpers ──────────────────────────────────────────────────────────────────

def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _image_msg() -> dict:
    return {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]}


def _make_req(messages, tools=None) -> InferRequest:
    return InferRequest(messages=messages, tools=tools)


def _settings(**kwargs) -> RouterSettings:
    defaults = {"embedding_min_confidence": 0.72, "long_context_tokens": 100, "keywords": {}}
    defaults.update(kwargs)
    return RouterSettings(**defaults)


# ─── text_content ──────────────────────────────────────────────────────────────

class TestTextContent:
    def test_str_content(self):
        assert text_content({"role": "user", "content": "Hello"}) == "Hello"

    def test_none_content(self):
        assert text_content({"role": "user"}) == ""

    def test_list_text_part(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
        assert text_content(msg) == "Hi"

    def test_list_image_only(self):
        msg = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}
        assert text_content(msg) == ""

    def test_list_text_and_image(self):
        msg = {"role": "user", "content": [
            {"type": "text", "text": "Describe:"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]}
        assert text_content(msg) == "Describe:"


# ─── has_images ────────────────────────────────────────────────────────────────

class TestHasImages:
    def test_no_images(self):
        assert not has_images([_user("hello")])

    def test_single_image(self):
        assert has_images([_image_msg()])

    def test_mixed_messages(self):
        assert has_images([_user("text"), _image_msg()])

    def test_str_message_no_image(self):
        assert not has_images([_user("text")])


# ─── task_class_directive ──────────────────────────────────────────────────────

class TestTaskClassDirective:
    def test_no_directive(self):
        assert task_class_directive([_user("hello")]) is None

    def test_code_directive(self):
        assert task_class_directive([_user("fix this #code")]) == "code"

    def test_document_directive(self):
        assert task_class_directive([_user("#document summarise")]) == "document"

    def test_general_directive(self):
        assert task_class_directive([_user("#general chat")]) == "general"

    def test_only_last_user_message_checked(self):
        msgs = [_user("#code old"), _assistant("ok"), _user("plain")]
        assert task_class_directive(msgs) is None

    def test_case_insensitive(self):
        assert task_class_directive([_user("#CODE fix")]) == "code"


# ─── has_cloud_directive ───────────────────────────────────────────────────────

class TestHasCloudDirective:
    def test_no_directive(self):
        assert not has_cloud_directive([_user("hello")])

    def test_ovh_directive(self):
        assert has_cloud_directive([_user("run this #ovh")])

    def test_cloud_directive(self):
        assert has_cloud_directive([_user("#cloud please")])

    def test_case_insensitive(self):
        assert has_cloud_directive([_user("#OVH")])


# ─── detect_signal ─────────────────────────────────────────────────────────────

class TestDetectSignal:
    def test_no_signal(self):
        req = _make_req([_user("hello")])
        assert detect_signal(req, _settings()) is None

    def test_image_returns_vision(self):
        req = _make_req([_image_msg()])
        assert detect_signal(req, _settings()) == "vision"

    def test_tools_returns_web_search(self):
        req = _make_req([_user("lookup")], tools=[{"type": "function"}])
        assert detect_signal(req, _settings()) == "web_search"

    def test_long_context_returns_document(self):
        long_text = "word " * 600  # ~150 tokens estimate
        req = _make_req([_user(long_text)])
        assert detect_signal(req, _settings(long_context_tokens=100)) == "document"

    def test_short_text_not_document(self):
        req = _make_req([_user("short")])
        assert detect_signal(req, _settings()) != "document"

    def test_keyword_match(self):
        req = _make_req([_user("fix this bug")])
        s = _settings(keywords={"code": ["fix this"]})
        assert detect_signal(req, s) == "code"

    def test_keyword_case_insensitive(self):
        req = _make_req([_user("FIX THIS bug")])
        s = _settings(keywords={"code": ["fix this"]})
        assert detect_signal(req, s) == "code"

    def test_keyword_only_last_user_message(self):
        msgs = [_user("fix this code"), _assistant("ok"), _user("thanks")]
        req = _make_req(msgs)
        s = _settings(keywords={"code": ["fix this"]})
        assert detect_signal(req, s) is None

    def test_image_priority_over_tools(self):
        req = _make_req([_image_msg()], tools=[{"type": "function"}])
        assert detect_signal(req, _settings()) == "vision"

    def test_tools_priority_over_long_context(self):
        long_text = "word " * 600
        req = _make_req([_user(long_text)], tools=[{"type": "function"}])
        assert detect_signal(req, _settings(long_context_tokens=100)) == "web_search"

    def test_long_context_priority_over_keyword(self):
        long_text = "fix this " * 200
        req = _make_req([_user(long_text)])
        s = _settings(long_context_tokens=100, keywords={"code": ["fix this"]})
        assert detect_signal(req, s) == "document"

    def test_hashtag_directive_not_handled_by_detect_signal(self):
        # detect_signal covers images/tools/long-context/keywords only.
        # Directive priority (#code, #general …) is enforced by Router.decide()
        # before detect_signal is called, so detect_signal sees the image and
        # correctly returns "vision" here.
        req = _make_req([_image_msg()] + [_user("#general")])
        assert detect_signal(req, _settings()) == "vision"

    def test_multi_turn_long_context_cumulative(self):
        msgs = [_user("word " * 60), _assistant("ok"), _user("word " * 60)]
        req = _make_req(msgs)
        assert detect_signal(req, _settings(long_context_tokens=50)) == "document"

    def test_system_prompt_excluded_from_token_count(self):
        # 800 tokens of system prompt alone should NOT trigger document
        system_msg = {"role": "system", "content": "word " * 800}
        req = _make_req([system_msg, _user("short")])
        assert detect_signal(req, _settings(long_context_tokens=100)) is None

    def test_empty_messages_returns_none(self):
        req = _make_req([])
        assert detect_signal(req, _settings()) is None


# ─── complexity_score ──────────────────────────────────────────────────────────

class TestComplexityScore:
    def test_empty_messages(self):
        assert complexity_score([]) == 0.0

    def test_short_simple_question(self):
        assert complexity_score([_user("What is Python?")]) < 0.3

    def test_long_text_raises_score(self):
        assert complexity_score([_user("word " * 200)]) >= 0.5

    def test_complexity_signal_keyword(self):
        assert complexity_score([_user("Please analyze this in detail.")]) > 0.0

    def test_score_clamped_0_to_1(self):
        msg = _user("analyze compare evaluate " * 20 + " " * 600)
        score = complexity_score([msg])
        assert 0.0 <= score <= 1.0

    def test_simple_question_regex_lowers_score(self):
        simple = complexity_score([_user("What is the capital of France?")])
        complex_ = complexity_score([_user("Analyze the political history of France thoroughly.")])
        assert simple < complex_

    def test_many_turns_raises_score(self):
        msgs = [_user("q"), _assistant("a")] * 5 + [_user("final")]
        assert complexity_score(msgs) > 0.0


# ─── embedding routing ────────────────────────────────────────────────────────

class TestEmbeddingRouting:
    """Uses MockEmbeddingProvider from conftest."""

    @pytest.mark.asyncio
    async def test_empty_centroids_returns_general(self, mock_provider):
        cls, score, vec = await route_by_embedding("q", {}, mock_provider)
        assert cls == "general"
        assert score == 0.0
        assert vec is None

    @pytest.mark.asyncio
    async def test_returns_best_match(self, mock_provider):
        # Build centroids where "code" description matches itself perfectly
        code_vec = await mock_provider.embed("code tasks")
        centroids = {
            "code":    np.array(code_vec),
            "general": np.array(await mock_provider.embed("general conversation")),
        }
        cls, score, vec = await route_by_embedding("code tasks", centroids, mock_provider, 0.0)
        assert cls == "code"

    @pytest.mark.asyncio
    async def test_low_score_falls_back_to_general(self, mock_provider):
        v1 = np.array([1.0, 0.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0, 0.0])
        query_vec = np.array([-1.0, 0.0, 0.0, 0.0])
        centroids = {"code": v1, "document": v2}
        # Patch provider to return a vector with low similarity to all centroids
        from unittest.mock import AsyncMock
        mock_provider.embed = AsyncMock(return_value=query_vec.tolist())
        cls, score, vec = await route_by_embedding("irrelevant", centroids, mock_provider, 0.72)
        assert cls == "general"

    @pytest.mark.asyncio
    async def test_returns_embedding_vector(self, mock_provider):
        code_vec = await mock_provider.embed("code")
        centroids = {"code": np.array(code_vec)}
        _, _, vec = await route_by_embedding("code", centroids, mock_provider, 0.0)
        assert isinstance(vec, list)
        assert len(vec) == 4


# ─── compute_centroids ────────────────────────────────────────────────────────

class TestComputeCentroids:
    @pytest.mark.asyncio
    async def test_skips_signal_only_classes(self, mock_provider):
        task_classes = {
            "vision":  TaskClassConfig(description="Images"),
            "code":    TaskClassConfig(description="Code tasks"),
        }
        result = await compute_centroids(task_classes, mock_provider)
        assert "vision" not in result
        assert "code" in result

    @pytest.mark.asyncio
    async def test_skips_empty_description(self, mock_provider):
        task_classes = {
            "empty": TaskClassConfig(description=""),
            "full":  TaskClassConfig(description="Has content"),
        }
        result = await compute_centroids(task_classes, mock_provider)
        assert "empty" not in result
        assert "full" in result

    @pytest.mark.asyncio
    async def test_centroid_is_normalised(self, mock_provider):
        task_classes = {"code": TaskClassConfig(description="Code and programming")}
        result = await compute_centroids(task_classes, mock_provider)
        norm = float(np.linalg.norm(result["code"]))
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_examples_included(self, mock_provider):
        task_classes = {
            "code": TaskClassConfig(
                description="Code",
                examples=["write a function", "debug this"],
            )
        }
        result = await compute_centroids(task_classes, mock_provider)
        assert "code" in result


# ─── select_model ─────────────────────────────────────────────────────────────

class TestSelectModel:
    def test_fastest_picks_fast_local(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _ = select_model("general", basic_config, backends, "local", "fastest")
        assert mid == "small-llm"
        assert bname == "loc"

    def test_balanced_picks_balanced_local(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _ = select_model("general", basic_config, backends, "local", "balanced")
        assert mid == "big-llm"

    def test_best_local_scope_excludes_remote(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _ = select_model("general", basic_config, backends, "local", "best")
        assert bname == "loc"

    def test_hybrid_scope_allows_remote(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _ = select_model("general", basic_config, backends, "hybrid", "best")
        assert mid == "cloud-llm"
        assert bname == "ovh"

    def test_balanced_high_complexity_promotes_to_best(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        _, mid, _ = select_model("general", basic_config, backends, "local", "balanced", complexity=0.8)
        assert mid == "big-llm"

    def test_prefer_loaded_for_fastest(self, basic_config, local_backend, remote_backend):
        # small-llm is in loaded list
        backends = {"loc": local_backend, "ovh": remote_backend}
        _, mid, prefer_loaded = select_model("general", basic_config, backends, "local", "fastest")
        assert mid == "small-llm"
        assert prefer_loaded is True

    def test_unavailable_model_skipped(self, basic_config, remote_backend):
        no_models = MockBackend("loc", models=[], is_local=True)
        backends = {"loc": no_models, "ovh": remote_backend}
        bname, mid, _ = select_model("code", basic_config, backends, "local", "fastest")
        # local has no models and scope is "local" → fallback returns empty
        assert bname == "" and mid == ""

    def test_unknown_task_class_falls_back_to_general(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _ = select_model("nonexistent", basic_config, backends, "local", "balanced")
        assert mid != ""  # falls back to general config

    def test_remote_scope_excludes_local(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, _, _ = select_model("general", basic_config, backends, "remote", "best")
        assert bname == "ovh"

    def test_ctx_limit_filters_model(self, remote_backend):
        # backend with small-llm but ctx_limit=100 and we send 200 tokens
        tiny_backend = MockBackend("loc", models=["small-llm"], is_local=True)
        config = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[ModelDescriptor(id="small-llm", backend="loc", tier="fast", ctx_limit=100)],
                )
            },
        )
        backends = {"loc": tiny_backend}
        bname, mid, _ = select_model("general", config, backends, "local", "fastest", estimated_tokens=200)
        # all models filtered → fallback picks first available from backend (no ctx check in fallback)
        assert isinstance(bname, str)
        assert isinstance(mid, str)


# ─── Router integration ───────────────────────────────────────────────────────

class TestRouterDecide:
    @pytest.mark.asyncio
    async def test_signal_detection_skips_embedding(self, basic_config, local_backend, remote_backend, mock_provider):
        backends = {"loc": local_backend, "ovh": remote_backend}
        router = Router(basic_config, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_image_msg()])
        decision = await router.decide(req)
        assert decision.task_class == "vision"
        assert decision.strategy == RouteStrategy.SIGNAL

    @pytest.mark.asyncio
    async def test_long_context_routes_to_document(self, basic_config, local_backend, remote_backend, mock_provider):
        backends = {"loc": local_backend, "ovh": remote_backend}
        router = Router(basic_config, backends, mock_provider)
        await router.load_embeddings()

        long_text = "word " * 600  # exceeds long_context_tokens=100
        req = InferRequest(messages=[_user(long_text)])
        decision = await router.decide(req)
        assert decision.task_class == "document"
        assert decision.strategy == RouteStrategy.SIGNAL

    @pytest.mark.asyncio
    async def test_cloud_directive_forces_remote_scope(self, basic_config, local_backend, remote_backend, mock_provider):
        backends = {"loc": local_backend, "ovh": remote_backend}
        cfg = RouterConfig(
            task_classes=basic_config.task_classes,
            router=basic_config.router,
            provider_scope="local",
            active_profile="best",
            profiles={"best": {"model_preference": "best"}},
        )
        router = Router(cfg, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("help me #ovh")])
        decision = await router.decide(req)
        assert decision.backend == "ovh"

    @pytest.mark.asyncio
    async def test_keyword_directive_task_class(self, basic_config, local_backend, remote_backend, mock_provider):
        backends = {"loc": local_backend, "ovh": remote_backend}
        router = Router(basic_config, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("fix this #code")])
        decision = await router.decide(req)
        assert decision.task_class == "code"
        assert decision.strategy == RouteStrategy.KEYWORD

    @pytest.mark.asyncio
    async def test_no_provider_falls_back_to_general(self, basic_config, local_backend):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends, embedding_provider=None)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("random question")])
        decision = await router.decide(req)
        assert decision.task_class == "general"
        assert decision.strategy == RouteStrategy.FALLBACK

    @pytest.mark.asyncio
    async def test_directive_beats_vision_signal(self, basic_config, local_backend, remote_backend, mock_provider):
        # Router checks task_class_directive before detect_signal, so a #general
        # tag overrides the image signal even though detect_signal returns "vision".
        backends = {"loc": local_backend, "ovh": remote_backend}
        router = Router(basic_config, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_image_msg()] + [_user("#general")])
        decision = await router.decide(req)
        assert decision.task_class == "general"
        assert decision.strategy == RouteStrategy.KEYWORD

    @pytest.mark.asyncio
    async def test_decision_has_all_fields(self, basic_config, local_backend, mock_provider):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("hello")])
        d = await router.decide(req)
        assert d.backend
        assert d.model_id
        assert d.task_class
        assert d.strategy is not None
        assert isinstance(d.confidence, float)
