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
from infergate.types import NoModelAvailable
from infergate.types import RouteStrategy
from infergate.types import RouteTrace


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

    def test_tools_custom_task_class(self):
        req = _make_req([_user("lookup")], tools=[{"type": "function"}])
        assert detect_signal(req, _settings(tools_task_class="agent")) == "agent"

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
            "vision":  TaskClassConfig(description="Images", signal_only=True),
            "code":    TaskClassConfig(description="Code tasks"),
        }
        result = await compute_centroids(task_classes, mock_provider)
        assert "vision" not in result
        assert "code" in result

    @pytest.mark.asyncio
    async def test_signal_only_false_is_included(self, mock_provider):
        task_classes = {
            "vision":  TaskClassConfig(description="Images", signal_only=False),
            "code":    TaskClassConfig(description="Code tasks"),
        }
        result = await compute_centroids(task_classes, mock_provider)
        assert "vision" in result
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
        bname, mid, _, _ = select_model("general", basic_config, backends, "local", "fastest")
        assert mid == "small-llm"
        assert bname == "loc"

    def test_balanced_picks_balanced_local(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _, _ = select_model("general", basic_config, backends, "local", "balanced")
        assert mid == "big-llm"

    def test_best_local_scope_excludes_remote(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _, _ = select_model("general", basic_config, backends, "local", "best")
        assert bname == "loc"

    def test_hybrid_scope_allows_remote(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _, _ = select_model("general", basic_config, backends, "hybrid", "best")
        assert mid == "cloud-llm"
        assert bname == "ovh"

    def test_balanced_high_complexity_promotes_to_best(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        _, mid, _, _ = select_model("general", basic_config, backends, "local", "balanced", complexity=0.8)
        assert mid == "big-llm"

    def test_prefer_loaded_for_fastest(self, basic_config, local_backend, remote_backend):
        # small-llm is in loaded list
        backends = {"loc": local_backend, "ovh": remote_backend}
        _, mid, prefer_loaded, _ = select_model("general", basic_config, backends, "local", "fastest")
        assert mid == "small-llm"
        assert prefer_loaded is True

    def test_unavailable_model_skipped(self, basic_config, remote_backend):
        no_models = MockBackend("loc", models=[], is_local=True)
        backends = {"loc": no_models, "ovh": remote_backend}
        # scope="local" filters out ovh; no local models → NoModelAvailable
        with pytest.raises(NoModelAvailable):
            select_model("code", basic_config, backends, "local", "fastest")

    def test_no_backends_raises_no_model_available(self, basic_config):
        with pytest.raises(NoModelAvailable) as exc_info:
            select_model("general", basic_config, {}, "local", "balanced")
        assert exc_info.value.task_class == "general"
        assert exc_info.value.scope == "local"

    def test_unknown_task_class_falls_back_to_general(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, mid, _, _ = select_model("nonexistent", basic_config, backends, "local", "balanced")
        assert mid != ""  # falls back to general config

    def test_remote_scope_excludes_local(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        bname, _, _, _ = select_model("general", basic_config, backends, "remote", "best")
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
        bname, mid, _, _ = select_model("general", config, backends, "local", "fastest", estimated_tokens=200)
        # all models filtered → fallback picks first available from backend (no ctx check in fallback)
        assert isinstance(bname, str)
        assert isinstance(mid, str)


# ─── P3 gap coverage ─────────────────────────────────────────────────────────

class TestModalityFilter:
    def test_text_model_excluded_for_vision(self, local_backend):
        config = RouterConfig(
            task_classes={
                "vision": TaskClassConfig(
                    description="Vision",
                    signal_only=True,
                    models=[
                        ModelDescriptor(id="small-llm", backend="loc", tier="fast", modality="text"),
                        ModelDescriptor(id="big-llm",   backend="loc", tier="fast", modality="vision"),
                    ],
                ),
            },
        )
        backends = {"loc": local_backend}
        bname, mid, _, _ = select_model("vision", config, backends, "local", "fastest",
                                     required_modality="vision")
        assert mid == "big-llm"

    def test_any_modality_satisfies_vision_requirement(self, local_backend):
        config = RouterConfig(
            task_classes={
                "vision": TaskClassConfig(
                    description="Vision",
                    signal_only=True,
                    models=[
                        ModelDescriptor(id="small-llm", backend="loc", tier="fast", modality="any"),
                    ],
                ),
            },
        )
        backends = {"loc": local_backend}
        bname, mid, _, _ = select_model("vision", config, backends, "local", "fastest",
                                     required_modality="vision")
        assert mid == "small-llm"

    def test_no_modality_filter_when_none(self, basic_config, local_backend):
        backends = {"loc": local_backend}
        bname, mid, _, _ = select_model("general", basic_config, backends, "local", "fastest",
                                     required_modality=None)
        assert mid != ""


class TestComplexityPromoteFast:
    def test_fast_promoted_to_balanced_when_threshold_met(self, local_backend):
        config = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[
                        ModelDescriptor(id="small-llm", backend="loc", tier="fast"),
                        ModelDescriptor(id="big-llm",   backend="loc", tier="balanced"),
                    ],
                ),
            },
            router=RouterSettings(complexity_promote_fast_threshold=0.5),
        )
        backends = {"loc": local_backend}
        _, mid, _, _ = select_model("general", config, backends, "local", "fastest", complexity=0.8)
        assert mid == "big-llm"  # promoted from fast → balanced

    def test_fast_not_promoted_when_threshold_none(self, basic_config, local_backend):
        backends = {"loc": local_backend}
        _, mid, _, _ = select_model("general", basic_config, backends, "local", "fastest", complexity=0.9)
        assert mid == "small-llm"  # threshold=None → no promotion

    def test_fast_not_promoted_when_below_threshold(self, local_backend):
        config = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[
                        ModelDescriptor(id="small-llm", backend="loc", tier="fast"),
                        ModelDescriptor(id="big-llm",   backend="loc", tier="balanced"),
                    ],
                ),
            },
            router=RouterSettings(complexity_promote_fast_threshold=0.8),
        )
        backends = {"loc": local_backend}
        _, mid, _, _ = select_model("general", config, backends, "local", "fastest", complexity=0.5)
        assert mid == "small-llm"  # below threshold → no promotion


class TestForceTier:
    def test_force_tier_overrides_profile(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        _, mid, _, _ = select_model("general", basic_config, backends, "hybrid", "fastest",
                                  force_tier="best")
        assert mid == "cloud-llm"

    def test_force_tier_skips_complexity_promotion(self, local_backend):
        config = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[
                        ModelDescriptor(id="small-llm", backend="loc", tier="fast"),
                        ModelDescriptor(id="big-llm",   backend="loc", tier="balanced"),
                    ],
                ),
            },
            router=RouterSettings(complexity_promote_fast_threshold=0.3),
        )
        backends = {"loc": local_backend}
        # force_tier="fastest" bypasses threshold even though complexity is high
        _, mid, _, _ = select_model("general", config, backends, "local", "balanced",
                                  complexity=0.9, force_tier="fastest")
        assert mid == "small-llm"

    @pytest.mark.asyncio
    async def test_force_tier_via_infer_request(self, basic_config, local_backend, remote_backend, mock_provider):
        backends = {"loc": local_backend, "ovh": remote_backend}
        cfg = RouterConfig(
            task_classes=basic_config.task_classes,
            router=basic_config.router,
            provider_scope="hybrid",
            active_profile="fast",
            profiles={"fast": {"model_preference": "fastest"}},
        )
        router = Router(cfg, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("hello")], force_tier="best")
        decision = await router.decide(req)
        assert decision.model_id == "cloud-llm"


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
    async def test_custom_task_class_directive(self, local_backend, mock_provider):
        config = RouterConfig(
            task_classes={
                "sql": TaskClassConfig(
                    description="SQL queries and database operations",
                    models=[ModelDescriptor(id="small-llm", backend="loc", tier="fast")],
                ),
                "general": TaskClassConfig(
                    description="General conversation",
                    models=[ModelDescriptor(id="small-llm", backend="loc", tier="fast")],
                ),
            },
            router=RouterSettings(),
            provider_scope="local",
            active_profile="fast",
            profiles={"fast": {"model_preference": "fastest"}},
        )
        backends = {"loc": local_backend}
        router = Router(config, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("run this #sql")])
        decision = await router.decide(req)
        assert decision.task_class == "sql"
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


class TestTaskDirectiveField:
    @pytest.mark.asyncio
    async def test_directive_present_sets_field(self, basic_config, local_backend, mock_provider):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("fix this #code")])
        d = await router.decide(req)
        assert d.task_directive == "code"

    @pytest.mark.asyncio
    async def test_no_directive_is_none(self, basic_config, local_backend, mock_provider):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("hello world")])
        d = await router.decide(req)
        assert d.task_directive is None

    @pytest.mark.asyncio
    async def test_signal_route_has_no_directive(self, basic_config, local_backend, mock_provider):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_image_msg()])
        d = await router.decide(req)
        assert d.strategy == RouteStrategy.SIGNAL
        assert d.task_directive is None


class TestRouterReselect:
    def test_reselect_returns_local_model(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        router = Router(basic_config, backends)
        d = router.reselect("code", scope="local")
        assert d.backend == "loc"
        assert d.strategy == RouteStrategy.RESELECT
        assert d.confidence == 1.0

    def test_reselect_remote_scope_picks_remote(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        router = Router(basic_config, backends)
        d = router.reselect("code", scope="remote")
        assert d.backend == "ovh"
        assert d.strategy == RouteStrategy.RESELECT

    def test_reselect_force_tier_best(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        router = Router(basic_config, backends)
        d = router.reselect("code", scope="local", force_tier="best")
        assert d.model_id == "big-llm"

    def test_reselect_local_plus_remote_scope(self, basic_config, local_backend, remote_backend):
        backends = {"loc": local_backend, "ovh": remote_backend}
        cfg = RouterConfig(
            task_classes=basic_config.task_classes,
            router=basic_config.router,
            provider_scope="local",
            active_profile="best",
            profiles={"best": {"model_preference": "best"}},
        )
        router = Router(cfg, backends)
        d = router.reselect("code", scope="local+remote", force_tier="best")
        assert d.strategy == RouteStrategy.RESELECT
        assert d.backend in ("loc", "ovh")

    def test_reselect_task_directive_not_set(self, basic_config, local_backend):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends)
        d = router.reselect("general")
        assert d.task_directive is None


class TestEstimatedTokens:
    @pytest.mark.asyncio
    async def test_tokens_nonzero_for_real_prompt(self, basic_config, local_backend):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends)
        req = InferRequest(messages=[{"role": "user", "content": "word " * 40}])
        d = await router.decide(req)
        assert d.estimated_tokens > 0

    @pytest.mark.asyncio
    async def test_tokens_proportional_to_length(self, basic_config, local_backend):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends)
        short = InferRequest(messages=[{"role": "user", "content": "hi"}])
        long  = InferRequest(messages=[{"role": "user", "content": "word " * 200}])
        d_short = await router.decide(short)
        d_long  = await router.decide(long)
        assert d_long.estimated_tokens > d_short.estimated_tokens

    @pytest.mark.asyncio
    async def test_tokens_empty_message_is_zero(self, basic_config, local_backend):
        backends = {"loc": local_backend}
        router = Router(basic_config, backends)
        req = InferRequest(messages=[{"role": "user", "content": ""}])
        d = await router.decide(req)
        assert d.estimated_tokens == 0


class TestCostField:
    def test_cost_defaults_to_none(self):
        from infergate.config import ModelDescriptor
        m = ModelDescriptor(id="x", backend="b", tier="fast")
        assert m.cost_per_1k_tokens is None

    def test_cost_roundtrips_through_from_dict(self):
        cfg = RouterConfig.from_dict({
            "task_classes": {
                "general": {
                    "description": "general",
                    "models": [{"id": "m", "backend": "b", "tier": "fast",
                                "cost_per_1k_tokens": 0.002}],
                }
            }
        })
        assert cfg.task_classes["general"].models[0].cost_per_1k_tokens == 0.002

    def test_cost_absent_in_dict_stays_none(self):
        cfg = RouterConfig.from_dict({
            "task_classes": {
                "general": {
                    "description": "general",
                    "models": [{"id": "m", "backend": "b", "tier": "fast"}],
                }
            }
        })
        assert cfg.task_classes["general"].models[0].cost_per_1k_tokens is None

    @pytest.mark.asyncio
    async def test_estimated_cost_usd_none_when_no_cost_data(self, basic_config, local_backend):
        router = Router(basic_config, {"loc": local_backend})
        d = await router.decide(InferRequest(messages=[_user("hello")]))
        assert d.estimated_cost_usd is None

    @pytest.mark.asyncio
    async def test_estimated_cost_usd_computed_when_cost_set(self, local_backend):
        from conftest import MockBackend
        priced_backend = MockBackend(name="loc", models=["priced-llm"], loaded=[], is_local=True)
        cfg = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[ModelDescriptor(id="priced-llm", backend="loc", tier="fast",
                                           cost_per_1k_tokens=0.002)],
                ),
            },
            router=RouterSettings(),
            provider_scope="local",
            active_profile="fast",
            profiles={"fast": {"model_preference": "fastest"}},
        )
        router = Router(cfg, {"loc": priced_backend})
        # ~4 chars per token; 400 chars → 100 tokens
        d = await router.decide(InferRequest(messages=[_user("x" * 400)]))
        assert d.estimated_tokens == 100
        assert d.estimated_cost_usd == pytest.approx(0.002 * 100 / 1000)

    @pytest.mark.asyncio
    async def test_estimated_cost_usd_none_when_tokens_zero(self, local_backend):
        from conftest import MockBackend
        priced_backend = MockBackend(name="loc", models=["priced-llm"], loaded=[], is_local=True)
        cfg = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[ModelDescriptor(id="priced-llm", backend="loc", tier="fast",
                                           cost_per_1k_tokens=0.002)],
                ),
            },
            router=RouterSettings(),
            provider_scope="local",
            active_profile="fast",
            profiles={"fast": {"model_preference": "fastest"}},
        )
        router = Router(cfg, {"loc": priced_backend})
        d = await router.decide(InferRequest(messages=[_user("")]))
        assert d.estimated_cost_usd is None


class TestRouteTrace:
    @pytest.mark.asyncio
    async def test_trace_false_gives_none(self, basic_config, local_backend):
        router = Router(basic_config, {"loc": local_backend})
        d = await router.decide(InferRequest(messages=[_user("hello")]))
        assert d.trace is None

    @pytest.mark.asyncio
    async def test_trace_true_gives_route_trace(self, basic_config, local_backend):
        from infergate.types import RouteTrace
        router = Router(basic_config, {"loc": local_backend})
        d = await router.decide(InferRequest(messages=[_user("hello")]), trace=True)
        assert isinstance(d.trace, RouteTrace)

    def test_scope_source_default_is_global(self):
        trace = RouteTrace()
        assert trace.scope_source == "global"

    @pytest.mark.asyncio
    async def test_scope_source_global(self, basic_config, local_backend):
        router = Router(basic_config, {"loc": local_backend})
        d = await router.decide(InferRequest(messages=[_user("hello")]), trace=True)
        assert d.trace.scope_source == "global"

    @pytest.mark.asyncio
    async def test_scope_source_cloud_directive(self, basic_config, local_backend, remote_backend):
        router = Router(basic_config, {"loc": local_backend, "ovh": remote_backend})
        d = await router.decide(InferRequest(messages=[_user("help #ovh")]), trace=True)
        assert d.trace.scope_source == "cloud_directive"

    @pytest.mark.asyncio
    async def test_scope_source_class_override(self, local_backend):
        cfg = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[ModelDescriptor(id="small-llm", backend="loc", tier="fast")],
                    scope_override="local",
                ),
            },
            router=RouterSettings(),
            provider_scope="local",
            active_profile="fast",
            profiles={"fast": {"model_preference": "fastest"}},
        )
        router = Router(cfg, {"loc": local_backend})
        d = await router.decide(InferRequest(messages=[_user("hello")]), trace=True)
        assert d.trace.scope_source == "class_override"

    @pytest.mark.asyncio
    async def test_eliminated_scope_reason(self, basic_config, local_backend, remote_backend):
        # provider_scope=local → remote backend models eliminated with "scope"
        router = Router(basic_config, {"loc": local_backend, "ovh": remote_backend})
        d = await router.decide(
            InferRequest(messages=[_user("fix this #code")]), trace=True
        )
        reasons = {e.reason for e in d.trace.eliminated}
        assert "scope" in reasons
        scoped_out = [e for e in d.trace.eliminated if e.reason == "scope"]
        assert all(e.backend == "ovh" for e in scoped_out)

    @pytest.mark.asyncio
    async def test_eliminated_no_backend_reason(self, local_backend):
        # Config references a backend "ghost" that is not registered
        cfg = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[
                        ModelDescriptor(id="ghost-model", backend="ghost", tier="fast"),
                        ModelDescriptor(id="small-llm",   backend="loc",   tier="fast"),
                    ],
                ),
            },
            router=RouterSettings(),
            provider_scope="local",
            active_profile="fast",
            profiles={"fast": {"model_preference": "fastest"}},
        )
        router = Router(cfg, {"loc": local_backend})
        d = await router.decide(InferRequest(messages=[_user("hello")]), trace=True)
        assert any(e.reason == "no_backend" and e.model_id == "ghost-model"
                   for e in d.trace.eliminated)

    @pytest.mark.asyncio
    async def test_eliminated_ctx_limit_reason(self, local_backend):
        cfg = RouterConfig(
            task_classes={
                "general": TaskClassConfig(
                    description="General",
                    models=[
                        ModelDescriptor(id="small-llm", backend="loc", tier="fast", ctx_limit=10),
                        ModelDescriptor(id="big-llm",   backend="loc", tier="best", ctx_limit=32768),
                    ],
                ),
            },
            router=RouterSettings(),
            provider_scope="local",
            active_profile="best",
            profiles={"best": {"model_preference": "best"}},
        )
        router = Router(cfg, {"loc": local_backend})
        long_req = InferRequest(messages=[_user("word " * 200)])
        d = await router.decide(long_req, trace=True)
        assert any(e.reason == "ctx_limit" and e.model_id == "small-llm"
                   for e in d.trace.eliminated)
        assert d.model_id == "big-llm"

    @pytest.mark.asyncio
    async def test_eliminated_modality_reason(self, local_backend):
        from conftest import MockBackend
        vision_backend = MockBackend(
            name="loc",
            models=["text-model", "vision-model"],
            loaded=[],
            is_local=True,
        )
        cfg = RouterConfig(
            task_classes={
                "vision": TaskClassConfig(
                    description="Vision tasks",
                    models=[
                        ModelDescriptor(id="text-model",   backend="loc", tier="fast", modality="text"),
                        ModelDescriptor(id="vision-model", backend="loc", tier="fast", modality="vision"),
                    ],
                    signal_only=True,
                ),
            },
            router=RouterSettings(),
            provider_scope="local",
            active_profile="fast",
            profiles={"fast": {"model_preference": "fastest"}},
        )
        router = Router(cfg, {"loc": vision_backend})
        d = await router.decide(InferRequest(messages=[_image_msg()]), trace=True)
        assert any(e.reason == "modality" and e.model_id == "text-model"
                   for e in d.trace.eliminated)
        assert d.model_id == "vision-model"

    @pytest.mark.asyncio
    async def test_embedding_ms_none_on_signal_path(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()
        d = await router.decide(InferRequest(messages=[_image_msg()]), trace=True)
        assert d.strategy == RouteStrategy.SIGNAL
        assert d.trace.embedding_ms is None

    @pytest.mark.asyncio
    async def test_embedding_ms_set_on_embedding_path(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()
        # plain message with no signal triggers embedding path
        d = await router.decide(InferRequest(messages=[_user("hello world")]), trace=True)
        assert d.strategy in (RouteStrategy.EMBEDDING, RouteStrategy.FALLBACK)
        assert d.trace.embedding_ms is not None
        assert d.trace.embedding_ms >= 0.0


class TestEmbedCache:
    @pytest.mark.asyncio
    async def test_cache_miss_on_first_call(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()
        d = await router.decide(InferRequest(messages=[_user("hello world")]), trace=True)
        assert d.trace.cache_hit is False

    @pytest.mark.asyncio
    async def test_cache_hit_on_second_call(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()
        req = InferRequest(messages=[_user("hello world")])
        await router.decide(req)
        d = await router.decide(req, trace=True)
        assert d.trace.cache_hit is True

    @pytest.mark.asyncio
    async def test_cache_hit_skips_embed_call(self, basic_config, local_backend):
        embed_calls = []

        class CountingProvider:
            async def embed(self, text):
                embed_calls.append(text)
                import numpy as np
                return np.random.default_rng(0).standard_normal(4).tolist()
            async def embed_batch(self, texts):
                return [await self.embed(t) for t in texts]

        router = Router(basic_config, {"loc": local_backend}, CountingProvider())
        await router.load_embeddings()
        req = InferRequest(messages=[_user("unique query xyz")])
        await router.decide(req)
        calls_after_first = len(embed_calls)
        await router.decide(req)
        assert len(embed_calls) == calls_after_first  # no new embed call

    @pytest.mark.asyncio
    async def test_cache_hit_none_on_signal_path(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()
        d = await router.decide(InferRequest(messages=[_image_msg()]), trace=True)
        assert d.trace.cache_hit is None  # signal path, embedding never attempted

    @pytest.mark.asyncio
    async def test_cache_disabled_when_size_zero(self, local_backend, mock_provider):
        from infergate.config import RouterSettings
        cfg = RouterConfig(
            task_classes=RouterConfig.from_dict({
                "task_classes": {"general": {"description": "g", "models": [
                    {"id": "small-llm", "backend": "loc", "tier": "fast"}
                ]}}
            }).task_classes,
            router=RouterSettings(embedding_cache_size=0),
            provider_scope="local",
            active_profile="fast",
            profiles={"fast": {"model_preference": "fastest"}},
        )
        router = Router(cfg, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()
        req = InferRequest(messages=[_user("hello world")])
        await router.decide(req)
        d = await router.decide(req, trace=True)
        assert d.trace.cache_hit is False  # always miss when disabled

    def test_lru_eviction(self, basic_config, local_backend):
        from infergate.router import _EmbedCache
        cache = _EmbedCache(maxsize=2)
        cache.put("a", ("general", 0.5, None))
        cache.put("b", ("code",    0.9, None))
        cache.get("a")           # touch "a" → "b" is now LRU
        cache.put("c", ("doc",   0.8, None))  # evicts "b"
        assert cache.get("b") is None
        assert cache.get("a") is not None
        assert cache.get("c") is not None

    def test_cache_size_config_field(self):
        cfg = RouterConfig.from_dict({
            "task_classes": {},
            "router": {"embedding_cache_size": 100},
        })
        assert cfg.router.embedding_cache_size == 100

    def test_cache_size_default(self):
        cfg = RouterConfig.from_dict({"task_classes": {}})
        assert cfg.router.embedding_cache_size == 512

    def test_cache_stats_initial(self, basic_config, local_backend):
        router = Router(basic_config, {"loc": local_backend})
        stats = router.cache_stats()
        assert stats == {"hits": 0, "misses": 0, "size": 0, "capacity": 512}

    def test_cache_stats_hit_miss_tracking(self, basic_config, local_backend):
        from infergate.router import _EmbedCache
        cache = _EmbedCache(maxsize=4)
        cache.get("x")                        # miss
        cache.put("x", ("general", 0.5, None))
        cache.get("x")                        # hit
        cache.get("y")                        # miss
        assert cache._hits == 1
        assert cache._misses == 2

    def test_cache_stats_capacity_from_config(self, local_backend):
        from infergate.config import RouterSettings
        cfg = RouterConfig(task_classes={}, router=RouterSettings(embedding_cache_size=64))
        router = Router(cfg, {"loc": local_backend})
        assert router.cache_stats()["capacity"] == 64

    def test_cache_stats_zero_capacity(self, basic_config, local_backend):
        from infergate.config import RouterSettings
        cfg = RouterConfig(task_classes={}, router=RouterSettings(embedding_cache_size=0))
        router = Router(cfg, {"loc": local_backend})
        from infergate.router import _EmbedCache
        cache = _EmbedCache(maxsize=0)
        cache.get("x")  # miss (disabled)
        assert cache._misses == 1
        assert cache._hits == 0


class TestDecideBatch:
    @pytest.mark.asyncio
    async def test_empty_batch_returns_empty(self, basic_config, local_backend):
        router = Router(basic_config, {"loc": local_backend})
        assert await router.decide_batch([]) == []

    @pytest.mark.asyncio
    async def test_batch_length_matches_input(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()
        reqs = [InferRequest(messages=[_user(f"query {i}")]) for i in range(5)]
        results = await router.decide_batch(reqs)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_identical_queries_one_embed_batch_call(self, basic_config, local_backend):
        batch_calls = []

        class CountingProvider:
            async def embed(self, text):
                import numpy as np
                return np.random.default_rng(0).standard_normal(4).tolist()
            async def embed_batch(self, texts):
                batch_calls.append(len(texts))
                import numpy as np
                rng = np.random.default_rng(0)
                return [rng.standard_normal(4).tolist() for _ in texts]

        router = Router(basic_config, {"loc": local_backend}, CountingProvider())
        await router.load_embeddings()
        batch_calls.clear()  # reset — load_embeddings() calls embed_batch per task class

        reqs = [InferRequest(messages=[_user("same query")]) for _ in range(4)]
        await router.decide_batch(reqs)

        # embed_batch() called exactly once, with exactly one unique query
        assert sum(batch_calls) == 1

    @pytest.mark.asyncio
    async def test_distinct_queries_batched_in_one_call(self, basic_config, local_backend):
        batch_calls: list[int] = []

        class CountingProvider:
            async def embed(self, text):
                import numpy as np
                return np.random.default_rng(hash(text) % 2**31).standard_normal(4).tolist()
            async def embed_batch(self, texts):
                batch_calls.append(len(texts))
                import numpy as np
                return [
                    np.random.default_rng(hash(t) % 2**31).standard_normal(4).tolist()
                    for t in texts
                ]

        router = Router(basic_config, {"loc": local_backend}, CountingProvider())
        await router.load_embeddings()
        batch_calls.clear()

        reqs = [InferRequest(messages=[_user(f"distinct query {i}")]) for i in range(3)]
        await router.decide_batch(reqs)

        assert len(batch_calls) == 1      # exactly one embed_batch() call
        assert batch_calls[0] == 3        # all 3 distinct queries sent together

    @pytest.mark.asyncio
    async def test_signal_requests_skip_embed_batch(self, basic_config, local_backend):
        batch_calls: list[int] = []

        class CountingProvider:
            async def embed(self, text):
                import numpy as np
                return np.random.default_rng(0).standard_normal(4).tolist()
            async def embed_batch(self, texts):
                batch_calls.append(len(texts))
                import numpy as np
                return [np.random.default_rng(0).standard_normal(4).tolist() for _ in texts]

        router = Router(basic_config, {"loc": local_backend}, CountingProvider())
        await router.load_embeddings()
        batch_calls.clear()

        reqs = [
            InferRequest(messages=[_image_msg()]),   # signal → vision, no embed
            InferRequest(messages=[_image_msg()]),   # signal → vision, no embed
            InferRequest(messages=[_user("hello")]), # embedding path
        ]
        await router.decide_batch(reqs)

        assert len(batch_calls) == 1
        assert batch_calls[0] == 1   # only the plain-text request needed embedding

    @pytest.mark.asyncio
    async def test_batch_populates_cache_for_decide(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("unique batch query")])
        await router.decide_batch([req])

        # Subsequent single decide() should be a cache hit
        d = await router.decide(req, trace=True)
        assert d.trace.cache_hit is True

    @pytest.mark.asyncio
    async def test_batch_result_matches_single_decide(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()

        req = InferRequest(messages=[_user("hello world")])
        single = await router.decide(req)
        batch = await router.decide_batch([req])

        assert batch[0].backend    == single.backend
        assert batch[0].model_id   == single.model_id
        assert batch[0].task_class == single.task_class
        assert batch[0].strategy   == single.strategy

    @pytest.mark.asyncio
    async def test_batch_trace_cache_hit_flags(self, basic_config, local_backend, mock_provider):
        router = Router(basic_config, {"loc": local_backend}, mock_provider)
        await router.load_embeddings()

        req_plain = InferRequest(messages=[_user("plain text")])
        req_image = InferRequest(messages=[_image_msg()])

        # First batch — plain text is a miss, image is signal (None)
        results = await router.decide_batch([req_plain, req_image], trace=True)
        assert results[0].trace.cache_hit is False
        assert results[1].trace.cache_hit is None

        # Second batch — plain text is now a hit
        results2 = await router.decide_batch([req_plain], trace=True)
        assert results2[0].trace.cache_hit is True
