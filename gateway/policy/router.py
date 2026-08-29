import asyncio
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from gateway.adapters.base import BaseAdapter, NormalizedResponse
from gateway.policy.circuit_breaker import CircuitBreakerRegistry

logger = logging.getLogger(__name__)

# Domain vocabulary suggesting a request needs real reasoning. Matches are
# counted INDIVIDUALLY, not per-group: grouping them meant a prompt saying
# "optimize and refactor and benchmark" scored the same as one saying
# "optimize" once, which is why such prompts routed to the cheap model.
#
# Coverage is inherently incomplete — these are software, maths, and a
# sampling of professional domains. A hard question in a vocabulary not listed
# here scores zero, which is the structural limit of lexical matching; see
# F7 in docs/AI_ML_FLAWS.md.
DOMAIN_TERMS = re.compile(
    r"\b("
    # software engineering
    r"optimi[sz]e|optimi[sz]ation|benchmark\w*|refactor\w*|profil(?:e|ing)|"
    r"concurren\w+|race\s+condition|deadlock|mutex|idempoten\w+|"
    r"memory\s+leak|segmentation\s+fault|stack\s?trace|traceback|"
    r"algorithm\w*|data\s+structure|dynamic\s+programming|complexity\s+analysis|"
    r"time\s+complexity|space\s+complexity|throughput|latency|scalab\w+|"
    r"cache\s+invalidation|stale\s+reads?|consistency|distributed|sharding|"
    # maths / science
    r"proof|prove|theorem|lemma|derivation|differential|calculus|eigenvalue|"
    r"probabilit\w+|stochastic|regression|variance|entropy|"
    # medicine
    r"titrat\w+|contraindicat\w+|dosage|dosing|prognosis|diagnos\w+|"
    r"pharmacokinetic\w*|comorbid\w*|"
    # law / finance
    r"liabilit\w+|indemnif\w+|jurisdiction|statutor\w+|arbitration|"
    r"amorti[sz]\w+|hedg\w+|derivative\s+contract|valuation|solvency"
    r")\b",
    re.IGNORECASE,
)

CODE_BLOCK = re.compile(r"```[a-zA-Z0-9_-]*\n[\s\S]+?```")

# Framing that signals analysis rather than recall — causal, comparative, or
# conditional reasoning. Deliberately narrower than a bare "why": "why is the
# sky blue" is recall, "why would X happen under Y" is analysis.
ANALYTICAL_FRAMING = re.compile(
    r"\bwhy\s+(would|might|could|does)\b[^.?!]*\b(when|under|if|despite|given|"
    r"while|during|after|before)\b"
    r"|\bwhat\s+causes?\b[^.?!]*\bto\b"
    r"|\b(trade[- ]?offs?|compare|comparison|versus|vs\.?)\b"
    r"|\bshould\s+\w+[^.?!]*\b(or|rather\s+than|instead\s+of|before|after)\b"
    r"|\bunder\s+what\s+(conditions?|circumstances?)\b"
    r"|\bhow\s+(would|might|does)\b[^.?!]*\baffect\b"
    r"|\b(explain|walk\s+me\s+through)\s+(why|how)\b[^.?!]*\b(when|under|if|"
    r"despite|given)\b",
    re.IGNORECASE,
)

SIMPLE_PREFIXES = re.compile(
    r"^(why is|why are|why do|why does|what is|what are|how are|who is|who are|tell me about|explain simply)\b",
    re.IGNORECASE,
)

# Scoring weights and thresholds for _is_complex_request.
_SCORE_CODE_BLOCK = 3
_SCORE_ANALYTICAL = 2
_SCORE_PER_DOMAIN_TERM = 1
_MAX_DOMAIN_TERM_SCORE = 3
_SCORE_LONG_INPUT = 2
_COMPLEXITY_THRESHOLD = 3
# Long input only counts as complex if it is actually substantive. Repeated
# filler reaches any token threshold while containing nothing to reason about,
# so require genuine lexical variety before length earns a score.
_MIN_DISTINCT_WORDS_FOR_LENGTH = 40

# Refusals / non-answers. Anchored to the start of the response: these
# phrases appear mid-answer in perfectly good replies ("I'm sorry to hear
# that. Here is the fix: ...") and matching them anywhere escalated complete
# answers to a pricier backend for the sake of a polite opener.
REFUSAL_PATTERNS = re.compile(
    r"^\W*("
    r"i don'?t know|i do not know|i'?m not sure|i am not sure|"
    r"i cannot|i can'?t|i'?m unable to|i am unable to|"
    r"i don'?t have (enough|access)|as an ai|"
    r"unfortunately[, ]|sorry[, ]|i apologi[sz]e"
    # A handful of common non-English refusals. Not comprehensive — full
    # multilingual coverage needs a classifier — but these are cheap and a
    # model answering in the user's language is the common case.
    r"|je ne sais pas|je ne peux pas|d[ée]sol[ée]"
    r"|no s[ée]\b|no puedo|lo siento"
    r"|ich wei[sß]+ nicht|ich kann nicht|es tut mir leid"
    r"|n[ãa]o sei|n[ãa]o posso|desculpe"
    r"|non lo so|non posso|mi dispiace"
    r")",
    re.IGNORECASE,
)
# A refusal opener only means the answer failed if the response is *mostly*
# that refusal. Beyond this length there is substantive content after it.
MAX_REFUSAL_RESPONSE_CHARS = 120

# Prompts that ask for something substantive, where a one-line reply is
# probably a failure. Without this, a correct terse answer ("Paris") to a
# terse question was escalated — the opposite of the cost goal.
SUBSTANTIVE_REQUEST = re.compile(
    r"\b(explain|describe|elaborate|walk\s+me\s+through|write|draft|compose|"
    # "list" only as an imperative — "a list comprehension" is a noun phrase
    # and asking for one warrants a one-line answer.
    r"list\s+(the|all|out|every|some|each)|enumerate|"
    r"compare|analyz|analys|summari[sz]e|outline|"
    r"step[- ]by[- ]step|in\s+detail|why|how\s+(do|does|would|can|should))\b",
    re.IGNORECASE,
)
# Only an empty reply is a non-answer regardless of what was asked. Any
# higher blanket floor escalates correct terse answers ("Paris", "42"),
# which is the opposite of what cascade is for.
MIN_ADEQUATE_RESPONSE_CHARS = 1
# Applied only when the prompt actually asked for something substantive.
MIN_SUBSTANTIVE_RESPONSE_CHARS = 80


class NoAvailableBackendException(Exception):
    pass


class DeadlineExceededException(Exception):
    """The request's total time budget ran out before any backend answered."""

    pass


# Below this much remaining budget an attempt cannot plausibly finish, so
# starting one only delays the error the caller is already going to receive.
_MIN_ATTEMPT_SEC = 1.0


# A provider saying "too fast" is not a provider that is broken, and the two
# need opposite responses: back off and keep using a throttled backend, stop
# using a failing one. Counting 429s toward the breaker opened it after three
# rate-limit replies and pushed traffic onto a pricier backend — so hitting a
# free tier's limit quietly cost money.
_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "quota exceeded",
)


def is_rate_limit_error(exc: Exception) -> bool:
    """Whether an adapter failure means "slow down" rather than "you're broken".

    Matches on the message because adapters wrap provider errors in
    AdapterException rather than surfacing a status code. Crude, but the
    alternative — treating throttling as breakage — is actively harmful.
    """
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def retry_after_seconds(exc: Exception) -> Optional[float]:
    """The provider's own Retry-After hint, when it sent one.

    Preferring it to a guess is the difference between backing off for exactly
    as long as required and hammering a throttled endpoint.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


class ContextLengthExceededException(Exception):
    """No healthy backend has a context window large enough for this request.

    Distinct from NoAvailableBackendException because it is a client error —
    retrying or failing over cannot help, the input simply does not fit."""

    pass


class Router:
    def __init__(
        self,
        adapters: List[BaseAdapter],
        circuit_registry: CircuitBreakerRegistry,
        strategy: str = "cost_first",
        timeout_sec: float = 30.0,
        min_tokens_for_complex: int = 500,
        latency_recheck_sec: float = 300.0,
        total_deadline_sec: Optional[float] = None,
    ):
        self.adapters = adapters
        self.circuit_registry = circuit_registry
        self.strategy = strategy
        self.timeout_sec = timeout_sec
        # Ceiling on the whole request, across every retry and every failover.
        # `timeout_sec` bounds a single attempt, which bounds nothing useful:
        # 3 attempts against 2 backends plus backoff is 6 minutes before the
        # caller sees anything. Real clients give up long before that and
        # retry, adding load to a backend that is already struggling.
        self.total_deadline_sec = total_deadline_sec or (timeout_sec * 2)
        self.min_tokens_for_complex = min_tokens_for_complex
        # How long a latency measurement stays trusted before the backend is
        # re-probed under latency_first. Without this a single slow sample
        # sidelines a backend permanently.
        self.latency_recheck_sec = latency_recheck_sec
        self.latency_ema: Dict[str, float] = {}
        self._latency_measured_at: Dict[str, float] = {}
        self._rr_counter = 0

    async def close(self) -> None:
        """Close all managed backend adapters to release connection pools and sockets."""
        for adapter in self.adapters:
            try:
                await adapter.close()
            except Exception as e:
                logger.error(
                    f"Error closing adapter {getattr(adapter, 'id', 'unknown')}: {e}"
                )

    def _complexity_score(self, messages: Optional[List[Dict[str, str]]]) -> int:
        """Weighted complexity score for a request.

        Additive scoring rather than the previous rule chain, because the
        signals genuinely compound: a long prompt full of domain terms is
        more likely to need a capable model than either alone. Each signal is
        capped so no single one can dominate.
        """
        if not messages:
            return 0

        contents = [
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        ]
        full_prompt = " ".join(contents).strip()
        if not full_prompt:
            return 0

        score = 0

        if CODE_BLOCK.search(full_prompt):
            score += _SCORE_CODE_BLOCK

        # Count every occurrence, capped — three distinct domain terms is a
        # stronger signal than one, which per-group matching could not express.
        score += min(
            len(DOMAIN_TERMS.findall(full_prompt)) * _SCORE_PER_DOMAIN_TERM,
            _MAX_DOMAIN_TERM_SCORE,
        )

        if ANALYTICAL_FRAMING.search(full_prompt):
            score += _SCORE_ANALYTICAL

        approx_tokens = len(full_prompt) / 4.0
        if approx_tokens >= self.min_tokens_for_complex:
            distinct_words = len({w.lower() for w in re.findall(r"\w+", full_prompt)})
            if distinct_words >= _MIN_DISTINCT_WORDS_FOR_LENGTH:
                score += _SCORE_LONG_INPUT

        # A short, plainly conversational question with no other signal is
        # recall, not analysis — keep it on the cheap backend.
        if score <= _SCORE_PER_DOMAIN_TERM and approx_tokens < 60:
            first_user_msg = next(
                (
                    m.get("content", "").strip()
                    for m in messages
                    if m.get("role") == "user"
                ),
                "",
            )
            if SIMPLE_PREFIXES.search(first_user_msg):
                return 0

        return score

    def _is_complex_request(self, messages: Optional[List[Dict[str, str]]]) -> bool:
        """Whether a request warrants a more capable (pricier) backend."""
        return self._complexity_score(messages) >= _COMPLEXITY_THRESHOLD

    @staticmethod
    def _is_response_inadequate(
        response: NormalizedResponse,
        messages: Optional[List[Dict[str, str]]] = None,
        finish_reason: Optional[str] = None,
    ) -> bool:
        """Whether a completion looks like it failed to answer the request.

        Judged against the prompt, not in isolation: "Paris" is a complete
        answer to a factual question and a failure in response to "explain
        the water cycle in detail". Scoring length alone escalated correct
        terse answers to a pricier backend, which inverts the cost goal.

        This detects *non-answers*, not wrong ones. A fluent, confident,
        entirely incorrect answer passes — see F9 in docs/AI_ML_FLAWS.md.
        """
        text = " ".join(m.content for m in response.messages).strip()

        if not text:
            return True

        # Hard evidence the backend stopped early rather than finished.
        if finish_reason == "length":
            return True

        # A refusal only counts when it is essentially the whole reply.
        if len(text) <= MAX_REFUSAL_RESPONSE_CHARS and REFUSAL_PATTERNS.search(text):
            return True

        if len(text) < MIN_ADEQUATE_RESPONSE_CHARS:
            return True

        # Terse replies are only suspect when the prompt asked for substance.
        prompt = " ".join(
            m.get("content", "") for m in (messages or []) if m.get("content")
        )
        if (
            prompt
            and SUBSTANTIVE_REQUEST.search(prompt)
            and len(text) < MIN_SUBSTANTIVE_RESPONSE_CHARS
        ):
            return True

        return False

    async def get_ranked_adapters(
        self, messages: Optional[List[Dict[str, str]]] = None, **kwargs
    ) -> List[BaseAdapter]:
        """Rank adapters based on the routing strategy and capabilities."""
        candidates = []
        for adapter in self.adapters:
            # Check circuit breaker
            breaker = self.circuit_registry.get_breaker(adapter.id)
            if not await breaker.can_request():
                continue

            candidates.append(adapter)

        if not candidates:
            raise NoAvailableBackendException(
                "No healthy backends available to serve the request."
            )

        # Capability-aware filtering. Callers may pass an explicit
        # `required_capabilities` list; json_mode is also inferred from
        # response_format / json_mode kwargs. Falls back to all candidates
        # when no backend advertises every required capability.
        required = {str(c) for c in (kwargs.get("required_capabilities") or [])}
        response_format = kwargs.get("response_format") or {}
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
        ) or kwargs.get("json_mode", False):
            required.add("json_mode")
        if required:
            capable_candidates = [
                a
                for a in candidates
                if required.issubset(set(a.config.get("capabilities") or []))
            ]
            if capable_candidates:
                candidates = capable_candidates

        # Context-window filtering. Unlike the filters above this is a HARD
        # constraint with no fallback: routing to a backend whose window
        # cannot fit the request guarantees an upstream failure, so keeping
        # such a candidate would only burn the retry/failover budget on a
        # request no backend can serve. Backends that declare no
        # context_length are treated as unconstrained rather than excluded.
        requested_max_tokens = kwargs.get("max_tokens") or 0
        if messages:
            fitting: List[BaseAdapter] = []
            # Report the closest miss, since token counts are per-tokenizer
            # and so differ between backends.
            best_shortfall: Optional[Tuple[int, int, int]] = None
            for adapter in candidates:
                if not adapter.context_length:
                    fitting.append(adapter)
                    continue
                needed = adapter.count_prompt_tokens(messages) + requested_max_tokens
                if needed <= adapter.context_length:
                    fitting.append(adapter)
                    continue
                shortfall = needed - adapter.context_length
                if best_shortfall is None or shortfall < best_shortfall[0]:
                    best_shortfall = (shortfall, needed, adapter.context_length)

            if not fitting:
                shortfall, needed, window = best_shortfall  # type: ignore[misc]
                raise ContextLengthExceededException(
                    f"Request needs ~{needed} tokens (prompt + "
                    f"max_tokens={requested_max_tokens}) but the largest "
                    f"available backend window is {window} — over by "
                    f"{shortfall}."
                )
            candidates = fitting

        # Honor an explicitly requested model when a healthy backend serves
        # it, same filter-then-fallback shape as the capability check above:
        # if nothing matches (wrong name, or that backend is down), silently
        # fall through to strategy-based ranking over all candidates rather
        # than failing the request outright.
        requested_model = kwargs.get("model")
        if requested_model:
            model_candidates = [a for a in candidates if a.model == requested_model]
            if model_candidates:
                candidates = model_candidates

        # Rank candidates
        if self.strategy == "cost_first":
            candidates.sort(key=self._cheapest_first_key)
        elif self.strategy == "cascade":
            # Ordered weakest-and-cheapest first so that escalating to the
            # next candidate in execute() always moves *up* in capability.
            candidates.sort(key=self._cascade_key)
        elif self.strategy == "latency_first":
            candidates.sort(key=self._fastest_first_key)
        elif self.strategy == "round_robin" or self.strategy == "weighted_round_robin":
            if len(candidates) > 1:
                idx = self._rr_counter % len(candidates)
                self._rr_counter += 1
                candidates = candidates[idx:] + candidates[:idx]
        elif self.strategy == "complexity":
            if self._is_complex_request(messages):
                candidates.sort(key=self._most_capable_first_key)
            else:
                candidates.sort(key=self._cheapest_first_key)

        return candidates

    # ── Ranking keys ─────────────────────────────────────────────────────────
    # Capability and price are separate axes. Ranking "the better model" by
    # price assumes they correlate, and they often don't: a small fast model
    # can be priced above a large self-hosted one, in which case escalating by
    # cost escalates to the *weaker* backend. Backends therefore declare
    # `capability_tier`, and cost is used only to break ties or when no
    # backend declares a tier.

    def _any_tier_declared(self) -> bool:
        return any(a.capability_tier for a in self.adapters)

    def _fastest_first_key(self, adapter: BaseAdapter) -> Tuple[float, float]:
        """Fastest measured first, with unmeasured backends tried first.

        Defaulting an unknown backend to a large latency ranked it last
        forever: never selected, so never measured, so never selected — a
        newly added fast backend could be starved indefinitely. Optimism in
        the face of uncertainty costs at most one probe request, after which
        the backend has a real measurement and competes on merit.

        A measurement also goes stale: a backend that was slow once stays
        ranked slow even after recovering. Past `latency_recheck_sec` it is
        treated as unmeasured again so it gets re-probed.
        """
        measured_at = self._latency_measured_at.get(adapter.id)
        if (
            measured_at is None
            or adapter.id not in self.latency_ema
            or (time.time() - measured_at) > self.latency_recheck_sec
        ):
            return (
                0.0,
                adapter.cost_per_1k_prompt,
            )  # sorts ahead of any real measurement
        return (1.0, self.latency_ema[adapter.id])

    def _cheapest_first_key(self, adapter: BaseAdapter) -> Tuple[float, int]:
        """Cheapest first; prefer the more capable backend at equal price."""
        return (adapter.cost_per_1k_prompt, -adapter.capability_tier)

    def _cascade_key(self, adapter: BaseAdapter) -> Tuple[float, float]:
        """Weakest-and-cheapest first, so escalation moves up in capability.

        Falls back to pure cost ordering when no tier is declared, matching
        the previous behaviour.
        """
        if not self._any_tier_declared():
            return (adapter.cost_per_1k_prompt, 0.0)
        return (float(adapter.capability_tier), adapter.cost_per_1k_prompt)

    def _most_capable_first_key(self, adapter: BaseAdapter) -> Tuple[float, float]:
        """Most capable first; cheapest wins within a tier.

        With no tiers declared anywhere this degrades to the previous
        cost-descending behaviour, so existing configs are unaffected.
        """
        if not self._any_tier_declared():
            return (-adapter.cost_per_1k_prompt, 0.0)
        return (-float(adapter.capability_tier), adapter.cost_per_1k_prompt)

    async def execute(self, messages: List[Dict[str, str]], **kwargs):
        started = time.monotonic()
        deadline = started + self.total_deadline_sec
        ranked = await self.get_ranked_adapters(messages=messages, **kwargs)

        last_error = None
        # Cascade escalation discards a cheap attempt's response but not its
        # cost — that call still happened and (against a real paid backend)
        # still cost money, so it must be folded into whatever response is
        # ultimately billed, or spend tracking would silently undercount it.
        cascade_discarded_cost = 0.0
        for rank_idx, adapter in enumerate(ranked):
            breaker = self.circuit_registry.get_breaker(adapter.id)
            # Map request kwargs model to the adapter's native model during failover
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = adapter.model

            max_retries = 2
            for attempt in range(max_retries + 1):
                # An attempt may never outlive the request's own budget, and
                # starting one that cannot finish inside it only delays the
                # error the caller is already going to get.
                remaining = deadline - time.monotonic()
                if remaining <= _MIN_ATTEMPT_SEC:
                    logger.warning(
                        f"Deadline reached after {time.monotonic() - started:.1f}s; "
                        f"not attempting {adapter.id}"
                    )
                    raise DeadlineExceededException(
                        f"Request exceeded its {self.total_deadline_sec:.0f}s budget "
                        f"across {rank_idx + 1} backend(s). "
                        f"Last error: {last_error}"
                    )
                attempt_timeout = min(self.timeout_sec, remaining)
                try:
                    logger.info(
                        f"Routing request to {adapter.id} using model {adapter.model} "
                        f"(attempt {attempt + 1}/{max_retries + 1}, "
                        f"timeout {attempt_timeout:.1f}s, "
                        f"{remaining:.1f}s of budget left)"
                    )
                    response = await asyncio.wait_for(
                        adapter.complete(messages, **call_kwargs),
                        timeout=attempt_timeout,
                    )

                    # Update Exponential Moving Average (EMA) for latency
                    alpha = 0.2
                    current_ema = self.latency_ema.get(adapter.id, response.latency_ms)
                    self.latency_ema[adapter.id] = (alpha * response.latency_ms) + (
                        (1 - alpha) * current_ema
                    )
                    self._latency_measured_at[adapter.id] = time.time()

                    await breaker.record_success()

                    if (
                        self.strategy == "cascade"
                        and rank_idx < len(ranked) - 1
                        and self._is_response_inadequate(response, messages)
                    ):
                        logger.info(
                            f"Cascade: {adapter.id}'s response looked inadequate "
                            f"(too short or a hedge/refusal) — escalating to "
                            f"{ranked[rank_idx + 1].id}"
                        )
                        cascade_discarded_cost += response.cost_usd
                        break  # move to the next, pricier adapter

                    updates: Dict[str, Any] = {}
                    if cascade_discarded_cost:
                        updates["cost_usd"] = response.cost_usd + cascade_discarded_cost
                    if rank_idx > 0:
                        # Answered by something other than the router's first
                        # choice. Availability-wise that is the whole point of
                        # failover, but the caller may have received a much
                        # weaker model than was selected, so mark it rather
                        # than let the substitution pass silently.
                        updates["is_fallback"] = True
                    if updates:
                        response = response.model_copy(update=updates)
                    return response
                except asyncio.TimeoutError:
                    last_error = asyncio.TimeoutError(
                        f"Backend {adapter.id} timed out after {attempt_timeout:.1f}s"
                    )
                    logger.warning(str(last_error))
                    if attempt < max_retries:
                        # Capped at the remaining budget: sleeping longer than
                        # the request has left only postpones its failure.
                        backoff = min(2**attempt, max(0.0, deadline - time.monotonic()))
                        logger.warning(f"Retrying {adapter.id} in {backoff:.1f}s...")
                        await asyncio.sleep(backoff)
                    else:
                        await breaker.record_failure()
                        break  # move to next adapter
                except Exception as e:
                    last_error = e
                    throttled = is_rate_limit_error(e)

                    if attempt < max_retries:
                        # Honour the provider's own Retry-After when it sent
                        # one; guessing is what turns a throttle into an
                        # outage.
                        backoff = retry_after_seconds(e)
                        if backoff is None:
                            backoff = 2**attempt
                        # Never sleep past the deadline — waiting to retry a
                        # request that can no longer finish is pure delay.
                        backoff = min(backoff, max(0.0, deadline - time.monotonic()))
                        logger.warning(
                            f"{'Throttled by' if throttled else 'Transient failure on'} "
                            f"{adapter.id} ({str(e)}). Retrying in {backoff:.1f}s..."
                        )
                        await asyncio.sleep(backoff)
                    else:
                        logger.warning(
                            f"Backend {adapter.id} failed after {max_retries + 1} attempts: {str(e)}"
                        )
                        if throttled:
                            # Being throttled means the backend is healthy and
                            # busy. Opening its breaker would route traffic to
                            # a pricier one for the cooldown, so a free tier's
                            # limit would quietly become a bill.
                            logger.info(
                                f"{adapter.id} is rate limited, not failing — "
                                f"leaving its circuit breaker closed"
                            )
                        else:
                            await breaker.record_failure()
                        break  # move to next adapter

        raise NoAvailableBackendException(
            f"All capable backends failed. Last error: {str(last_error)}"
        )

    async def execute_stream(
        self,
        messages: List[Dict[str, str]],
        backend_info: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """`backend_info`, if given, is populated with the id of the adapter
        that ends up serving the stream once one is selected. Callers that
        need to attribute cost/telemetry to a backend (which chunk payloads
        don't carry, to keep them a clean pass-through of the upstream SSE
        format) should read it back out after/while iterating."""
        ranked = await self.get_ranked_adapters(messages=messages, **kwargs)

        last_error = None
        target_adapter = None
        stream_generator = None
        stream_start = time.time()

        for adapter in ranked:
            breaker = self.circuit_registry.get_breaker(adapter.id)
            call_kwargs = dict(kwargs)
            call_kwargs["model"] = adapter.model

            try:
                logger.info(
                    f"Opening stream routing to {adapter.id} using model {adapter.model} "
                    f"(timeout {self.timeout_sec}s)"
                )
                gen = adapter.complete_stream(messages, **call_kwargs)
                # Enforce timeout on the first chunk so a hung backend doesn't
                # block indefinitely before we can fail over.
                first_chunk = await asyncio.wait_for(
                    gen.__anext__(),
                    timeout=self.timeout_sec,
                )

                # Stream successfully initialized
                target_adapter = adapter
                stream_generator = gen
                if backend_info is not None:
                    backend_info["backend_id"] = adapter.id
                yield first_chunk
                break
            except (asyncio.TimeoutError, StopAsyncIteration, Exception) as e:
                last_error = e
                logger.warning(
                    f"Backend stream startup on {adapter.id} failed: {str(e)}"
                )
                await breaker.record_failure()
                continue

        if not target_adapter or not stream_generator:
            raise NoAvailableBackendException(
                f"All backends failed to initialize stream. Last error: {str(last_error)}"
            )

        accumulated_text = ""
        last_chunk_id = "stream-chunk"
        last_model = target_adapter.model

        try:
            async for chunk in stream_generator:
                # Enforce the remaining deadline on each chunk read.
                elapsed = time.time() - stream_start
                remaining = self.timeout_sec - elapsed
                if remaining <= 0:
                    raise asyncio.TimeoutError(
                        f"Stream timed out after {self.timeout_sec}s on {target_adapter.id}"
                    )

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        accumulated_text += content
                if chunk.get("id"):
                    last_chunk_id = chunk["id"]
                yield chunk

            breaker = self.circuit_registry.get_breaker(target_adapter.id)
            await breaker.record_success()
        except asyncio.TimeoutError as e:
            logger.error(f"Stream timed out on {target_adapter.id}: {str(e)}")
            await self.circuit_registry.get_breaker(target_adapter.id).record_failure()
            raise
        except Exception as e:
            logger.error(
                f"Stream interrupted on {target_adapter.id} after initial chunks: {str(e)}"
            )
            raise
        finally:
            latency_ms = (time.time() - stream_start) * 1000.0

            try:
                import tiktoken

                encoding = tiktoken.get_encoding("cl100k_base")
                prompt_tokens = sum(
                    len(encoding.encode(m.get("content", ""))) for m in messages
                )
                completion_tokens = len(encoding.encode(accumulated_text))
            except Exception:
                approx_p = sum(len(m.get("content", "")) for m in messages) / 4.0
                prompt_tokens = max(1, int(approx_p))
                completion_tokens = max(1, int(len(accumulated_text) / 4.0))

            cost_usd = (prompt_tokens / 1000.0) * target_adapter.cost_per_1k_prompt + (
                completion_tokens / 1000.0
            ) * target_adapter.cost_per_1k_completion

            # ── Spend recording intentionally omitted here ───────────────────
            # Recording is handled by the caller (main.py stream_generator)
            # which has access to the correct gateway request_id, the
            # AsyncLedgerQueue, and the pre-reservation amount. Doing it
            # here would use a backend chunk-id as the req_id, bypass the
            # queue, and double-count the reserved budget.
