"""Coup language model: tokenisation, loading and candidate scoring."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_CHECKPOINT = str(Path(__file__).resolve().parents[1] / "checkpoints" / "coup-lm")

_CACHE: dict[tuple[str, str], "CoupLM"] = {}


def encode_lines(tokenizer, lines):
	"""Tokenise each line as line + newline separately and concatenate.

	Returns (ids, spans) where spans[i] = (start, end) token range of line i.
	"""
	ids: list[int] = []
	spans: list[tuple[int, int]] = []
	for line in lines:
		toks = tokenizer(line + "\n", add_special_tokens=False)["input_ids"]
		spans.append((len(ids), len(ids) + len(toks)))
		ids.extend(toks)
	return ids, spans


def _structure(lines):
	"""Return (history_starts, body_start): indices of `past` lines and of the
	first line after the `n .. res ..` line (0 if no such line)."""
	starts = [i for i, l in enumerate(lines) if l.startswith("past ")]
	body_start = 0
	for i, line in enumerate(lines):
		if line.startswith("n ") and " res " in line:
			body_start = i + 1
			break
	return starts, body_start


class ScoringSession:
	"""Cached prompt token ids + batch-1 KV cache for incremental `CoupLM.score` calls
	within one game (one session per agent)."""

	def __init__(self):
		self.ids: list[int] = []
		self.cache = None
		self.last_lp = None  # log-probs for the token following `ids`

	def reset(self) -> None:
		self.ids, self.cache, self.last_lp = [], None, None


class CoupLM:
	def __init__(self, model, tokenizer, device, max_len: int = 3072):  # match train.py --max-len
		self.model = model.eval()
		self.tokenizer = tokenizer
		self.device = device
		self.max_len = max_len
		self.empty_cache_every = 25
		self._calls = 0

	@classmethod
	def load(cls, path: str | None = None, device: str | None = None) -> "CoupLM":
		path = str(path) if path else DEFAULT_CHECKPOINT
		if not Path(path).exists():
			print(
				f"[CoupLM] checkpoint {path!r} not found; using base model {DEFAULT_BASE!r}.",
				file=sys.stderr,
			)
			path = DEFAULT_BASE
		device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		key = (path, device)
		if key in _CACHE:
			return _CACHE[key]
		dtype = torch.float32
		if device.startswith("cuda"):
			dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
		tokenizer = AutoTokenizer.from_pretrained(path)
		model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)
		lm = cls(model, tokenizer, device)
		_CACHE[key] = lm
		return lm

	def _prefix_ids(self, prefix_lines, room: int) -> list[int]:
		"""Tokenise the prompt, dropping oldest history games first, then oldest body lines."""
		lines = list(prefix_lines)
		budget = self.max_len - room
		lens = [len(self.tokenizer(l + "\n", add_special_tokens=False)["input_ids"]) for l in lines]
		keep = [True] * len(lines)
		starts, body_start = _structure(lines)
		total = sum(lens)
		if starts:
			# history games: from each `past` line up to the next one (or the `now` line)
			hist_end = next((i for i, l in enumerate(lines) if l == "now"), starts[-1] + 1)
			bounds = starts + [hist_end]
			for g in range(len(starts)):
				if total <= budget:
					break
				for i in range(bounds[g], bounds[g + 1]):
					keep[i] = False
					total -= lens[i]
		i = body_start
		while total > budget and i < len(lines):
			if keep[i]:
				keep[i] = False
				total -= lens[i]
			i += 1
		ids, _ = encode_lines(self.tokenizer, [l for l, k in zip(lines, keep) if k])
		return ids

	def score_tensor(self, prefix_lines, candidate_lines) -> torch.Tensor:
		"""Differentiable counterpart of `score`: float32 tensor of shape (n,) with the
		summed log-prob of each candidate. Same tokenisation/truncation; the caller
		controls train/eval mode and autocast. Runs prefix+candidate as one padded batch."""
		cands = [
			self.tokenizer(c + "\n", add_special_tokens=False)["input_ids"]
			for c in candidate_lines
		]
		if not cands:
			return torch.zeros(0, device=self.device)
		longest = max(len(c) for c in cands)
		prefix = self._prefix_ids(prefix_lines, longest)
		if not prefix:
			bos = self.tokenizer.bos_token_id
			if bos is None:
				bos = self.tokenizer.eos_token_id
			prefix = [bos]
		n, plen = len(cands), len(prefix)
		pad = self.tokenizer.eos_token_id or 0
		inp = torch.full((n, plen + longest - 1), pad, dtype=torch.long)
		tgt = torch.zeros((n, longest), dtype=torch.long)
		mask = torch.zeros((n, longest))
		inp[:, :plen] = torch.tensor(prefix)
		for i, c in enumerate(cands):
			if len(c) > 1:
				inp[i, plen : plen + len(c) - 1] = torch.tensor(c[:-1])
			tgt[i, : len(c)] = torch.tensor(c)
			mask[i, : len(c)] = 1.0
		inp, tgt, mask = inp.to(self.device), tgt.to(self.device), mask.to(self.device)
		# right padding + causal attention: padded positions never influence real ones.
		logits = self.model(input_ids=inp, use_cache=False, logits_to_keep=longest).logits
		lp = F.log_softmax(logits.float(), dim=-1)
		return (lp.gather(2, tgt[:, :, None])[:, :, 0] * mask).sum(dim=1)

	def new_session(self) -> "ScoringSession":
		"""A per-caller incremental KV cache for `score` (never share between agents)."""
		return ScoringSession()

	def _bos_prefix(self, prefix: list[int]) -> list[int]:
		if prefix:
			return prefix
		bos = self.tokenizer.bos_token_id
		if bos is None:
			bos = self.tokenizer.eos_token_id
		return [bos]

	def _prefix_state(self, prefix: list[int], session):
		"""Run the prefix, return (cache, last-position log-probs). With a session, only
		the tokens beyond the cached ones are run; the returned cache is the session's own
		(batch 1) and must not be mutated by the caller."""
		old = session.ids if session is not None else None
		if old and len(prefix) >= len(old) and prefix[: len(old)] == old:
			if len(prefix) == len(old):
				return session.cache, session.last_lp
			x = torch.tensor([prefix[len(old):]], device=self.device)
			# cache_position (hence position ids) continues from the cached length
			out = self.model(input_ids=x, past_key_values=session.cache, use_cache=True)
		else:
			x = torch.tensor([prefix], device=self.device)
			out = self.model(input_ids=x, use_cache=True)
		last_lp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
		if session is not None:
			session.ids, session.cache, session.last_lp = list(prefix), out.past_key_values, last_lp
		return out.past_key_values, last_lp

	def _attn_mask(self, allowed):
		"""[1,1,Q,K] mask from a bool `allowed` matrix: additive float for eager attention,
		boolean otherwise (dtype-safe under autocast, where q is bf16 but weights fp32)."""
		if getattr(self.model.config, "_attn_implementation", "sdpa") == "eager":
			dtype = self.model.get_input_embeddings().weight.dtype
			allowed = torch.zeros(allowed.shape, dtype=dtype).masked_fill_(
				~allowed, torch.finfo(dtype).min)
		return allowed[None, None].to(self.device)

	def _score_tree(self, cache, p: int, cands):
		"""Rest-of-candidate log-probs in ONE batch-1 forward appended after the cached prefix
		(length p): all candidates' tokens concatenated, positions restarting at p for each,
		a tree mask so a token sees the whole prefix + earlier tokens of its own candidate.
		The session cache is never mutated: the pass runs on a shallow cache whose layers
		share the original tensors (DynamicCache.update rebinds, it does not write in place)."""
		count = sum(len(c) - 1 for c in cands)
		ids = torch.zeros(count, dtype=torch.long)
		pos = torch.zeros(count, dtype=torch.long)
		tgt = torch.zeros(count, dtype=torch.long)
		seg = torch.zeros(count, dtype=torch.long)
		allowed = torch.zeros(count, p + count, dtype=torch.bool)
		allowed[:, :p] = True
		cur = 0
		for i, c in enumerate(cands):
			m = len(c) - 1
			if m:
				ids[cur: cur + m] = torch.tensor(c[:-1])
				tgt[cur: cur + m] = torch.tensor(c[1:])
				pos[cur: cur + m] = torch.arange(p, p + m)
				seg[cur: cur + m] = i
				allowed[cur: cur + m, p + cur: p + cur + m] = torch.tril(
					torch.ones(m, m, dtype=torch.bool))
				cur += m
		work = copy.copy(cache)
		work.layers = [copy.copy(layer) for layer in cache.layers]
		dev = self.device
		o = self.model(input_ids=ids[None].to(dev), position_ids=pos[None].to(dev),
			attention_mask=self._attn_mask(allowed), past_key_values=work, use_cache=True)
		lp = F.log_softmax(o.logits[0].float(), dim=-1)
		tok = lp.gather(1, tgt.to(dev)[:, None])[:, 0]
		return torch.zeros(len(cands), device=dev).index_add(0, seg.to(dev), tok).tolist()

	@torch.no_grad()
	def score(self, prefix_lines, candidate_lines, session=None) -> list[float]:
		cands = [
			self.tokenizer(c + "\n", add_special_tokens=False)["input_ids"]
			for c in candidate_lines
		]
		if not cands:
			return []
		longest = max(len(c) for c in cands)
		prefix = self._bos_prefix(self._prefix_ids(prefix_lines, longest))
		cache, first_lp = self._prefix_state(prefix, session)
		firsts = [first_lp[c[0]].item() for c in cands]
		if longest == 1:
			self._release_cache()
			return firsts
		if session is not None:
			# no per-candidate copy of the prefix KV (memory): one tree-masked pass
			rest = self._score_tree(cache, len(prefix), cands)
		else:
			# One batched pass over all candidates, reusing the prefix cache.
			# Candidates are right-padded; causal attention keeps real tokens unaffected.
			n = len(cands)
			pad = self.tokenizer.eos_token_id or 0
			inp = torch.full((n, longest - 1), pad, dtype=torch.long)
			tgt = torch.zeros((n, longest - 1), dtype=torch.long)
			mask = torch.zeros((n, longest - 1))
			for i, c in enumerate(cands):
				if len(c) > 1:
					inp[i, : len(c) - 1] = torch.tensor(c[:-1])
					tgt[i, : len(c) - 1] = torch.tensor(c[1:])
					mask[i, : len(c) - 1] = 1.0
			inp, tgt, mask = inp.to(self.device), tgt.to(self.device), mask.to(self.device)
			cache.batch_repeat_interleave(n)  # throwaway cache: in place is fine
			o = self.model(input_ids=inp, past_key_values=cache, use_cache=True)
			lp = F.log_softmax(o.logits.float(), dim=-1)
			rest = (lp.gather(2, tgt[:, :, None])[:, :, 0] * mask).sum(dim=1).tolist()
			del o, lp
		del cache
		self._release_cache()
		return [f + r for f, r in zip(firsts, rest)]

	def score_packed(self, items, chunk: int = 512) -> list[torch.Tensor]:
		"""Differentiable `score_tensor` for many (prefix_lines, candidate_lines) decisions
		at once. Consecutive decisions whose prefix tokens extend the previous one share one
		packed sequence [longest prefix | candidate tokens of every decision] with a custom
		4D mask: a candidate token of decision t sees prefix positions < p_t and earlier
		tokens of its own candidate only. Packs are split to stay within max_len; any
		decision whose prefix is not an extension gets its own pack (= per-decision
		scoring). Returns one (n_i,) tensor per item, in order."""
		prepared = []
		for prefix_lines, candidate_lines in items:
			cands = [self.tokenizer(c + "\n", add_special_tokens=False)["input_ids"]
				for c in candidate_lines]
			if not cands:
				prepared.append(None)
				continue
			prefix = self._bos_prefix(self._prefix_ids(prefix_lines, max(len(c) for c in cands)))
			prepared.append((prefix, cands))
		results: list = [None] * len(items)
		pack: list[int] = []  # item indices
		base: list[int] = []
		extra = 0

		def flush():
			if pack:
				for i, t in zip(pack, self._run_pack(base, [prepared[i] for i in pack], chunk)):
					results[i] = t
				pack.clear()

		for i, entry in enumerate(prepared):
			if entry is None:
				results[i] = torch.zeros(0, device=self.device)
				continue
			prefix, cands = entry
			cost = sum(len(c) - 1 for c in cands)
			if pack and prefix[: len(base)] == base and len(prefix) + extra + cost <= self.max_len:
				base, extra = prefix, extra + cost
			else:
				flush()
				base, extra = prefix, cost
			pack.append(i)
		flush()
		return results

	def _run_pack(self, base, entries, chunk):
		from torch.utils.checkpoint import checkpoint
		B = len(base)
		L = B + sum(len(c) - 1 for _, cands in entries for c in cands)
		ids = torch.zeros(L, dtype=torch.long)
		pos = torch.arange(L)
		ids[:B] = torch.tensor(base)
		allowed = torch.zeros(L, L, dtype=torch.bool)
		allowed[:B, :B] = torch.tril(torch.ones(B, B, dtype=torch.bool))
		# per target token: hidden-state row, target id, segment (flat candidate index)
		rows, tgts, segs = [], [], []
		seg_counts, cur, seg = [], B, 0
		for prefix, cands in entries:
			p = len(prefix)
			seg_counts.append(len(cands))
			for c in cands:
				rows.append(p - 1)
				tgts.append(c[0])
				segs.append(seg)
				m = len(c) - 1
				if m:
					ids[cur: cur + m] = torch.tensor(c[:-1])
					pos[cur: cur + m] = torch.arange(p, p + m)
					allowed[cur: cur + m, :p] = True
					allowed[cur: cur + m, cur: cur + m] = torch.tril(torch.ones(m, m, dtype=torch.bool))
					rows.extend(range(cur, cur + m))
					tgts.extend(c[1:])
					segs.extend([seg] * m)
					cur += m
				seg += 1
		dev = self.device
		mask = self._attn_mask(allowed)
		h = self.model.base_model(input_ids=ids[None].to(dev), position_ids=pos[None].to(dev),
			attention_mask=mask, use_cache=False).last_hidden_state[0]
		head = self.model.get_output_embeddings()
		rows_t, tgts_t, segs_t = (torch.tensor(x, device=dev) for x in (rows, tgts, segs))

		def tok_lp(hs, tg):
			lp = F.log_softmax(head(hs).float(), dim=-1)
			return lp.gather(1, tg[:, None])[:, 0]

		parts = []
		for s in range(0, len(rows), chunk):
			hs, tg = h[rows_t[s: s + chunk]], tgts_t[s: s + chunk]
			if torch.is_grad_enabled() and hs.requires_grad:
				parts.append(checkpoint(tok_lp, hs, tg, use_reentrant=False))
			else:
				parts.append(tok_lp(hs, tg))
		tok = torch.cat(parts)
		flat = torch.zeros(seg, device=dev, dtype=tok.dtype).index_add(0, segs_t, tok)
		return list(torch.split(flat, seg_counts))

	def _release_cache(self) -> None:
		# Prompts vary in length every decision, which fragments the CUDA caching allocator;
		# on Windows (WDDM) the overflow spills into shared system RAM and everything crawls.
		self._calls += 1
		if self.device.startswith("cuda") and self._calls % self.empty_cache_every == 0:
			torch.cuda.empty_cache()
