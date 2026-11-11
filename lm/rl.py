"""REINFORCE fine-tuning of the Coup LM against scripted bots, with a KL penalty to a
frozen reference (normally the behaviour-cloned checkpoint).

Policy = the agent's: softmax over the LEGAL candidate lines of their summed log-prob.
Usage: python -m lm.rl --init checkpoints/coup-lm --ref checkpoints/coup-lm \
		--out checkpoints/coup-lm-rl --updates 200
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, field

from agents.llm_agent import LLMAgent
from lm.evaluate import TEACHERS, play
from lm.generate_data import make_player
from src.game import Game


# ----------------------------------------------------------------------------
# Pieces that need no model / no torch
# ----------------------------------------------------------------------------
def compute_reward(won: bool, draw: bool, survivors: int) -> float:
	"""Return for the learning agent. EDIT HERE to change the reward.

	Win = 1, loss = 0; a turn-cap draw gives 1/survivors if the agent is still alive
	(`survivors` counts the living players, 0 if the agent is dead -> pass its own flag)."""
	if won:
		return 1.0
	if draw and survivors > 0:
		return 1.0 / survivors
	return 0.0


class Baseline:
	"""Running mean of the return per player count (EMA), initialised to 1/n."""

	def __init__(self, decay: float = 0.05):
		self.decay = decay
		self.value: dict[int, float] = {}

	def get(self, n: int) -> float:
		return self.value.get(n, 1.0 / n)

	def advantage(self, n: int, ret: float) -> float:
		"""Advantage against the current baseline, then move the baseline towards ret."""
		b = self.get(n)
		self.value[n] = b + self.decay * (ret - b)
		return ret - b

	def state_dict(self) -> dict:
		return {"decay": self.decay, "value": dict(self.value)}

	def load_state_dict(self, state: dict) -> None:
		self.decay = state["decay"]
		self.value = {int(k): v for k, v in state["value"].items()}


@dataclass
class Decision:
	prefix: list
	candidates: list
	chosen: int
	ref_logp: list  # reference log-softmax over the candidates
	advantage: float = 0.0


@dataclass
class GameResult:
	n: int
	ret: float
	won: bool
	draw: bool
	decisions: list = field(default_factory=list)
	info: dict | None = None  # league mode: opponent types / members at the table


def log_softmax_list(scores: list[float]) -> list[float]:
	top = max(scores)
	z = top + math.log(sum(math.exp(s - top) for s in scores))
	return [s - z for s in scores]


class SessionSource:
	"""Endless stream of games: one persistent learner per session (so it builds
	cross-game history) against a pool of bots with stable ids, like generate_data."""

	def __init__(self, rng, make_learner, games_per_session=10, min_pool=3, max_pool=7,
		min_players=2, max_players=6, teachers=None):
		self.rng = rng
		self.make_learner = make_learner
		self.games_per_session = games_per_session
		self.min_pool, self.max_pool = min_pool, max_pool
		self.min_players, self.max_players = min_players, max_players
		self.teachers = teachers or TEACHERS
		self._left = 0
		self.learner = None
		self.pool = []

	def _new_session(self):
		rng = self.rng
		self.pool = []
		for i in range(rng.randint(self.min_pool, self.max_pool)):
			cls = rng.choice(self.teachers)
			self.pool.append((cls, f"{cls.__name__} {i}"))
		self.learner = self.make_learner()
		self._left = self.games_per_session

	def next_game(self):
		"""Return (learner, players) for the next game."""
		if self._left <= 0:
			self._new_session()
		self._left -= 1
		rng = self.rng
		k = rng.randint(self.min_players, min(self.max_players, len(self.pool) + 1))
		chosen = rng.sample(self.pool, k - 1)
		players = [make_player(cls, name, name, rng.randrange(2**31)) for cls, name in chosen]
		players.append(self.learner)
		rng.shuffle(players)
		random.seed(rng.randrange(2**31))  # the engine shuffles with the global RNG
		return self.learner, players


def play_one_game(source: SessionSource, ref_logp_fn) -> GameResult:
	"""Play one game; record every real decision of the learner.

	ref_logp_fn(prefix, candidates) -> list of reference log-probs (log-softmax)."""
	learner, players = source.next_game()
	decisions: list[Decision] = []

	def on_decision(prefix, candidates, chosen):
		decisions.append(Decision(prefix, candidates, chosen, ref_logp_fn(prefix, candidates)))

	learner.on_decision = on_decision
	with contextlib.redirect_stdout(io.StringIO()):
		game = Game(players)
		draw = play(game)
	learner.on_decision = None
	info = getattr(source, "game_info", None)
	if hasattr(source, "after_game"):
		source.after_game()
	n = len(players)
	won = (not draw) and game.winner is learner
	survivors = len(game.alive_players) if learner in game.alive_players else 0
	ret = compute_reward(won, draw, survivors)
	return GameResult(n, ret, won, draw, decisions, info)


def assign_advantages(results: list[GameResult], baseline: Baseline) -> list[Decision]:
	"""Advantage per game from the baseline (updated game by game); flatten decisions."""
	flat = []
	for r in results:
		a = baseline.advantage(r.n, r.ret)
		for d in r.decisions:
			d.advantage = a
			flat.append(d)
	return flat


# ----------------------------------------------------------------------------
# Model-dependent parts
# ----------------------------------------------------------------------------
class AutocastScorer:
	"""Scorer for rollouts: fp32 master weights evaluated under bf16/fp16 autocast."""

	def __init__(self, lm, dtype):
		self.lm, self.dtype = lm, dtype

	def new_session(self):
		return self.lm.new_session()

	def score(self, prefix_lines, candidate_lines, session=None):
		import torch
		with torch.autocast("cuda", dtype=self.dtype, enabled=self.lm.device.startswith("cuda")):
			return self.lm.score(prefix_lines, candidate_lines, session=session)


def rss_mb() -> float | None:
	try:
		import psutil
		return psutil.Process().memory_info().rss / 2**20
	except Exception:
		return None


def save_checkpoint(path, lm, optimizer=None, state=None):
	import torch
	tmp = path + ".tmp"
	shutil.rmtree(tmp, ignore_errors=True)
	os.makedirs(tmp)
	lm.model.save_pretrained(tmp)
	lm.tokenizer.save_pretrained(tmp)
	if optimizer is not None:
		torch.save({"optimizer": optimizer.state_dict(), **(state or {})},
			os.path.join(tmp, "rl_state.pt"))
	shutil.rmtree(path, ignore_errors=True)
	os.replace(tmp, path)


def policy_update(lm, decisions, optimizer, args, scaler, amp_dtype):
	"""One optimisation step over `decisions`. Returns dict of stats (or None if skipped)."""
	import torch
	import torch.nn.functional as F

	if not decisions or all(abs(d.advantage) < 1e-9 for d in decisions):
		return None  # nothing to learn from (e.g. every advantage is 0)
	total = len(decisions)
	lm.model.train()
	optimizer.zero_grad(set_to_none=True)
	kl_sum = ent_sum = pg_sum = 0.0
	use_amp = lm.device.startswith("cuda")
	pack = getattr(args, "pack", True)
	chunks = [decisions[i:i + args.micro_batch] for i in range(0, total, args.micro_batch)]
	if pack:
		# a chunk is a run of whole decisions; packing inside it shares each game's prefix
		# (larger micro-batches = more sharing, more activation memory)
		chunks = [decisions[i:i + args.pack_decisions] for i in range(0, total, args.pack_decisions)]
	for group in chunks:
		loss = 0.0
		with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
			if pack:
				all_scores = lm.score_packed([(d.prefix, d.candidates) for d in group])
			else:
				all_scores = [lm.score_tensor(d.prefix, d.candidates) for d in group]
		for d, scores in zip(group, all_scores):
			logp = F.log_softmax(scores.float(), dim=0)
			ref = torch.tensor(d.ref_logp, device=logp.device, dtype=torch.float32)
			p = logp.exp()
			kl = (p * (logp - ref)).sum()
			pg = -d.advantage * logp[d.chosen]
			loss = loss + (pg + args.beta * kl) / total
			kl_sum += kl.item()
			ent_sum += -(p * logp).sum().item()
			pg_sum += pg.item()
		if scaler is not None:
			scaler.scale(loss).backward()
		else:
			loss.backward()
	if scaler is not None:
		scaler.unscale_(optimizer)
	gnorm = torch.nn.utils.clip_grad_norm_(lm.model.parameters(), args.grad_clip)
	if scaler is not None:
		scaler.step(optimizer)
		scaler.update()
	else:
		optimizer.step()
	lm.model.eval()
	if use_amp:
		# variable-length prompts fragment the CUDA cache; on Windows the overflow
		# spills into system RAM and everything slows to a crawl
		torch.cuda.empty_cache()
	return {"kl": kl_sum / total, "entropy": ent_sum / total, "pg_loss": pg_sum / total,
		"grad_norm": float(gnorm)}


def league_step(league, results, policy, update, args) -> str:
	"""Record the batch's results into the league stats, snapshot if due, return a log fragment."""
	from lm import league as lg
	per = {k: [0, 0] for k in lg.TYPE_KEYS}
	for r in results:
		league.record_game(r.info, r.won)
		types = r.info["types"] if r.info else set()
		keys = [lg.BOTS_ONLY] if not types else [
			k for t, k in (("bc", lg.WITH_BC), ("snap", lg.WITH_SNAP), ("self", lg.WITH_SELF)) if t in types]
		for k in keys:
			per[k][0] += int(r.won)
			per[k][1] += 1
	msg = "league[" + " ".join(
		f"{k}:{w / n:.2f}({n})" if n else f"{k}:-" for k, (w, n) in per.items()) + "] "
	if args.snapshot_every and update % args.snapshot_every == 0:
		snap = lg.snapshot_lm(policy)
		path = os.path.join(args.out, "snapshots", f"u{update:04d}")
		os.makedirs(os.path.dirname(path), exist_ok=True)
		save_checkpoint(path, snap)
		dropped = league.add_snapshot(update, snap, path)
		msg += f"+snap u{update:04d}" + "".join(f" -{m.id}" for m in dropped) + " "
	weights = " ".join(f"{m.id}:{w:.2f}" for m, w in zip(league.members, league.weights()))
	return msg + f"pool {len(league.members)} pfsp[{weights}] "


def main(argv=None):
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument("--init", default="checkpoints/coup-lm")
	p.add_argument("--ref", default=None, help="frozen reference model (default: --init)")
	p.add_argument("--out", default="checkpoints/coup-lm-rl")
	p.add_argument("--updates", type=int, default=200)
	p.add_argument("--games-per-update", type=int, default=16)
	p.add_argument("--lr", type=float, default=1e-5)
	p.add_argument("--beta", type=float, default=0.05, help="KL penalty weight")
	p.add_argument("--baseline-decay", type=float, default=0.05)
	p.add_argument("--grad-clip", type=float, default=1.0)
	p.add_argument("--micro-batch", type=int, default=4, help="decisions per backward chunk")
	p.add_argument("--no-pack", dest="pack", action="store_false",
		help="score each decision separately (old path) instead of packing per game")
	p.add_argument("--pack-decisions", type=int, default=64,
		help="decisions per backward chunk when packing (one forward per game prefix inside)")
	p.add_argument("--history-max-lines", type=int, default=200)
	p.add_argument("--games-per-session", type=int, default=10)
	p.add_argument("--min-players", type=int, default=2)
	p.add_argument("--max-players", type=int, default=6)
	p.add_argument("--save-every", type=int, default=20)
	p.add_argument("--resume", action="store_true", help="continue from <out>/last")
	p.add_argument("--seed", type=int, default=0)
	g = p.add_argument_group("league self-play (opt-in)")
	g.add_argument("--league", action="store_true",
		help="face bots, the frozen BC ref, frozen snapshots of the learner and the live policy")
	g.add_argument("--p-bots", type=float, default=0.4, help="fraction of opponent identities that are scripted bots")
	g.add_argument("--p-self", type=float, default=0.2, help="fraction of LLM opponent identities that are current-self")
	g.add_argument("--snapshot-every", type=int, default=10, help="updates between snapshots")
	g.add_argument("--max-snapshots", type=int, default=5, help="snapshots kept in GPU memory")
	g.add_argument("--pfsp-decay", type=float, default=0.1, help="EMA rate of per-member learner win rates")
	g.add_argument("--pfsp-power", type=float, default=2.0, help="weight = (1-winrate)^power + 0.05")
	args = p.parse_args(argv)
	args.ref = args.ref or args.init

	import torch
	from transformers import AutoModelForCausalLM, AutoTokenizer
	from lm.model import DEFAULT_BASE, CoupLM

	device = "cuda" if torch.cuda.is_available() else "cpu"
	amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
	last = os.path.join(args.out, "last")
	resume = args.resume and os.path.isdir(last)
	if args.resume and not resume:
		print(f"[rl] --resume given but {last} not found; starting fresh")
	init = last if resume else args.init
	if not os.path.exists(init):
		print(f"[rl] {init!r} not found; using base model {DEFAULT_BASE!r}")
		init = DEFAULT_BASE

	# policy: fp32 master weights (bf16 weights would swallow 1e-5-sized updates)
	tok = AutoTokenizer.from_pretrained(init)
	model = AutoModelForCausalLM.from_pretrained(init, dtype=torch.float32).to(device)
	model.gradient_checkpointing_enable()
	policy = CoupLM(model, tok, device)
	ref = CoupLM.load(args.ref)  # frozen, bf16
	ref.model.requires_grad_(False)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
	scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 and device == "cuda" else None
	baseline = Baseline(args.baseline_decay)
	start_update, win_ema = 0, None
	rng = random.Random(args.seed)
	if resume:
		state = torch.load(os.path.join(last, "rl_state.pt"), weights_only=False)
		optimizer.load_state_dict(state["optimizer"])
		baseline.load_state_dict(state["baseline"])
		start_update, win_ema = state["update"], state["win_ema"]
		rng.seed(args.seed + start_update)  # fresh, but deterministic, game stream
		print(f"[rl] resumed from {last} at update {start_update}")

	rollout_scorer = AutocastScorer(policy, amp_dtype)

	def ref_logp(prefix, candidates):
		return log_softmax_list(ref.score(prefix, candidates))

	def make_learner():
		return LLMAgent("LLM", name="LLM", seed=rng.randrange(2**31), scorer=rollout_scorer,
			temperature=1.0, history_max_lines=args.history_max_lines)

	league = None
	if args.league:
		from lm import league as lg
		league = lg.League(args.max_snapshots, args.pfsp_decay, args.pfsp_power,
			make_scorer=lambda lm: AutocastScorer(lm, amp_dtype))
		league.add_bc(ref)  # the already-loaded reference: no extra memory
		snap_root = os.path.join(args.out, "snapshots")
		if resume:
			saved = state.get("league")
			if saved is None and os.path.exists(os.path.join(args.out, "league.json")):
				with open(os.path.join(args.out, "league.json")) as f:
					saved = json.load(f)
			if saved:
				def load_snap(path, upd):
					if not os.path.isdir(path):
						path = os.path.join(snap_root, os.path.basename(path))
					return lg.load_snapshot_lm(path, device)
				league.load_state_dict(saved, load_snap)
				print(f"[league] restored pool: {[m.id for m in league.members]}")
		source = lg.LeagueSessionSource(rng, make_learner, league, rollout_scorer,
			p_bots=args.p_bots, p_self=args.p_self, history_max_lines=args.history_max_lines,
			games_per_session=args.games_per_session,
			min_players=args.min_players, max_players=args.max_players)
	else:
		source = SessionSource(rng, make_learner, args.games_per_session,
			min_players=args.min_players, max_players=args.max_players)

	def checkpoint(path, update):
		extra = {"league": league.state_dict()} if league else {}
		save_checkpoint(path, policy, optimizer, {
			"baseline": baseline.state_dict(), "update": update, "win_ema": win_ema,
			"args": vars(args), **extra})
		if league:
			league.save_json(os.path.join(args.out, "league.json"), {"update": update})

	for update in range(start_update, args.updates):
		t0 = time.time()
		policy.model.eval()
		with torch.no_grad():
			results = [play_one_game(source, ref_logp) for _ in range(args.games_per_update)]
		t_roll = time.time() - t0
		batch_win = sum(r.won for r in results) / len(results)
		win_ema = batch_win if win_ema is None else 0.9 * win_ema + 0.1 * batch_win
		decisions = assign_advantages(results, baseline)
		mean_adv = sum(abs(d.advantage) for d in decisions) / max(len(decisions), 1)
		stats = policy_update(policy, decisions, optimizer, args, scaler, amp_dtype)
		league_msg = ""
		if league:
			league_msg = league_step(league, results, policy, update + 1, args)
		dt = time.time() - t0
		rss = rss_mb()
		extra = (f"kl {stats['kl']:.4f} ent {stats['entropy']:.3f} gnorm {stats['grad_norm']:.2f}"
			if stats else "SKIPPED (no signal)")
		print(f"update {update + 1}/{args.updates} win {batch_win:.2f} (ema {win_ema:.3f}) "
			f"|adv| {mean_adv:.3f} dec {len(decisions)} {extra} "
			f"{league_msg}{dt:.1f}s/update (rollout {t_roll:.1f}s, {dt / len(results):.1f}s/game)"
			+ (f" rss {rss:.0f}MB" if rss else "")
			+ (f" gpu {torch.cuda.max_memory_allocated() / 2**20:.0f}MB" if device == "cuda" else ""),
			flush=True)
		if args.save_every and (update + 1) % args.save_every == 0 and update + 1 < args.updates:
			checkpoint(last, update + 1)

	checkpoint(last, args.updates)
	save_checkpoint(args.out + ".final.tmp", policy)
	# final model goes to <out>/ itself (keep <out>/last)
	for name in os.listdir(args.out + ".final.tmp"):
		dst = os.path.join(args.out, name)
		if os.path.isfile(dst):
			os.remove(dst)
		os.replace(os.path.join(args.out + ".final.tmp", name), dst)
	shutil.rmtree(args.out + ".final.tmp", ignore_errors=True)
	print(f"[rl] saved to {args.out}")


if __name__ == "__main__":
	main()
