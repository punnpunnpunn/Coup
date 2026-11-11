"""Fine-tune a small causal LM on Coup transcripts.

Usage: python -m lm.train --data data/transcripts.jsonl --out checkpoints/coup-lm
       python -m lm.train --export-base --out checkpoints/coup-lm-untrained
Loss is only on move lines (see lm.encoding.is_move_line).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm.encoding import header_lines, is_move_line, is_own_move_line
from lm.model import DEFAULT_BASE, DEFAULT_CHECKPOINT

try:
	import psutil
	_PROC = psutil.Process()
except ImportError:  # pragma: no cover
	_PROC = None

# per-token codes: 0 = no loss, 1 = other player's move, 2 = own (P0) move
NO_LOSS, OTHER_MOVE, OWN_MOVE = 0, 1, 2


def rss_gb():
	return _PROC.memory_info().rss / 2**30 if _PROC else float("nan")


class LineCache:
	"""line -> (int32 ids of line + newline, move code). Lines repeat massively."""

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
		self.ids: dict[str, np.ndarray] = {}
		self.code: dict[str, int] = {}

	def add(self, lines):
		new = list({l for l in lines if l not in self.ids})
		if not new:
			return
		enc = self.tokenizer([l + "\n" for l in new], add_special_tokens=False)["input_ids"]
		for line, toks in zip(new, enc):
			self.ids[line] = np.asarray(toks, dtype=np.int32)
			self.code[line] = (
				OWN_MOVE if is_own_move_line(line) else OTHER_MOVE if is_move_line(line) else NO_LOSS
			)

	def encode(self, lines, with_codes):
		self.add(lines)
		parts = [self.ids[l] for l in lines]
		ids = np.concatenate(parts) if parts else np.zeros(0, np.int32)
		if with_codes:
			codes = np.concatenate([np.full(len(p), self.code[l], np.uint8) for p, l in zip(parts, lines)]) \
				if parts else np.zeros(0, np.uint8)
		else:
			codes = np.zeros(len(ids), np.uint8)
		return ids, codes


def build_example(rec, cache, max_len):
	"""RULES + history + now + n/res + body; loss only on body move lines.
	Returns (int32 ids, uint8 codes), truncated to max_len."""
	rules = header_lines(rec["n"], rec["result"])[:-1]
	history = rec.get("history") or []
	tail = ["now", f"n {rec['n']} res {rec['result']}"]
	r_ids, _ = cache.encode(rules, False)
	t_ids, _ = cache.encode(tail, False)
	b_ids, b_codes = cache.encode(rec["body"], True)
	# split history into whole games on `past` lines; drop oldest first
	games, cur = [], []
	for line in history:
		if line.startswith("past ") and cur:
			games.append(cur)
			cur = []
		cur.append(line)
	if cur:
		games.append(cur)
	g_ids = [cache.encode(g, False)[0] for g in games]
	fixed = len(r_ids) + len(t_ids) + len(b_ids)
	total_hist = sum(len(g) for g in g_ids)
	while g_ids and fixed + total_hist > max_len:
		total_hist -= len(g_ids.pop(0))
	prefix = np.concatenate([r_ids] + g_ids + [t_ids])
	ids = np.concatenate([prefix, b_ids])[:max_len]
	codes = np.concatenate([np.zeros(len(prefix), np.uint8), b_codes])[:max_len]
	return ids, codes


def load_examples(path, tokenizer, max_len):
	cache = LineCache(tokenizer)
	examples = []
	with open(path, encoding="utf-8") as f:
		for raw in f:
			raw = raw.strip()
			if raw:
				examples.append(build_example(json.loads(raw), cache, max_len))
	print(f"tokenised {len(examples)} examples, {len(cache.ids)} unique lines", flush=True)
	return examples


def collate(batch, pad_id, device, loss_on):
	"""Returns x, attention, targets for the optimised loss (-100 elsewhere), own-move mask."""
	longest = max(len(ids) for ids, _ in batch)
	x = np.full((len(batch), longest), pad_id, dtype=np.int64)
	y = np.full((len(batch), longest), -100, dtype=np.int64)
	own = np.zeros((len(batch), longest), dtype=bool)
	attn = np.zeros((len(batch), longest), dtype=np.int64)
	for i, (ids, codes) in enumerate(batch):
		n = len(ids)
		x[i, :n] = ids
		attn[i, :n] = 1
		keep = codes == OWN_MOVE if loss_on == "own" else codes >= OTHER_MOVE
		y[i, :n][keep] = ids[keep]
		own[i, :n] = codes == OWN_MOVE
	t = lambda a: torch.from_numpy(a).to(device)
	return t(x), t(attn), t(y), t(own)


def token_losses(model, x, attn, y, own, amp):
	"""CE per labelled token (flat), plus the own-move flag of each. The LM head runs only
	on labelled positions (own-move tokens are always a subset), which saves GBs of logits."""
	with amp:
		hidden = model.model(input_ids=x, attention_mask=attn).last_hidden_state[:, :-1]
		tgt = y[:, 1:]
		sel = tgt != -100
		logits = model.lm_head(hidden[sel])
	ce = torch.nn.functional.cross_entropy(logits.float(), tgt[sel], reduction="none")
	return ce, own[:, 1:][sel]


@torch.no_grad()
def evaluate(model, data, batch_size, pad_id, device, loss_on):
	model.eval()
	tot, cnt, otot, ocnt = 0.0, 0, 0.0, 0
	for i in range(0, len(data), batch_size):
		x, a, y, own = collate(data[i : i + batch_size], pad_id, device, loss_on)
		ce, o = token_losses(model, x, a, y, own, torch.autocast(device_type="cpu", enabled=False))
		tot += ce.sum().item()
		cnt += ce.numel()
		otot += ce[o].sum().item()
		ocnt += o.sum().item()
	model.train()
	return tot / max(cnt, 1), otot / max(ocnt, 1)


def save_dir(model, tokenizer, path):
	model.save_pretrained(path)
	tokenizer.save_pretrained(path)


def epoch_batches(train, args, epoch):
	"""Deterministic batch order for an epoch (so resume can skip to a position)."""
	rng = random.Random(args.seed * 1000 + epoch)
	idx = list(range(len(train)))
	rng.shuffle(idx)
	chunk = args.batch_size * 50
	order = []
	for i in range(0, len(idx), chunk):
		order.extend(sorted(idx[i : i + chunk], key=lambda j: len(train[j][0])))
	batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
	rng.shuffle(batches)
	return batches


def save_last(out, model, tokenizer, state):
	tmp, last, old = (os.path.join(out, n) for n in ("last.tmp", "last", "last.old"))
	shutil.rmtree(tmp, ignore_errors=True)
	save_dir(model, tokenizer, tmp)
	torch.save(state, os.path.join(tmp, "trainer_state.pt"))
	shutil.rmtree(old, ignore_errors=True)
	if os.path.exists(last):
		os.rename(last, old)
	os.rename(tmp, last)
	shutil.rmtree(old, ignore_errors=True)


def main():
	p = argparse.ArgumentParser()
	p.add_argument("--data", default="data/transcripts.jsonl")
	p.add_argument("--base", default=DEFAULT_BASE)
	p.add_argument("--out", default=DEFAULT_CHECKPOINT)
	p.add_argument("--epochs", type=int, default=1)
	p.add_argument("--lr", type=float, default=1e-4)
	p.add_argument("--batch-size", type=int, default=16, help="effective batch size")
	p.add_argument("--micro-batch", type=int, default=2, help="sequences per forward pass")
	p.add_argument("--max-len", type=int, default=3072)
	p.add_argument("--val-frac", type=float, default=0.02)
	p.add_argument("--seed", type=int, default=0)
	p.add_argument("--max-steps", type=int, default=0, help="stop after N steps (LR schedule still spans the full run)")
	p.add_argument("--log-every", type=int, default=50)
	p.add_argument("--loss-on", choices=["moves", "own"], default="moves",
		help="moves: every player's move lines; own: only P0's move lines")
	p.add_argument("--save-every", type=int, default=0, help="write <out>/last/ every N steps")
	p.add_argument("--resume", action="store_true", help="continue from <out>/last/")
	p.add_argument("--empty-cache-every", type=int, default=10, help="release cached CUDA memory every N steps")
	p.add_argument("--export-base", action="store_true", help="just save the pretrained base model to --out")
	args = p.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	tokenizer = AutoTokenizer.from_pretrained(args.base)
	if args.export_base:
		model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
		save_dir(model, tokenizer, args.out)
		print(f"exported base model {args.base} to {args.out}")
		return

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	pad_id = tokenizer.pad_token_id
	if pad_id is None:
		pad_id = tokenizer.eos_token_id
	last_dir = os.path.join(args.out, "last")
	state = None
	if args.resume:
		if not os.path.exists(os.path.join(last_dir, "trainer_state.pt")):
			raise SystemExit(f"--resume: no trainer state in {last_dir}")
		state = torch.load(os.path.join(last_dir, "trainer_state.pt"), weights_only=False)
	model = AutoModelForCausalLM.from_pretrained(last_dir if state else args.base, torch_dtype=torch.float32).to(device)
	model.config.use_cache = False
	model.gradient_checkpointing_enable()
	model.train()

	t_tok = time.time()
	data = load_examples(args.data, tokenizer, args.max_len)
	print(f"tokenisation took {time.time() - t_tok:.1f}s, rss {rss_gb():.2f} GB", flush=True)
	random.Random(args.seed).shuffle(data)
	n_val = int(len(data) * args.val_frac)
	val, train = data[:n_val], data[n_val:]
	print(f"{len(train)} train / {len(val)} val examples, device={device}, loss-on={args.loss_on}")

	steps_per_epoch = max(1, math.ceil(len(train) / args.batch_size))
	total_steps = steps_per_epoch * args.epochs
	stop_at = min(total_steps, args.max_steps) if args.max_steps else total_steps
	warmup = max(1, int(0.03 * total_steps))
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
	sched = torch.optim.lr_scheduler.LambdaLR(
		opt,
		lambda s: (s + 1) / warmup if s < warmup
		else max(0.0, (total_steps - s) / max(1, total_steps - warmup)),
	)
	use_amp = device == "cuda"
	amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
	scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

	step = 0
	if state:
		if state["total_steps"] != total_steps:
			print(f"WARNING: total steps changed {state['total_steps']} -> {total_steps}; LR schedule differs")
		opt.load_state_dict(state["optimizer"])
		sched.load_state_dict(state["scheduler"])
		scaler.load_state_dict(state["scaler"])
		random.setstate(state["py_rng"])
		torch.set_rng_state(state["torch_rng"])
		if torch.cuda.is_available() and state.get("cuda_rng") is not None:
			torch.cuda.set_rng_state(state["cuda_rng"])
		step = state["step"]
		print(f"resumed from step {step}, lr {sched.get_last_lr()[0]:.2e}", flush=True)
		del state

	def make_state():
		return {
			"step": step, "total_steps": total_steps, "args": vars(args),
			"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
			"py_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
			"cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
			"epoch": step // steps_per_epoch, "batch_in_epoch": step % steps_per_epoch,
		}

	t0 = time.time()
	run = [0.0, 0, 0.0, 0]  # optimised loss sum, tokens, own loss sum, own tokens
	start_step = step
	peak_rss = rss_gb()
	for epoch in range(step // steps_per_epoch, args.epochs):
		batches = epoch_batches(train, args, epoch)
		for bidx in batches[step % steps_per_epoch :]:
			if step >= stop_at:
				break
			batch = [train[j] for j in bidx]
			opt.zero_grad(set_to_none=True)
			n_tok = 0
			for ids, codes in batch:
				n_tok += int((codes == OWN_MOVE).sum() if args.loss_on == "own" else (codes >= OTHER_MOVE).sum())
			n_tok = max(n_tok, 1)
			amp = torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp)
			s_loss = s_own = 0.0
			s_cnt = s_ocnt = 0
			for m in range(0, len(batch), args.micro_batch):
				x, a, y, own = collate(batch[m : m + args.micro_batch], pad_id, device, args.loss_on)
				ce, o = token_losses(model, x, a, y, own, amp)
				loss = ce.sum() / n_tok
				scaler.scale(loss).backward()
				s_loss += ce.sum().item()
				s_cnt += ce.numel()
				s_own += ce[o].sum().item()
				s_ocnt += o.sum().item()
				del x, a, y, own, ce, o, loss
			scaler.unscale_(opt)
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			scaler.step(opt)
			scaler.update()
			sched.step()
			step += 1
			if device == "cuda" and step % args.empty_cache_every == 0:
				# variable-length batches fragment the CUDA cache; on Windows (WDDM) the
				# overflow spills into shared system RAM, which showed up as RSS growth
				torch.cuda.empty_cache()
			run[0] += s_loss; run[1] += s_cnt; run[2] += s_own; run[3] += s_ocnt
			if step % args.log_every == 0 or step == start_step + 1:
				peak_rss = max(peak_rss, rss_gb())
				print(f"epoch {epoch} step {step}/{total_steps} loss {run[0] / max(run[1], 1):.4f} "
					f"own {run[2] / max(run[3], 1):.4f} lr {sched.get_last_lr()[0]:.2e} "
					f"rss {rss_gb():.2f}GB {time.time() - t0:.0f}s "
					f"({(time.time() - t0) / (step - start_step):.2f}s/step)", flush=True)
				run = [0.0, 0, 0.0, 0]
			if args.save_every and step % args.save_every == 0:
				save_last(args.out, model, tokenizer, make_state())
				print(f"saved {last_dir} at step {step}", flush=True)
		if step >= stop_at:
			break
		if val:
			vl, vo = evaluate(model, val, args.micro_batch, pad_id, device, args.loss_on)
			print(f"epoch {epoch} val loss {vl:.4f} own {vo:.4f}", flush=True)

	if args.save_every and step % args.save_every:
		save_last(args.out, model, tokenizer, make_state())
	save_dir(model, tokenizer, args.out)
	print(f"saved to {args.out} (peak rss {peak_rss:.2f} GB)")


if __name__ == "__main__":
	main()
