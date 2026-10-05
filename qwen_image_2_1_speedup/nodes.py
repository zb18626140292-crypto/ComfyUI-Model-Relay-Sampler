"""Qwen Image 2.1 Speedup node.

Step-level residual caching (TeaCache-style) for the Qwen Image 2.1
transformer with sigma-space residual extrapolation (TaylorCache-style) and
an optional error-feedback controller. The cache rides on the
DIFFUSION_MODEL wrapper so the model's built-in prefix K/V cache (text +
reference images, computed once per sampling run) stays active on full steps.
"""

import logging
import math
import time

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

_MIB = 1024 * 1024
_PATCH_KEY = "qwen_image_2_1_speedup"
_MODEL_NAME = "QwenImage21Transformer2DModel"


class _StreamState:
    def __init__(self):
        self.residuals = []  # newest-first [(sigma, tensor)], up to 3 for second-order forecast
        self.prev_indicator = None
        self.last_sigma = None
        self.last_used = None
        self.threshold_eff = None
        self.starve_count = 0
        self.accumulated = 0.0
        self.consecutive_skips = 0
        self.full_steps = 0
        self.cache_hits = 0
        self.drift_sum = 0.0
        self.drift_max = 0.0
        self.drift_count = 0
        self.pred_err_sum = 0.0
        self.pred_err_max = 0.0
        self.pred_err_count = 0

    def clear_tensors(self):
        self.residuals = []
        self.prev_indicator = None
        self.last_used = None


class _TurboCache:
    """Caches the whole-forward residual (out - x) per conditioning stream and
    replays it while the accumulated drift of the timestep embedding stays
    under the threshold. The replayed residual is a sigma-space extrapolation
    from the measured residual history: first-order secant, or second-order
    Newton form with the quadratic term clamped to the linear term's
    magnitude. With a target error set, measured replay errors steer the
    effective threshold between corrections."""

    _MAX_EXTRAPOLATE = 1.5
    _HISTORY = 3

    def __init__(self, threshold, start_percent, end_percent, max_consecutive_skips, cache_device,
                 forecast="first", target_error=0.0, debug=False):
        self.threshold = threshold
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_consecutive_skips = max_consecutive_skips
        self.cache_device = cache_device
        self.forecast = forecast
        self.target_error = target_error
        self.debug = debug
        self.streams = {}

    def reset(self):
        self.streams = {}

    def finish(self):
        full = sum(s.full_steps for s in self.streams.values())
        hits = sum(s.cache_hits for s in self.streams.values())
        drift_count = sum(s.drift_count for s in self.streams.values())
        if full + hits > 0:
            logging.info(
                "QwenImage21Speedup: %d cached of %d model forwards (%.1f%% skipped)",
                hits, full + hits, hits / (full + hits) * 100)
        if drift_count > 0 and (self.debug or hits == 0):
            drift_sum = sum(s.drift_sum for s in self.streams.values())
            drift_max = max(s.drift_max for s in self.streams.values())
            logging.info(
                "QwenImage21Speedup: per-step indicator drift mean %.4f, max %.4f over %d steps "
                "(a useful cache_threshold must exceed the mean; current %.3f)",
                drift_sum / drift_count, drift_max, drift_count, self.threshold)
        pred_err_count = sum(s.pred_err_count for s in self.streams.values())
        if pred_err_count > 0 and (self.debug or self.target_error > 0):
            err_sum = sum(s.pred_err_sum for s in self.streams.values())
            err_max = max(s.pred_err_max for s in self.streams.values())
            eff = [s.threshold_eff for s in self.streams.values() if s.threshold_eff is not None]
            logging.info(
                "QwenImage21Speedup: replayed-residual error vs actual forward mean %.4f, max %.4f over %d checks%s",
                err_sum / pred_err_count, err_max, pred_err_count,
                f", final effective threshold {sum(eff) / len(eff):.3f}" if eff else "")
        for s in self.streams.values():
            s.clear_tensors()
        self.streams = {}

    @staticmethod
    def _stream_key(x, context, ref_latents, transformer_options):
        uuids = transformer_options.get("uuids")
        stream = tuple(str(u) for u in uuids) if uuids else ("default",)
        refs = tuple(tuple(r.shape) for r in (ref_latents or []))
        return (stream, tuple(x.shape), str(x.dtype), tuple(context.shape), refs)

    @staticmethod
    def _step_info(transformer_options):
        sigmas = transformer_options.get("sigmas")
        sample_sigmas = transformer_options.get("sample_sigmas")
        if sigmas is None or sample_sigmas is None:
            return None
        sigma = float(sigmas[0])
        schedule = [float(s) for s in sample_sigmas]
        step = min(range(len(schedule)), key=lambda i: abs(schedule[i] - sigma))
        percent = min(1.0, step / max(1, len(schedule) - 2))
        return sigma, percent

    @staticmethod
    def _indicator(model, timestep, dtype):
        # same rounding as the model: target rows of the timestep embedding
        t = ((timestep * 1000).to(dtype) / 1000).to(dtype)
        temb = model.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
        return temb[:-1].detach().float().flatten()

    def _store_residual(self, state, residual, sigma):
        # residuals outlive the forward, keep them out of the malloc graph
        with comfy.model_prefetch.pause_malloc_graph():
            residual = residual.detach()
            location = self.cache_device
            if residual.device.type == "cpu":
                location = "cpu"
            elif location == "auto":
                free = comfy.model_management.get_free_memory(residual.device)
                location = "gpu" if free > 10 * residual.numel() * residual.element_size() + 256 * _MIB else "cpu"
            if location == "gpu":
                stored = residual.clone()
            else:
                stored = torch.empty(residual.shape, dtype=residual.dtype, device="cpu",
                                     pin_memory=torch.cuda.is_available())
                stored.copy_(residual, non_blocking=False)
            state.residuals.insert(0, (sigma, stored))
            del state.residuals[self._HISTORY:]

    def _replay(self, state, x, sigma):
        """Residual for a cached step: plain replay, or sigma-space
        extrapolation from the measured residual history. The horizon is
        clamped to _MAX_EXTRAPOLATE measured intervals, and the second-order
        term elementwise to the linear term's magnitude."""
        hist = state.residuals
        t2 = hist[0][1].to(device=x.device, dtype=x.dtype)
        r1 = t2
        f = 0.0
        if self.forecast != "off" and sigma is not None and len(hist) > 1:
            s1, s0 = hist[0][0], hist[1][0]
            if s1 is not None and s0 is not None and abs(s1 - s0) > 1e-8:
                f = min(max((sigma - s1) / (s1 - s0), 0.0), self._MAX_EXTRAPOLATE)
        if f > 0.0:
            t1 = hist[1][1].to(device=x.device, dtype=x.dtype)
            r1 = torch.lerp(t1, t2, 1.0 + f)
            if self.forecast == "second" and len(hist) > 2 and hist[2][0] is not None and abs(hist[0][0] - hist[2][0]) > 1e-8:
                s2, s1, s0 = hist[0][0], hist[1][0], hist[2][0]
                t0 = hist[2][1].to(device=x.device, dtype=x.dtype)
                d1 = (t2 - t1) / (s2 - s1)
                d0 = (t1 - t0) / (s1 - s0)
                linear = d1 * (sigma - s2)
                quad = (d1 - d0) / (s2 - s0) * ((sigma - s2) * (sigma - s1))
                r1 = r1 + torch.clamp(quad, -linear.abs(), linear.abs())
        if self.debug or self.target_error > 0:
            with comfy.model_prefetch.pause_malloc_graph():
                state.last_used = r1.detach().clone()
        if self.debug:
            logging.info("QwenImage21Speedup: replayed residual, extrapolation factor %.2f", f)
        return x + r1

    def __call__(self, executor, x, timestep, context, ref_latents=None, image_slots=None, transformer_options=None, **kwargs):
        transformer_options = transformer_options or {}
        model = executor.class_obj
        state = self.streams.setdefault(
            self._stream_key(x, context, ref_latents, transformer_options), _StreamState())
        step_info = self._step_info(transformer_options)
        sigma = step_info[0] if step_info is not None else None

        with comfy.model_prefetch.pause_malloc_graph():
            indicator = self._indicator(model, timestep, x.dtype)

        eligible = False
        reason = "no cached residual yet"
        if step_info is not None and state.residuals and state.prev_indicator is not None:
            sigma, percent = step_info
            if state.last_sigma is not None and sigma > state.last_sigma + 1e-6:
                reason = "sigma moved backwards (new run), forcing full"
            else:
                threshold = state.threshold_eff if state.threshold_eff is not None else self.threshold
                diff = float((indicator - state.prev_indicator).abs().mean()
                             / state.prev_indicator.abs().mean().clamp_min(1e-6))
                state.accumulated += diff
                state.drift_sum += diff
                state.drift_max = max(state.drift_max, diff)
                state.drift_count += 1
                if not (self.start_percent <= percent <= self.end_percent):
                    reason = f"outside window ({percent:.2f})"
                elif state.accumulated >= threshold:
                    reason = f"accumulated drift {state.accumulated:.4f} >= threshold {threshold:.3f}"
                    if self.target_error > 0:
                        # starvation recovery: without skips there are no error
                        # measurements, so a shrunk threshold could never climb back
                        state.starve_count += 1
                        if state.starve_count >= 3:
                            state.threshold_eff = min(threshold * 1.5, 2.0)
                            state.starve_count = 0
                            reason += f"; no skips for 3 steps, effective threshold raised to {state.threshold_eff:.3f}"
                elif state.consecutive_skips >= self.max_consecutive_skips:
                    reason = "max consecutive skips reached"
                else:
                    eligible = True
                    state.starve_count = 0
                if self.debug:
                    logging.info(
                        "QwenImage21Speedup: sigma %.4f, percent %.2f, drift %.4f, accumulated %.4f -> %s",
                        sigma, percent, diff, state.accumulated, "cached" if eligible else f"full ({reason})")

        if eligible:
            out = self._replay(state, x, sigma)
            state.cache_hits += 1
            state.consecutive_skips += 1
        else:
            out = executor(x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs)
            residual = out - x
            if state.last_used is not None:
                # measured error of the last replay, the feedback signal for adaptive mode
                err = float((residual - state.last_used).abs().mean() / out.abs().mean().clamp_min(1e-6))
                state.pred_err_sum += err
                state.pred_err_max = max(state.pred_err_max, err)
                state.pred_err_count += 1
                state.last_used = None
                if self.target_error > 0:
                    base = state.threshold_eff if state.threshold_eff is not None else self.threshold
                    ratio = min(max(self.target_error / max(err, 1e-4), 0.7), 1.3)
                    state.threshold_eff = min(max(base * ratio, 0.05), 2.0)
                if self.debug:
                    logging.info(
                        "QwenImage21Speedup: actual forward vs last replayed residual, relative error %.4f%s",
                        err, f", effective threshold -> {state.threshold_eff:.3f}"
                        if state.threshold_eff is not None else "")
            self._store_residual(state, residual, sigma)
            state.full_steps += 1
            state.consecutive_skips = 0
            state.accumulated = 0.0

        with comfy.model_prefetch.pause_malloc_graph():
            state.prev_indicator = indicator
        if step_info is not None:
            state.last_sigma = step_info[0]
        return out


class _SamplingScope:
    def __init__(self, cache):
        self.cache = cache

    def __call__(self, executor, *args, **kwargs):
        self.cache.reset()
        try:
            return executor(*args, **kwargs)
        finally:
            self.cache.finish()


class QwenImage21Speedup(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21Speedup",
            display_name="Qwen Image 2.1 Speedup",
            category="model/patch",
            description="Sampling accelerator for Qwen Image 2.1: caches the whole-model residual and replays it on "
                        "low-drift steps (TeaCache-style) with sigma-space extrapolation (TaylorCache-style). "
                        "Compatible with the model's built-in prefix K/V cache.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enable_cache", default=True,
                                 tooltip="Skip whole model forwards by replaying the cached residual while the "
                                         "accumulated timestep-embedding drift stays under the threshold."),
                io.Float.Input("cache_threshold", default=0.30, min=0.0, max=2.0, step=0.01,
                               tooltip="Accumulated relative drift allowed before forcing a full forward. "
                                       "Measured drift on this model is ~0.13 per step at 40 steps, so the "
                                       "threshold is roughly 0.13 x the skip run length: 0.3 skips ~2 steps, "
                                       "0.5 ~3-4, 0.8 ~6. Starting point for the adaptive controller."),
                io.Float.Input("target_error", default=0.05, min=0.0, max=1.0, step=0.005,
                               tooltip="Adaptive mode: the effective threshold is adjusted after every measured "
                                       "replay error to hold the error near this target (multiplicative "
                                       "feedback, 0.7x-1.3x per correction). 0.06-0.08 is faster with slightly "
                                       "lower fidelity. 0 uses the fixed cache_threshold."),
                io.Float.Input("cache_start_percent", default=0.15, min=0.0, max=1.0, step=0.01,
                               tooltip="Caching only kicks in after this point in the sampling schedule. "
                                       "Measured replay error is highest right after the start, keep >= 0.15."),
                io.Float.Input("cache_end_percent", default=0.90, min=0.0, max=1.0, step=0.01,
                               tooltip="Caching stops after this point; the tail always runs full forwards."),
                io.Int.Input("max_consecutive_skips", default=3, min=1, max=10, step=1,
                             tooltip="Upper bound on cached forwards in a row before a full refresh."),
                io.Combo.Input("forecast", options=["first", "second", "off"], default="first",
                               tooltip="Residual extrapolation order for cached steps. first: sigma-space linear "
                                       "extrapolation from the last two measured residuals. second: adds the "
                                       "quadratic term from the last three, clamped to the linear term's "
                                       "magnitude. off: replay the latest residual verbatim."),
                io.Combo.Input("cache_device", options=["auto", "gpu", "cpu"], default="auto",
                               tooltip="Where the cached residual lives. auto uses spare VRAM, else pinned RAM."),
                io.Boolean.Input("debug_log", default=False,
                                 tooltip="Log per-step sigma, drift and cache decisions to the console."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enable_cache, cache_threshold, target_error, cache_start_percent, cache_end_percent,
                max_consecutive_skips, forecast, cache_device, debug_log=False):
        diffusion_model = model.get_model_object("diffusion_model")
        if type(diffusion_model).__name__ != _MODEL_NAME:
            raise ValueError(f"QwenImage21Speedup only supports Qwen Image 2.1 ({_MODEL_NAME}), "
                             f"got {type(diffusion_model).__name__}")
        if not enable_cache:
            logging.info("QwenImage21Speedup: cache disabled, model passed through unchanged")
            return io.NodeOutput(model)

        m = model.clone()
        cache = _TurboCache(cache_threshold, cache_start_percent, cache_end_percent,
                            max_consecutive_skips, cache_device, forecast=forecast,
                            target_error=target_error, debug=debug_log)
        m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY, cache)
        m.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY)
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY, _SamplingScope(cache))
        return io.NodeOutput(m)


_SAMPLER_PRESETS = {
    "保真": dict(threshold=0.20, target_error=0.025, start_percent=0.20,
               end_percent=0.85, max_consecutive_skips=2),
    "均衡": dict(threshold=0.30, target_error=0.05, start_percent=0.15,
               end_percent=0.90, max_consecutive_skips=3),
    "快速": dict(threshold=0.45, target_error=0.08, start_percent=0.15,
               end_percent=0.90, max_consecutive_skips=4),
}


class QwenImage21FastSampler:
    """Qwen 2.1 single-stage sampler using the existing run-scoped speedup.

    Native KSampler handles noise, masks, previews, and latent metadata. The
    latent returned by TextEncodeQwenImage21 is never resized here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL", {"display_name": "模型", "tooltip": "仅支持 Qwen-Image 2.1；可先连接原生 Qwen Image 2.1 Cache 节点。"}),
            "positive": ("CONDITIONING", {"display_name": "正向", "tooltip": "连接 Qwen-Image 2.1 编码节点的正向条件。"}),
            "negative": ("CONDITIONING", {"display_name": "负向", "tooltip": "连接同一编码节点的负向条件。"}),
            "latent_image": ("LATENT", {"display_name": "输入潜变量", "tooltip": "图像编辑时直接连接 Qwen 编码节点的潜变量；不会内部放大。"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                            "control_after_generate": True, "display_name": "种子"}),
            "steps": ("INT", {"default": 40, "min": 1, "max": 10000,
                             "display_name": "步数", "tooltip": "40 步是 Qwen 官方质量基线；加速档位减少实际模型前向次数。"}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                              "step": 0.1, "round": 0.01, "display_name": "CFG"}),
            "sampler_name": (comfy.samplers.KSampler.SAMPLERS,
                             {"default": "euler", "display_name": "采样器"}),
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                          {"default": "simple", "display_name": "调度器"}),
            "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                  "step": 0.01, "display_name": "降噪强度"}),
            "acceleration": (("均衡", "保真", "快速", "关闭"),
                             {"default": "均衡", "display_name": "加速档位",
                              "tooltip": "均衡采用现有实测参数；保真减少近似，快速增加近似。关闭可作同节点基线。"}),
            "cache_device": (("自动", "显存", "内存"),
                             {"default": "自动", "display_name": "残差缓存位置",
                              "tooltip": "auto 优先使用空闲显存，否则使用内存；与 Qwen 原生 KV 缓存不同。"}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("采样结果",)
    FUNCTION = "sample"
    CATEGORY = "采样/Qwen Image 2.1"
    DESCRIPTION = "Qwen-Image 2.1 专用单采：复用按步残差缓存，不改变参考图潜变量尺寸。"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               sampler_name, scheduler, denoise, acceleration, cache_device):
        if acceleration not in _SAMPLER_PRESETS and acceleration != "关闭":
            raise ValueError(f"未知加速档位: {acceleration}")
        diffusion_model = model.get_model_object("diffusion_model")
        if type(diffusion_model).__name__ != _MODEL_NAME:
            raise ValueError(
                f"QwenImage21FastSampler 仅支持 {_MODEL_NAME}，实际为 {type(diffusion_model).__name__}"
            )

        sampling_model = model.clone()
        sampling_model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY)
        sampling_model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY)
        if acceleration != "关闭":
            cache = _TurboCache(cache_device={"自动": "auto", "显存": "gpu", "内存": "cpu"}[cache_device],
                                forecast="first",
                                **_SAMPLER_PRESETS[acceleration])
            sampling_model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY, cache)
            sampling_model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY,
                                                _SamplingScope(cache))

        import nodes as comfy_nodes

        started = time.perf_counter()
        try:
            return comfy_nodes.common_ksampler(
                sampling_model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent_image, denoise=denoise,
            )
        finally:
            logging.info(
                "QwenImage21FastSampler: mode=%s, steps=%d, latent=%dx%d, "
                "sampler=%s/%s, elapsed=%.3fs",
                acceleration, steps, latent_image["samples"].shape[-1],
                latent_image["samples"].shape[-2], sampler_name, scheduler,
                time.perf_counter() - started,
            )


_DUAL_MISSING = object()
_DUAL_STAGE_TWO_INPUTS = ("stage_two_model", "stage_two_positive", "stage_two_negative")
_DUAL_MEMORY_MODES = ("省显存", "自动")
_DUAL_PREFIX_CACHE_MODES = ("安全关闭", "沿用输入模型设置")


def _dual_model_signature(model, stage_name):
    diffusion = model.get_model_object("diffusion_model")
    if type(diffusion).__name__ != _MODEL_NAME:
        raise ValueError(f"{stage_name}仅支持 Qwen-Image 2.1，实际为 {type(diffusion).__name__}")
    latent_format = model.get_model_object("latent_format")
    return (
        type(latent_format),
        int(latent_format.latent_channels),
        int(latent_format.latent_dimensions),
        int(latent_format.spacial_downscale_ratio),
        int(latent_format.temporal_downscale_ratio),
    )


def _dual_sigmas(model, steps, scheduler, denoise):
    sampling = model.get_model_object("model_sampling")
    if denoise >= 0.9999:
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps)
    else:
        expanded_steps = max(steps, int(steps / denoise))
        sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, expanded_steps)
        sigmas = sigmas[-(steps + 1):]
    return sigmas.to(comfy.model_management.intermediate_device())


def _dual_nearest_sigma(sigmas, target):
    # Exclude the initial and terminal sigma: each model must run >= 1 step.
    candidates = sigmas[1:-1].detach().float().cpu().clamp_min(1e-12)
    if candidates.numel() == 0:
        raise ValueError("双采时每个模型至少需要 2 个计划步数。")
    distances = (torch.log(candidates) - math.log(max(float(target), 1e-12))).abs()
    return int(torch.argmin(distances)) + 1


def _dual_sigma_pair(model_one, model_two, steps_one, steps_two,
                     scheduler_one, scheduler_two, denoise, handoff_mode, handoff_value,
                     minimum_first_steps=1):
    sigmas_one = _dual_sigmas(model_one, steps_one, scheduler_one, denoise)
    sigmas_two = _dual_sigmas(model_two, steps_two, scheduler_two, denoise)
    if handoff_mode == "percent":
        two_start = max(1, min(int(round(steps_two * handoff_value / 100.0)), steps_two - 1))
    else:
        target_sigma = 10.0 ** (-float(handoff_value) / 20.0)
        two_start = _dual_nearest_sigma(sigmas_two, target_sigma)
    boundary = sigmas_two[two_start].clone()
    if float(boundary) <= 0.0 or float(boundary) >= float(sigmas_one[0]):
        raise ValueError("两个 Qwen 模型的 sigma 时间线没有可用的交接区间。")
    one_end = _dual_nearest_sigma(sigmas_one, boundary)
    minimum_first_steps = max(1, min(int(minimum_first_steps), steps_one - 1))
    if one_end < minimum_first_steps:
        # A one-step x0 is not suitable for latent enlargement. Move both
        # models to the nearest common sigma after enough stage-one work.
        safe_sigma = float(sigmas_one[minimum_first_steps])
        two_start = _dual_nearest_sigma(sigmas_two, safe_sigma)
        boundary = sigmas_two[two_start].clone()
        if float(boundary) >= float(sigmas_one[0]):
            raise ValueError("放大前无法找到有效的共同 sigma，请增加两个模型的计划步数。")
        one_end = _dual_nearest_sigma(sigmas_one, boundary)
        one_end = max(minimum_first_steps, one_end)
    first = torch.cat((sigmas_one[:one_end], boundary.reshape(1)))
    second = sigmas_two[two_start:].clone()
    second[0] = boundary
    return first, second, one_end, two_start, float(boundary)


def _dual_reference_info(conditioning):
    has_images = False
    first_shapes = set()
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        details = entry[1]
        if not isinstance(details, dict):
            continue
        if "image_slots" in details:
            has_images = True
        refs = details.get("reference_latents")
        if refs is not None:
            has_images = True
            if isinstance(refs, (list, tuple)) and refs and torch.is_tensor(refs[0]):
                first_shapes.add(tuple(int(v) for v in refs[0].shape[-2:]))
    if len(first_shapes) > 1:
        raise ValueError("参考图条件包含不同的第一参考图尺寸，无法安全放大。")
    return has_images, next(iter(first_shapes)) if first_shapes else None


def _dual_validate_upscale_conditions(samples, target_shape, conditionings, dual):
    info = [_dual_reference_info(conditioning) for conditioning in conditionings]
    if not any(has_images for has_images, _shape in info):
        return  # Plain text-to-image.
    if not dual:
        raise ValueError("图像编辑单采请先在 Qwen 编码节点设置目标尺寸，不要在采样器内放大。")
    if not all(has_images and shape is not None for has_images, shape in info):
        raise ValueError(
            "编辑放大需要两个阶段都用带 VAE 的 Qwen 图像编码节点，"
            "分别编码同一参考图；不能只复用低分辨率条件。"
        )
    low_shape = tuple(int(v) for v in samples.shape[-2:])
    if any(shape != low_shape for _images, shape in info[:2]):
        raise ValueError("模型一参考图编码尺寸必须等于输入潜变量尺寸。")
    if any(shape != target_shape for _images, shape in info[2:]):
        raise ValueError(
            "模型二参考图编码尺寸必须等于最终宽高；"
            "请用第二个 Qwen 编码节点按目标分辨率重新编码参考图。"
        )


def _dual_target_latent_size(samples, model, final_width, final_height):
    ratio = int(model.get_model_object("latent_format").spacial_downscale_ratio)
    current_width = int(samples.shape[-1]) * ratio
    current_height = int(samples.shape[-2]) * ratio
    if final_width <= 0 and final_height <= 0:
        return int(samples.shape[-1]), int(samples.shape[-2])
    if final_width <= 0:
        final_width = round(current_width * final_height / current_height)
    if final_height <= 0:
        final_height = round(current_height * final_width / current_width)
    target_width = max(1, round(final_width / ratio))
    target_height = max(1, round(final_height / ratio))
    if target_width < samples.shape[-1] or target_height < samples.shape[-2]:
        raise ValueError("最终宽高只支持放大；目标宽高不能小于输入潜变量尺寸。")
    return target_width, target_height


def _dual_upscale_restart_sigmas(sigmas, strength):
    boundary = float(sigmas[0])
    restart = boundary * float(strength)
    scaled = sigmas * (restart / boundary)
    scaled[-1] = 0.0
    return scaled, restart


def _dual_sampling_model(model, acceleration, cache_device, disable_prefix_cache):
    if acceleration not in _SAMPLER_PRESETS and acceleration != "关闭":
        raise ValueError(f"未知加速档位: {acceleration}")
    sampling_model = model.clone()
    sampling_model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY)
    sampling_model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY)
    if disable_prefix_cache:
        options = sampling_model.model_options.setdefault("transformer_options", {})
        previous = options.get("qwen_image21_cache", {})
        options["qwen_image21_cache"] = {**previous, "device": "off"}
    if acceleration != "关闭":
        cache = _TurboCache(
            cache_device={"自动": "auto", "显存": "gpu", "内存": "cpu"}[cache_device],
            forecast="first", **_SAMPLER_PRESETS[acceleration],
        )
        sampling_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _PATCH_KEY, cache,
        )
        sampling_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, _PATCH_KEY, _SamplingScope(cache),
        )
    return sampling_model


def _dual_sample_stage(model, positive, negative, cfg, sampler_name, sigmas,
                       samples, noise, noise_mask, seed):
    callback = latent_preview.prepare_callback(model, sigmas.numel() - 1)
    return comfy.sample.sample_custom(
        model, noise, cfg, comfy.samplers.sampler_object(sampler_name), sigmas,
        positive, negative, samples, noise_mask=noise_mask,
        callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )


def _dual_should_offload_first(model_one, model_two, memory_mode):
    if memory_mode == "省显存":
        return True
    if model_one is model_two:
        return False
    base_one = getattr(model_one, "clone_base_uuid", None)
    return not (
        base_one is not None
        and base_one == getattr(model_two, "clone_base_uuid", None)
        and getattr(model_one, "patches_uuid", None) == getattr(model_two, "patches_uuid", None)
    )


def _dual_unload(model):
    comfy.model_management.unload_model_and_clones(
        model, unload_additional_models=True, all_devices=True,
    )


def _dual_inputs(handoff_mode):
    required = {
        "stage_one_model": ("MODEL", {"display_name": "模型一"}),
        "stage_one_positive": ("CONDITIONING", {"display_name": "模型一正向"}),
        "stage_one_negative": ("CONDITIONING", {"display_name": "模型一负向"}),
        "latent_image": ("LATENT", {"display_name": "输入潜变量",
                                   "tooltip": "文生图连接空潜变量；图像编辑连接模型一 Qwen 编码节点的潜变量。设置最终宽高后，双采在两阶段之间放大。"}),
        "stage_one_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                                    "step": 0.1, "display_name": "模型一CFG"}),
        "stage_one_steps": ("INT", {"default": 40, "min": 2, "max": 10000,
                                    "display_name": "模型一步数",
                                    "tooltip": "第一模型的计划步数；交接时只执行到共同 sigma，不会完整跑完再重跑。"}),
        "stage_one_sampler": (comfy.samplers.KSampler.SAMPLERS,
                              {"default": "euler", "display_name": "模型一采样"}),
        "stage_one_scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                                {"default": "simple", "display_name": "模型一调度"}),
        "stage_one_acceleration": (("均衡", "保真", "快速", "关闭"),
                                   {"default": "均衡", "display_name": "模型一加速",
                                    "tooltip": "均衡复用部分模型前向；保真减少近似，关闭保留全部实际计算。"}),
        "stage_two_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0,
                                    "step": 0.1, "display_name": "模型二CFG"}),
        "stage_two_steps": ("INT", {"default": 40, "min": 2, "max": 10000,
                                    "display_name": "模型二步数",
                                    "tooltip": "第二模型的计划时间表；交接后只执行剩余步数。"}),
        "stage_two_sampler": (comfy.samplers.KSampler.SAMPLERS,
                              {"default": "euler", "display_name": "模型二采样"}),
        "stage_two_scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                                {"default": "simple", "display_name": "模型二调度"}),
        "stage_two_acceleration": (("保真", "均衡", "快速", "关闭"),
                                   {"default": "保真", "display_name": "模型二加速",
                                    "tooltip": "默认保真，减少后半程近似对手、脸和动作的影响。"}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                         "control_after_generate": True, "display_name": "种子"}),
        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                             "step": 0.01, "display_name": "降噪强度",
                             "tooltip": "Qwen 图像编辑推荐先用 1.0；这里决定两阶段共同时间线的起点。"}),
    }
    if handoff_mode == "percent":
        required["handoff_percent"] = (
            "FLOAT", {"default": 50.0, "min": 1.0, "max": 99.0, "step": 1.0,
                      "display_name": "交接位置（百分比）",
                      "tooltip": "按模型二时间表定位共同 sigma；40/40 步、50% 时通常是模型一约20步加模型二约20步。"},
        )
    else:
        required["handoff_snr_db"] = (
            "FLOAT", {"default": 6.0, "min": -40.0, "max": 40.0, "step": 0.1,
                      "display_name": "SNR交接值（dB）",
                      "tooltip": "按 sigma 物理噪声强度交接；实际两阶段步数由模型/调度器的时间表决定。"},
        )
    required.update({
        "enable_stage_2": (("开启", "关闭"), {"default": "开启", "display_name": "启用模型二",
                                           "tooltip": "关闭时只运行模型一完整时间表，交接与模型二参数不参与。"}),
        "prefix_cache_mode": (_DUAL_PREFIX_CACHE_MODES,
                              {"default": "安全关闭", "display_name": "双采KV缓存",
                               "tooltip": "双采默认在模型副本上关闭 Qwen 原生 KV 缓存，避开已知多参考图交接错误；单采仍沿用输入模型设置。残差加速档继续生效。"}),
        "memory_mode": (_DUAL_MEMORY_MODES,
                        {"default": "省显存", "display_name": "显存模式",
                         "tooltip": "省显存：采样前清旧模型，每阶段完成后卸载；自动：同模型尽量复用，不同模型交接时卸载模型一。"}),
        "cache_device": (("自动", "显存", "内存"),
                         {"default": "自动", "display_name": "残差缓存位置",
                          "tooltip": "逐步残差缓存的位置，与原生 KV 缓存不同。"}),
        "final_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16,
                                "display_name": "最终宽度",
                                "tooltip": "宽高都为0保持输入尺寸。双采时一采先低分辨率成图，再放大到目标尺寸；编辑模式需要模型二按目标分辨率重新编码同一参考图。"}),
        "final_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 16,
                                 "display_name": "最终高度",
                                 "tooltip": "0 表示不指定。只填宽或高时自动按输入长宽比计算另一边；宽高都为 0 关闭内部放大。"}),
        "upscale_method": (("bislerp", "bilinear", "bicubic", "nearest-exact", "area"),
                           {"default": "bislerp", "display_name": "潜变量放大算法",
                            "tooltip": "只放大第一采结束后的干净潜变量。文生图可直接使用；编辑时模型二正负条件须来自第二个目标分辨率 Qwen 编码节点。"}),
        "redraw_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0,
                                     "step": 0.05, "round": 0.01, "display_name": "重绘强度",
                                     "tooltip": "仅双采中途放大时生效。1.0 使用交接 sigma；0.8 使用其 80%。编辑放大推荐默认 1.0，较低值可能产生重影或补不齐细节。"}),
        "minimum_upscale_stage_one_steps": (
            "INT", {"default": 10, "min": 1, "max": 100,
                    "advanced": True, "display_name": "放大前最低一采步数",
                    "tooltip": "仅中途放大时生效；交接过早会自动推迟共同 sigma，防止未成形图被放大。"},
        ),
    })
    optional = {
        "stage_two_model": ("MODEL", {"lazy": True, "display_name": "模型二"}),
        "stage_two_positive": ("CONDITIONING", {"lazy": True, "display_name": "模型二正向"}),
        "stage_two_negative": ("CONDITIONING", {"lazy": True, "display_name": "模型二负向"}),
    }
    return {"required": required, "optional": optional}


class _QwenImage21DualSampler:
    HANDOFF_MODE = None
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("最终结果", "第一采结果")
    OUTPUT_TOOLTIPS = (
        "双采完成后的潜变量；关闭模型二时等于第一采完整结果。",
        "第一阶段停在非零 sigma 时的潜变量，尚未完成去噪；不要把它当作最终图像。",
    )
    FUNCTION = "sample"
    CATEGORY = "采样/Qwen Image 2.1"

    def check_lazy_status(self, enable_stage_2, stage_two_model=_DUAL_MISSING,
                          stage_two_positive=_DUAL_MISSING, stage_two_negative=_DUAL_MISSING,
                          **kwargs):
        if enable_stage_2 != "开启":
            return []
        inputs = {"stage_two_model": stage_two_model,
                  "stage_two_positive": stage_two_positive,
                  "stage_two_negative": stage_two_negative}
        if any(value is _DUAL_MISSING for value in inputs.values()):
            return []
        return [name for name, value in inputs.items() if value is None]

    def _sample_impl(self, handoff_value, **kwargs):
        stage_two_inputs = [kwargs.get(name, _DUAL_MISSING) for name in _DUAL_STAGE_TWO_INPUTS]
        provided = [value is not _DUAL_MISSING and value is not None for value in stage_two_inputs]
        requested = kwargs["enable_stage_2"] == "开启"
        if requested and any(provided) and not all(provided):
            raise ValueError("启用模型二时，请同时连接模型二、模型二正向和模型二负向。")
        dual = requested and all(provided)
        if requested and not dual:
            logging.warning("Qwen Image 2.1 双采：模型二未连接完整，本次只运行模型一。")

        model_one = kwargs["stage_one_model"]
        signature = _dual_model_signature(model_one, "模型一")
        model_two = stage_two_inputs[0] if dual else None
        if dual and _dual_model_signature(model_two, "模型二") != signature:
            raise ValueError("两个 Qwen Image 2.1 模型的潜空间格式不同，不能直接交接。")
        if kwargs["memory_mode"] not in _DUAL_MEMORY_MODES:
            raise ValueError("未知显存模式。")
        if kwargs["prefix_cache_mode"] not in _DUAL_PREFIX_CACHE_MODES:
            raise ValueError("未知原生 KV 缓存模式。")
        if not 0.0 <= kwargs["denoise"] <= 1.0:
            raise ValueError("降噪强度必须在 0 到 1 之间。")

        output = kwargs["latent_image"].copy()
        samples = comfy.sample.fix_empty_latent_channels(
            model_one, output["samples"], output.get("downscale_ratio_spacial"),
            output.get("downscale_ratio_temporal"),
        )
        if samples.is_nested or samples.ndim != 4 or int(samples.shape[1]) != signature[1]:
            raise ValueError("输入潜变量必须是与 Qwen Image 2.1 兼容的四维图像 latent。")
        output["samples"] = samples
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        target_width, target_height = _dual_target_latent_size(
            samples, model_two if dual else model_one,
            kwargs["final_width"], kwargs["final_height"],
        )
        upscale_requested = (
            target_width != samples.shape[-1] or target_height != samples.shape[-2]
        )
        if upscale_requested:
            conditionings = [kwargs["stage_one_positive"], kwargs["stage_one_negative"]]
            if dual:
                conditionings.extend(stage_two_inputs[1:])
            _dual_validate_upscale_conditions(
                samples, (target_height, target_width), conditionings, dual,
            )
            if output.get("noise_mask") is not None:
                raise ValueError("带噪声遮罩的潜变量暂不支持节点内部放大。")
            if not dual:
                samples = comfy.utils.common_upscale(
                    samples, target_width, target_height, kwargs["upscale_method"], "disabled",
                )
                output["samples"] = samples
                logging.info(
                    "QwenImage21Dual: model two off; input latent resized before sampling to %dx%d",
                    target_width, target_height,
                )
        if kwargs["denoise"] == 0.0:
            return output, output.copy()

        if dual:
            first_sigmas, second_sigmas, first_count, second_start, boundary = _dual_sigma_pair(
                model_one, model_two, kwargs["stage_one_steps"], kwargs["stage_two_steps"],
                kwargs["stage_one_scheduler"], kwargs["stage_two_scheduler"],
                kwargs["denoise"], self.HANDOFF_MODE, handoff_value,
                kwargs["minimum_upscale_stage_one_steps"] if upscale_requested else 1,
            )
            logging.info(
                "QwenImage21Dual: mode=%s, stage1=%d/%d, stage2=%d/%d, boundary_sigma=%.6f",
                self.HANDOFF_MODE, first_count, kwargs["stage_one_steps"],
                kwargs["stage_two_steps"] - second_start, kwargs["stage_two_steps"], boundary,
            )
            if upscale_requested:
                first_sigmas = first_sigmas.clone()
                first_sigmas[-1] = 0.0
        else:
            first_sigmas = _dual_sigmas(
                model_one, kwargs["stage_one_steps"], kwargs["stage_one_scheduler"],
                kwargs["denoise"],
            )

        if kwargs["memory_mode"] == "省显存":
            comfy.model_management.unload_all_models()
        disable_prefix_cache = dual and kwargs["prefix_cache_mode"] == "安全关闭"
        first_model = _dual_sampling_model(
            model_one, kwargs["stage_one_acceleration"], kwargs["cache_device"],
            disable_prefix_cache,
        )
        noise = comfy.sample.prepare_noise(samples, kwargs["seed"], output.get("batch_index"))
        started = time.perf_counter()
        try:
            first_samples = _dual_sample_stage(
                first_model, kwargs["stage_one_positive"], kwargs["stage_one_negative"],
                kwargs["stage_one_cfg"], kwargs["stage_one_sampler"], first_sigmas,
                samples, noise, output.get("noise_mask"), kwargs["seed"],
            )
        finally:
            logging.info("QwenImage21Dual: stage1 elapsed=%.3fs", time.perf_counter() - started)
            if kwargs["memory_mode"] == "省显存" or (
                dual and _dual_should_offload_first(model_one, model_two, kwargs["memory_mode"])
            ):
                _dual_unload(model_one)
        first_output = output.copy()
        first_output["samples"] = first_samples
        if not dual:
            return first_output.copy(), first_output

        second_model = _dual_sampling_model(
            model_two, kwargs["stage_two_acceleration"], kwargs["cache_device"],
            disable_prefix_cache,
        )
        if upscale_requested:
            second_input = comfy.utils.common_upscale(
                first_samples, target_width, target_height, kwargs["upscale_method"], "disabled",
            )
            second_sigmas, restart_sigma = _dual_upscale_restart_sigmas(
                second_sigmas, kwargs["redraw_strength"],
            )
            second_noise = comfy.sample.prepare_noise(
                second_input, kwargs["seed"], output.get("batch_index"),
            )
            logging.info(
                "QwenImage21Dual: clean latent upscaled to %dx%d, redraw=%.2f, restart_sigma=%.6f",
                target_width, target_height, kwargs["redraw_strength"], restart_sigma,
            )
        else:
            second_input = first_samples
            second_noise = torch.zeros_like(first_samples, device="cpu")
        started = time.perf_counter()
        try:
            final_samples = _dual_sample_stage(
                second_model, stage_two_inputs[1], stage_two_inputs[2],
                kwargs["stage_two_cfg"], kwargs["stage_two_sampler"], second_sigmas,
                second_input, second_noise, output.get("noise_mask"), kwargs["seed"],
            )
        finally:
            logging.info("QwenImage21Dual: stage2 elapsed=%.3fs", time.perf_counter() - started)
            if kwargs["memory_mode"] == "省显存":
                _dual_unload(model_two)
        final_output = output.copy()
        final_output["samples"] = final_samples
        return final_output, first_output


class QwenImage21DualPercentSampler(_QwenImage21DualSampler):
    HANDOFF_MODE = "percent"
    DESCRIPTION = "Qwen Image 2.1 专用百分比双采；两套时间表在同一 sigma 交接，不重复加噪。"

    @classmethod
    def INPUT_TYPES(cls):
        return _dual_inputs("percent")

    def sample(self, handoff_percent, **kwargs):
        return self._sample_impl(handoff_percent, **kwargs)


class QwenImage21DualSNRSampler(_QwenImage21DualSampler):
    HANDOFF_MODE = "snr"
    DESCRIPTION = "Qwen Image 2.1 专用 SNR 双采；按物理噪声强度在同一 sigma 交接。"

    @classmethod
    def INPUT_TYPES(cls):
        return _dual_inputs("snr")

    def sample(self, handoff_snr_db, **kwargs):
        return self._sample_impl(handoff_snr_db, **kwargs)


class QwenImage21SpeedupExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [QwenImage21Speedup]


async def comfy_entrypoint():
    return QwenImage21SpeedupExtension()


NODE_CLASS_MAPPINGS = {
    "QwenImage21Speedup": QwenImage21Speedup,
    "QwenImage21FastSampler": QwenImage21FastSampler,
    "QwenImage21DualPercentSampler": QwenImage21DualPercentSampler,
    "QwenImage21DualSNRSampler": QwenImage21DualSNRSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImage21Speedup": "Qwen Image 2.1 Speedup",
    "QwenImage21FastSampler": "Qwen Image 2.1 加速采样器",
    "QwenImage21DualPercentSampler": "Qwen Image 2.1 双采（百分比）",
    "QwenImage21DualSNRSampler": "Qwen Image 2.1 双采（SNR）",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
