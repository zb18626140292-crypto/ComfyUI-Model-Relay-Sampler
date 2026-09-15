import logging
import time

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview


LOGGER = logging.getLogger(__name__)
CATEGORY = "采样/模型接力"
COMPATIBILITY_MODES = (
    "same_latent_format",
    "same_latent_shape",
    "unsafe",
)
HANDOFF_MODES = ("snr", "percent")
MEMORY_MODES = ("auto_balance", "vram_saver", "speed_first")
# Keep legacy booleans in the server-side combo contract so old API prompts
# still validate. The frontend extension exposes only the two Chinese labels.
STAGE_TOGGLE_OPTIONS = (False, True, "关闭", "开启")
UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")
UPSCALE_RENOISE_FRACTION = 1.0
_MISSING_INPUT = object()
_STAGE_TWO_CONNECTION_INPUTS = (
    "stage_two_model",
    "stage_two_positive",
    "stage_two_negative",
)


def _stage_is_enabled(value):
    """Accept the new Chinese selector and legacy boolean/API values."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"开启", "启用", "true", "1", "on", "enabled", "yes"}
    return bool(value)


def _latent_format(model):
    return model.get_model_object("latent_format")


def _format_signature(model):
    latent_format = _latent_format(model)
    return {
        "type": f"{type(latent_format).__module__}.{type(latent_format).__name__}",
        "channels": int(latent_format.latent_channels),
        "dimensions": int(latent_format.latent_dimensions),
        "spatial_ratio": int(latent_format.spacial_downscale_ratio),
        "temporal_ratio": int(latent_format.temporal_downscale_ratio),
    }


def _check_compatibility(models, mode):
    signatures = [_format_signature(model) for model in models]
    first = signatures[0]

    for index, signature in enumerate(signatures[1:], start=2):
        shape_keys = ("channels", "dimensions", "spatial_ratio", "temporal_ratio")
        shape_mismatches = [
            key for key in shape_keys if signature[key] != first[key]
        ]
        if shape_mismatches and mode != "unsafe":
            details = ", ".join(
                f"{key}: stage 1={first[key]}, stage {index}={signature[key]}"
                for key in shape_mismatches
            )
            raise ValueError(
                "Model Relay Sampler cannot directly hand off between these latent "
                f"spaces ({details}). Use models that share a latent space. The "
                "'unsafe' mode only bypasses this check; it cannot convert latent "
                "channels or VAE spaces."
            )

        if mode == "same_latent_format" and signature["type"] != first["type"]:
            raise ValueError(
                "Model Relay Sampler strict compatibility check failed: "
                f"stage 1 uses {first['type']}, while stage {index} uses "
                f"{signature['type']}. Select 'same_latent_shape' only when you "
                "know the models use compatible VAE latent semantics."
            )

    return signatures


def _master_sigmas(model, steps, scheduler, denoise):
    model_sampling = model.get_model_object("model_sampling")
    if denoise <= 0.0:
        return torch.FloatTensor([])

    if denoise >= 0.9999:
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps)
    else:
        expanded_steps = max(steps, int(steps / denoise))
        sigmas = comfy.samplers.calculate_sigmas(
            model_sampling, scheduler, expanded_steps
        )
        sigmas = sigmas[-(steps + 1) :]

    return sigmas.to(comfy.model_management.intermediate_device())


def _handoff_indices(sigmas, mode, handoffs):
    steps = sigmas.shape[-1] - 1
    if mode == "snr":
        # SNR = 1 / sigma^2, so SNR(dB) = -20 * log10(sigma).
        usable_sigmas = sigmas[1:steps].detach().float().cpu().clamp_min(1e-12)
        if usable_sigmas.numel() == 0:
            raise ValueError("SNR handoff needs at least two total sampling steps.")
        log_sigmas = torch.log(usable_sigmas)
        indices = []
        for snr_db in handoffs:
            target_sigma = 10.0 ** (-float(snr_db) / 20.0)
            nearest = int(
                torch.argmin(torch.abs(log_sigmas - torch.log(torch.tensor(target_sigma))))
            )
            indices.append(nearest + 1)
    else:
        indices = [
            int(round(steps * float(percent) / 100.0)) for percent in handoffs
        ]

    # Clamp collisions caused by short schedules or SNR targets outside the
    # available range, while keeping at least one step in every stage.
    adjusted = []
    for position, index in enumerate(indices):
        minimum = 1 if position == 0 else adjusted[-1] + 1
        maximum = steps - (len(indices) - position)
        adjusted.append(max(minimum, min(index, maximum)))
    indices = adjusted

    boundaries = [0, *indices, steps]
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            raise ValueError(
                "Every relay stage needs at least one sampling step. Increase the "
                "distance between handoff percentages or increase total steps."
            )
    return indices


def _stage_phases(start, end, stage_index, stage_count, transition_steps):
    transition_steps = max(0, int(transition_steps))
    cuts = {start, end}
    if transition_steps and stage_index > 0:
        cuts.add(min(end, start + transition_steps))
    if transition_steps and stage_index < stage_count - 1:
        cuts.add(max(start, end - transition_steps))
    cuts = sorted(cuts)

    phases = []
    for phase_start, phase_end in zip(cuts, cuts[1:]):
        near_entry = (
            transition_steps
            and stage_index > 0
            and phase_start < start + transition_steps
        )
        near_exit = (
            transition_steps
            and stage_index < stage_count - 1
            and phase_end > end - transition_steps
        )
        phases.append((phase_start, phase_end, bool(near_entry or near_exit)))
    return phases


def _should_offload(stages, stage_index, memory_mode, legacy_force_offload):
    if legacy_force_offload or memory_mode == "vram_saver":
        return True
    if memory_mode == "speed_first" or stage_index >= len(stages) - 1:
        return False

    # In balanced mode, proactively offload only when the next full model plus
    # a modest inference reserve would not fit in currently free VRAM.
    try:
        current_model = stages[stage_index]["model"]
        next_model = stages[stage_index + 1]["model"]
        same_base = (
            getattr(current_model, "clone_base_uuid", None) is not None
            and current_model.clone_base_uuid == next_model.clone_base_uuid
        )
        different_patches = (
            getattr(current_model, "patches_uuid", None)
            != getattr(next_model, "patches_uuid", None)
        )
        if same_base:
            # DynamicVRAM normally replaces a loaded clone with
            # detach(unpatch_all=False).  That is cheap when the patch set is
            # identical, but with different LoRAs it can carry the old clone's
            # VBAR/pinned-host state into the next stage.  Force a proper
            # managed unload so the next clone starts with clean patch state.
            return different_patches
        next_size = int(next_model.model_size())
        free_memory = int(
            comfy.model_management.get_free_memory(next_model.load_device)
        )
        required = _model_memory_requirement(next_size)
        return free_memory < required
    except Exception:
        LOGGER.warning(
            "Model Relay: offload decision failed at stage %s -> %s "
            "(memory_mode=%s). Keeping the existing fallback: skip proactive "
            "offload for this handoff; ComfyUI still manages model memory. "
            "See traceback below for the cause.",
            stage_index + 1,
            stage_index + 2,
            memory_mode,
            exc_info=True,
        )
        return False


def _model_memory_requirement(model_or_size):
    """Estimate full model residency plus a modest inference reserve."""
    size = (
        int(model_or_size.model_size())
        if hasattr(model_or_size, "model_size")
        else int(model_or_size)
    )
    return int(size * 1.10) + 768 * 1024 * 1024


def _prepare_first_stage_vram(memory_mode, first_model):
    """Release stale GPU models before stage 1 in the strict VRAM mode.

    Conditioning is already encoded when the sampler executes, so keeping a
    text encoder resident here only reduces the space available to the first
    diffusion model.  ComfyUI's dynamic-model loading intentionally avoids
    evicting other dynamic models; on a 16 GB card that can force several GB of
    stage-1 weights through CPU offload on every denoising step.

    ``vram_saver`` already promises cold, deterministic model residency and
    unloads the final stage, so a preflight unload does not weaken its cache
    contract. Balanced mode clears a resident same-base clone whose LoRA patch
    UUID differs, or stale models from the previous queue when the first model
    plus inference reserve cannot fit; speed-first retains its cache behavior.
    """
    if memory_mode == "speed_first":
        return

    loaded = comfy.model_management.loaded_models()
    conflicting_clones = []
    reason = None
    if memory_mode == "auto_balance":
        first_base = getattr(first_model, "clone_base_uuid", None)
        first_patches = getattr(first_model, "patches_uuid", None)
        conflicting_clones = [
            model
            for model in loaded
            if first_base is not None
            and getattr(model, "clone_base_uuid", None) == first_base
            and getattr(model, "patches_uuid", None) != first_patches
        ]

    device = comfy.model_management.get_torch_device()
    free_before = int(comfy.model_management.get_free_memory(device))
    loaded_before = len(loaded)
    if memory_mode == "auto_balance":
        if conflicting_clones:
            reason = f"stale-lora-clones:{len(conflicting_clones)}"
        else:
            try:
                required = _model_memory_requirement(first_model)
            except Exception:
                return
            if loaded_before == 0 or free_before >= required:
                return
            reason = (
                f"memory-pressure:{free_before / (1024 * 1024):.1f}<"
                f"{required / (1024 * 1024):.1f}MB"
            )

    started = time.perf_counter()
    if memory_mode == "vram_saver":
        comfy.model_management.unload_all_models()
        reason = "strict-mode"
    elif conflicting_clones:
        comfy.model_management.unload_model_and_clones(
            first_model,
            unload_additional_models=False,
            all_devices=True,
        )
    else:
        # Conditioning has already been encoded when this sampler executes.
        # Under real pressure, evict stale text encoders, VAEs, and the prior
        # queue's final diffusion model before DynamicVRAM starts paging the
        # first stage through host memory.
        comfy.model_management.unload_all_models()
    free_after = int(comfy.model_management.get_free_memory(device))
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "Model Relay stage-1 VRAM preflight: reason=%s, loaded=%d, "
        "free=%.1f -> %.1f MB, elapsed=%.3fs (memory_mode=%s)",
        reason,
        loaded_before,
        free_before / (1024 * 1024),
        free_after / (1024 * 1024),
        elapsed,
        memory_mode,
    )


def _empty_noise_like(latent_image):
    if latent_image.is_nested:
        import comfy.nested_tensor

        return comfy.nested_tensor.NestedTensor(
            [torch.zeros_like(item, device="cpu") for item in latent_image.unbind()]
        )
    return torch.zeros_like(latent_image, device="cpu")


def _log_stage_timing(stage_number, model, elapsed, will_offload):
    """Log enough DynamicVRAM state to distinguish compute from handoff stalls."""
    try:
        device = model.load_device
        free_mb = comfy.model_management.get_free_memory(device) / (1024 * 1024)
        pinned_mb = model.pinned_memory_size() / (1024 * 1024)
        loaded_ram_mb = model.loaded_ram_size() / (1024 * 1024)
        LOGGER.info(
            "Model Relay stage %d timing: %.3fs, free=%.1f MB, "
            "dynamic_pinned=%.1f MB, dynamic_ram=%.1f MB, offload=%s",
            stage_number,
            elapsed,
            free_mb,
            pinned_mb,
            loaded_ram_mb,
            will_offload,
        )
    except Exception:
        LOGGER.info(
            "Model Relay stage %d timing: %.3fs, offload=%s",
            stage_number,
            elapsed,
            will_offload,
        )


def _random_noise(latent, seed):
    batch_indices = latent.get("batch_index")
    return comfy.sample.prepare_noise(latent["samples"], seed, batch_indices)


def _target_latent_size(latent_image, model, final_width, final_height):
    if latent_image.is_nested:
        raise ValueError("模型接力采样器暂不支持嵌套 latent 的内部尺寸放大。")
    if latent_image.ndim < 4:
        raise ValueError(
            "内部尺寸放大需要图像 latent，形状应至少包含批次、通道、高度和宽度。"
        )

    latent_format = _latent_format(model)
    downscale = int(latent_format.spacial_downscale_ratio)
    current_pixel_width = int(latent_image.shape[-1]) * downscale
    current_pixel_height = int(latent_image.shape[-2]) * downscale

    if final_width <= 0 and final_height <= 0:
        return int(latent_image.shape[-1]), int(latent_image.shape[-2])
    if final_width <= 0:
        final_width = round(
            current_pixel_width * float(final_height) / current_pixel_height
        )
    if final_height <= 0:
        final_height = round(
            current_pixel_height * float(final_width) / current_pixel_width
        )

    target_width = max(1, round(float(final_width) / downscale))
    target_height = max(1, round(float(final_height) / downscale))
    return target_width, target_height


def _upscale_latent_if_needed(
    latent_image,
    model,
    final_width,
    final_height,
    upscale_method,
):
    target_width, target_height = _target_latent_size(
        latent_image, model, final_width, final_height
    )
    if (
        target_width == latent_image.shape[-1]
        and target_height == latent_image.shape[-2]
    ):
        return latent_image, False

    upscaled = comfy.utils.common_upscale(
        latent_image,
        target_width,
        target_height,
        upscale_method,
        "disabled",
    )
    return upscaled, True


def _upscale_restart_sigma_scale(
    model, handoff_sigma, upscale_denoise_strength=UPSCALE_RENOISE_FRACTION
):
    handoff_sigma = float(handoff_sigma)
    if handoff_sigma <= 0.0:
        return 1.0, handoff_sigma

    model_sampling = model.get_model_object("model_sampling")
    sigma_max = float(model_sampling.sigma_max)
    safe_sigma = max(
        0.0, sigma_max * max(0.0, min(1.0, float(upscale_denoise_strength)))
    )
    restart_sigma = min(handoff_sigma, safe_sigma)
    return restart_sigma / handoff_sigma, restart_sigma


def _nearest_sigma_index(sigmas, target_sigma, minimum=1, maximum=None):
    maximum = len(sigmas) - 2 if maximum is None else int(maximum)
    minimum = max(1, int(minimum))
    maximum = min(len(sigmas) - 2, maximum)
    if maximum < minimum:
        raise ValueError("采样时间表太短，无法保留有效的模型交接区间。")

    candidates = sigmas[minimum : maximum + 1].detach().float().cpu()
    target = torch.tensor(float(target_sigma)).clamp_min(1e-12)
    distances = torch.abs(torch.log(candidates.clamp_min(1e-12)) - torch.log(target))
    return minimum + int(torch.argmin(distances))


def _dual_sigma_pair(
    model_1,
    model_2,
    steps_1,
    scheduler_1,
    steps_2,
    scheduler_2,
    denoise,
    handoff_mode,
    handoff_value,
    minimum_stage_1_steps=1,
):
    sigmas_1 = _master_sigmas(model_1, steps_1, scheduler_1, denoise)
    sigmas_2 = _master_sigmas(model_2, steps_2, scheduler_2, denoise)
    if len(sigmas_1) < 3 or len(sigmas_2) < 3:
        raise ValueError("双采接力要求第一采和第二采都至少设置 2 个计划步数。")

    if handoff_mode == "percent":
        stage_2_start = int(round(steps_2 * float(handoff_value) / 100.0))
        stage_2_start = max(1, min(stage_2_start, steps_2 - 1))
    else:
        target_sigma = 10.0 ** (-float(handoff_value) / 20.0)
        stage_2_start = _nearest_sigma_index(sigmas_2, target_sigma)

    boundary_sigma = float(sigmas_2[stage_2_start])
    stage_1_end = _nearest_sigma_index(sigmas_1, boundary_sigma)

    protected = False
    minimum_stage_1_steps = max(
        1, min(int(minimum_stage_1_steps), len(sigmas_1) - 2)
    )
    if stage_1_end < minimum_stage_1_steps:
        # Very early SNR values (notably 0 dB on Krea's 8-step Simple
        # schedule) otherwise send a one-step, unfinished x0 estimate into
        # the internal upscaler. Move both schedules to the nearest safe
        # common boundary instead of producing a melted high-resolution image.
        safe_sigma = float(sigmas_1[minimum_stage_1_steps])
        stage_2_start = _nearest_sigma_index(sigmas_2, safe_sigma)
        boundary_sigma = float(sigmas_2[stage_2_start])
        stage_1_end = _nearest_sigma_index(
            sigmas_1, boundary_sigma, minimum=minimum_stage_1_steps
        )
        protected = True

    boundary = sigmas_2[stage_2_start].clone()
    stage_1_sigmas = torch.cat(
        (sigmas_1[:stage_1_end], boundary.reshape(1))
    )
    stage_2_sigmas = sigmas_2[stage_2_start:].clone()
    stage_2_sigmas[0] = boundary
    return (
        stage_1_sigmas,
        stage_2_sigmas,
        stage_1_end,
        stage_2_start,
        float(boundary),
        protected,
    )


def _sample_stage(
    model,
    positive,
    negative,
    cfg,
    sampler_name,
    sigmas,
    latent_image,
    noise,
    noise_mask,
    seed,
):
    if sigmas.shape[-1] <= 1:
        return latent_image

    sampler = comfy.samplers.sampler_object(sampler_name)
    stage_steps = sigmas.shape[-1] - 1
    callback = latent_preview.prepare_callback(model, stage_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    return comfy.sample.sample_custom(
        model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )


def _run_relay(
    stages,
    latent,
    seed,
    steps,
    scheduler,
    denoise,
    handoff_mode,
    handoffs,
    compatibility,
    transition_steps,
    memory_mode,
    force_offload_after_stage,
    final_width,
    final_height,
    upscale_method,
    upscale_denoise_strength=UPSCALE_RENOISE_FRACTION,
):
    models = [stage["model"] for stage in stages]
    signatures = _check_compatibility(models, compatibility)

    output = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(
        models[0],
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    output["samples"] = latent_image

    actual_channels = None if latent_image.is_nested else int(latent_image.shape[1])
    expected_channels = signatures[0]["channels"]
    if actual_channels is not None and actual_channels != expected_channels:
        raise ValueError(
            f"Input latent has {actual_channels} channels, but stage 1 expects "
            f"{expected_channels} channels."
        )

    if denoise <= 0.0:
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        return output, [output.copy() for _ in stages]

    sigmas = _master_sigmas(models[0], steps, scheduler, denoise)
    runtime_sigmas = sigmas.clone()
    indices = _handoff_indices(sigmas, handoff_mode, handoffs)
    boundaries = [0, *indices, steps]
    noise_mask = output.get("noise_mask")
    first_noise = _random_noise(output, seed)
    noise_has_been_added = False
    stage_outputs = []
    upscale_requested = False
    if len(stages) > 1 and (final_width > 0 or final_height > 0):
        target_width, target_height = _target_latent_size(
            latent_image, models[1], final_width, final_height
        )
        upscale_requested = (
            target_width != latent_image.shape[-1]
            or target_height != latent_image.shape[-2]
        )

    _prepare_first_stage_vram(memory_mode, models[0])

    for stage_index, (stage, start, end) in enumerate(
        zip(stages, boundaries, boundaries[1:])
    ):
        LOGGER.info(
            "Model Relay stage %d/%d: steps %d..%d, sampler=%s, cfg=%s, "
            "handoff_mode=%s",
            stage_index + 1,
            len(stages),
            start,
            end,
            stage["sampler"],
            stage["cfg"],
            handoff_mode,
        )
        try:
            phases = _stage_phases(
                start, end, stage_index, len(stages), transition_steps
            )
            for phase_start, phase_end, use_euler in phases:
                phase_sigmas = runtime_sigmas[
                    phase_start : phase_end + 1
                ].clone()
                if (
                    upscale_requested
                    and stage_index == 0
                    and phase_end == end
                ):
                    # Spatial interpolation is reliable on a clean latent, but
                    # interpolating the noisy sampler state destroys the
                    # high-resolution noise distribution. Resolve stage 1 to
                    # x0, then restart the remaining schedule with controlled
                    # target-resolution noise below.
                    phase_sigmas[-1] = 0.0
                noise = (
                    first_noise
                    if not noise_has_been_added
                    else _empty_noise_like(latent_image)
                )
                sampler_name = "euler" if use_euler else stage["sampler"]
                latent_image = _sample_stage(
                    stage["model"],
                    stage["positive"],
                    stage["negative"],
                    stage["cfg"],
                    sampler_name,
                    phase_sigmas,
                    latent_image,
                    noise,
                    noise_mask,
                    seed,
                )
                noise_has_been_added = True
        finally:
            if _should_offload(
                stages, stage_index, memory_mode, force_offload_after_stage
            ):
                LOGGER.info(
                    "Model Relay stage %d/%d complete; offloading model "
                    "(memory_mode=%s)",
                    stage_index + 1,
                    len(stages),
                    memory_mode,
                )
                comfy.model_management.unload_model_and_clones(
                    stage["model"],
                    unload_additional_models=True,
                    all_devices=True,
                )

        stage_output = output.copy()
        stage_output.pop("downscale_ratio_spacial", None)
        stage_output.pop("downscale_ratio_temporal", None)
        stage_output["samples"] = latent_image
        stage_outputs.append(stage_output)

        if upscale_requested and stage_index == 0:
            latent_image, did_upscale = _upscale_latent_if_needed(
                latent_image,
                models[1],
                final_width,
                final_height,
                upscale_method,
            )
            if did_upscale:
                scale, restart_sigma = _upscale_restart_sigma_scale(
                    models[1], sigmas[end], upscale_denoise_strength
                )
                runtime_sigmas[end:] = sigmas[end:] * scale
                runtime_sigmas[-1] = 0.0
                upscaled_latent = output.copy()
                upscaled_latent["samples"] = latent_image
                first_noise = _random_noise(upscaled_latent, seed)
                noise_has_been_added = False
                LOGGER.info(
                    "Model Relay internal latent upscale: %dx%d latent pixels, "
                    "method=%s; controlled restart sigma %.6f (original %.6f), "
                    "preserving %d remaining steps",
                    latent_image.shape[-1],
                    latent_image.shape[-2],
                    upscale_method,
                    restart_sigma,
                    float(sigmas[end]),
                    steps - end,
                )

    return stage_outputs[-1], stage_outputs


def _run_dual_independent(
    stages,
    latent,
    seed,
    steps_1,
    scheduler_1,
    steps_2,
    scheduler_2,
    denoise,
    handoff_mode,
    handoff_value,
    compatibility,
    memory_mode,
    force_offload_after_stage,
    final_width,
    final_height,
    upscale_method,
    upscale_denoise_strength,
    minimum_upscale_stage_1_steps,
):
    if len(stages) == 1:
        single_latent = latent
        if final_width > 0 or final_height > 0:
            resized_samples, did_resize = _upscale_latent_if_needed(
                latent["samples"],
                stages[0]["model"],
                final_width,
                final_height,
                upscale_method,
            )
            if did_resize:
                single_latent = latent.copy()
                single_latent["samples"] = resized_samples
                noise_mask = single_latent.get("noise_mask")
                if torch.is_tensor(noise_mask) and noise_mask.ndim in (3, 4):
                    mask_was_3d = noise_mask.ndim == 3
                    if mask_was_3d:
                        noise_mask = noise_mask.unsqueeze(1)
                    noise_mask = comfy.utils.common_upscale(
                        noise_mask,
                        resized_samples.shape[-1],
                        resized_samples.shape[-2],
                        "bilinear",
                        "disabled",
                    )
                    single_latent["noise_mask"] = (
                        noise_mask[:, 0] if mask_was_3d else noise_mask
                    )
                LOGGER.info(
                    "Model Relay single-stage input latent resize before "
                    "sampling: %dx%d latent pixels, method=%s",
                    resized_samples.shape[-1],
                    resized_samples.shape[-2],
                    upscale_method,
                )

        return _run_relay(
            stages,
            single_latent,
            seed,
            steps_1,
            scheduler_1,
            denoise,
            "percent",
            [],
            compatibility,
            0,
            memory_mode,
            force_offload_after_stage,
            0,
            0,
            upscale_method,
            upscale_denoise_strength,
        )

    models = [stage["model"] for stage in stages]
    signatures = _check_compatibility(models, compatibility)
    output = latent.copy()
    latent_image = comfy.sample.fix_empty_latent_channels(
        models[0],
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    output["samples"] = latent_image

    actual_channels = None if latent_image.is_nested else int(latent_image.shape[1])
    if actual_channels is not None and actual_channels != signatures[0]["channels"]:
        raise ValueError(
            f"输入 latent 有 {actual_channels} 个通道，但第一采模型需要 "
            f"{signatures[0]['channels']} 个通道。"
        )

    if denoise <= 0.0:
        output.pop("downscale_ratio_spacial", None)
        output.pop("downscale_ratio_temporal", None)
        return output, [output.copy(), output.copy()]

    upscale_requested = False
    if final_width > 0 or final_height > 0:
        target_width, target_height = _target_latent_size(
            latent_image, models[1], final_width, final_height
        )
        upscale_requested = (
            target_width != latent_image.shape[-1]
            or target_height != latent_image.shape[-2]
        )

    minimum_stage_1_steps = (
        minimum_upscale_stage_1_steps if upscale_requested else 1
    )
    (
        stage_1_sigmas,
        stage_2_sigmas,
        stage_1_end,
        stage_2_start,
        boundary_sigma,
        protected,
    ) = _dual_sigma_pair(
        models[0],
        models[1],
        steps_1,
        scheduler_1,
        steps_2,
        scheduler_2,
        denoise,
        handoff_mode,
        handoff_value,
        minimum_stage_1_steps,
    )

    stage_1_run_sigmas = stage_1_sigmas.clone()
    if upscale_requested:
        stage_1_run_sigmas[-1] = 0.0

    LOGGER.info(
        "Model Relay independent dual: mode=%s, requested=%s, "
        "stage1=%d/%d steps, stage2_start=%d/%d (%d remaining), "
        "boundary_sigma=%.6f, upscale=%s, early_handoff_protected=%s",
        handoff_mode,
        handoff_value,
        stage_1_end,
        steps_1,
        stage_2_start,
        steps_2,
        len(stage_2_sigmas) - 1,
        boundary_sigma,
        upscale_requested,
        protected,
    )

    _prepare_first_stage_vram(memory_mode, models[0])

    noise_mask = output.get("noise_mask")
    stage_1_started = time.perf_counter()
    try:
        latent_image = _sample_stage(
            models[0],
            stages[0]["positive"],
            stages[0]["negative"],
            stages[0]["cfg"],
            stages[0]["sampler"],
            stage_1_run_sigmas,
            latent_image,
            _random_noise(output, seed),
            noise_mask,
            seed,
        )
    finally:
        stage_1_offload = _should_offload(
            stages, 0, memory_mode, force_offload_after_stage
        )
        _log_stage_timing(
            1,
            models[0],
            time.perf_counter() - stage_1_started,
            stage_1_offload,
        )
        if stage_1_offload:
            comfy.model_management.unload_model_and_clones(
                models[0],
                unload_additional_models=True,
                all_devices=True,
            )

    stage_1_output = output.copy()
    stage_1_output.pop("downscale_ratio_spacial", None)
    stage_1_output.pop("downscale_ratio_temporal", None)
    stage_1_output["samples"] = latent_image

    stage_2_noise = _empty_noise_like(latent_image)
    if upscale_requested:
        latent_image, did_upscale = _upscale_latent_if_needed(
            latent_image,
            models[1],
            final_width,
            final_height,
            upscale_method,
        )
        if did_upscale:
            scale, restart_sigma = _upscale_restart_sigma_scale(
                models[1], boundary_sigma, upscale_denoise_strength
            )
            stage_2_sigmas = stage_2_sigmas * scale
            stage_2_sigmas[-1] = 0.0
            upscaled = output.copy()
            upscaled["samples"] = latent_image
            stage_2_noise = _random_noise(upscaled, seed)
            # A spatial noise mask from the base resolution cannot be reused
            # at the enlarged resolution without explicit mask resampling.
            if noise_mask is not None and noise_mask.shape[-2:] != latent_image.shape[-2:]:
                noise_mask = None
            LOGGER.info(
                "Model Relay upscale restart: strength=%.3f, sigma %.6f -> %.6f, "
                "%d high-resolution steps",
                upscale_denoise_strength,
                boundary_sigma,
                restart_sigma,
                len(stage_2_sigmas) - 1,
            )

    stage_2_started = time.perf_counter()
    try:
        latent_image = _sample_stage(
            models[1],
            stages[1]["positive"],
            stages[1]["negative"],
            stages[1]["cfg"],
            stages[1]["sampler"],
            stage_2_sigmas,
            latent_image,
            stage_2_noise,
            noise_mask,
            seed,
        )
    finally:
        stage_2_offload = _should_offload(
            stages, 1, memory_mode, force_offload_after_stage
        )
        _log_stage_timing(
            2,
            models[1],
            time.perf_counter() - stage_2_started,
            stage_2_offload,
        )
        if stage_2_offload:
            comfy.model_management.unload_model_and_clones(
                models[1],
                unload_additional_models=True,
                all_devices=True,
            )

    final_output = output.copy()
    final_output.pop("downscale_ratio_spacial", None)
    final_output.pop("downscale_ratio_temporal", None)
    final_output["samples"] = latent_image
    return final_output, [stage_1_output, final_output.copy()]


def _common_inputs(
    default_steps,
    default_scheduler="karras",
    default_handoff_mode="snr",
    default_transition_steps=1,
):
    return {
        "seed": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 0xFFFFFFFFFFFFFFFF,
                "control_after_generate": True,
                "display_name": "随机种子",
                "tooltip": "控制初始噪声。种子和其他参数相同时可复现结果。",
            },
        ),
        "steps": (
            "INT",
            {
                "default": default_steps,
                "min": 2,
                "max": 10000,
                "display_name": "总采样步数",
                "tooltip": (
                    "整个接力流程共用的总步数，不是每个模型分别执行这么多步。"
                    "Krea 2 Turbo 推荐总计 8 步。"
                ),
            },
        ),
        "scheduler": (
            comfy.samplers.KSampler.SCHEDULERS,
            {
                "default": default_scheduler,
                "display_name": "噪声调度器",
                "tooltip": (
                    "决定各步噪声强度的分布。Krea 2 Turbo 推荐 simple；"
                    "其他模型可按模型说明选择 karras、beta 等。"
                ),
            },
        ),
        "denoise": (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "display_name": "降噪强度",
                "tooltip": (
                    "1.0 表示完整去噪；较低值更多保留输入 latent 的原始结构，"
                    "常用于图生图。"
                ),
            },
        ),
        "compatibility": (
            COMPATIBILITY_MODES,
            {
                "default": "same_latent_format",
                "display_name": "潜空间兼容检查",
                "tooltip": (
                    "same_latent_format：严格要求同类潜空间，最安全；"
                    "same_latent_shape：仅检查通道、维度和下采样比例；"
                    "unsafe：跳过检查，不会自动转换不兼容潜空间。"
                ),
            },
        ),
        "handoff_mode": (
            HANDOFF_MODES,
            {
                "default": default_handoff_mode,
                "display_name": "模型交接方式",
                "tooltip": (
                    "snr：按实际噪声强度交接，改变步数或调度器时更稳定；"
                    "percent：按总步数百分比交接。Krea 2 Turbo 默认使用 percent。"
                ),
            },
        ),
        "transition_steps": (
            "INT",
            {
                "default": default_transition_steps,
                "min": 0,
                "max": 8,
                "display_name": "Euler 过渡步数",
                "tooltip": (
                    "交接点两侧自动使用 Euler 的步数，降低多步采样器历史重置"
                    "造成的接缝。0 表示关闭；两段本来就是 Euler 时无需开启。"
                ),
            },
        ),
        "memory_mode": (
            MEMORY_MODES,
            {
                "default": "vram_saver",
                "display_name": "显存管理模式",
                "tooltip": (
                    "auto_balance：按显存压力清理跨轮残留，并在下一模型可能"
                    "放不下时提前卸载；"
                    "vram_saver：每采结束必定卸载；"
                    "speed_first：尽量保留模型缓存以提高速度。"
                ),
            },
        ),
        "latent_image": (
            "LATENT",
            {
                "display_name": "输入潜变量",
                "tooltip": "需要进行接力采样的初始 latent，通常来自空潜变量或 VAE 编码。",
            },
        ),
    }


def _stage_inputs(
    number,
    default_cfg=7.0,
    default_sampler="dpmpp_2m",
    lazy=False,
):
    stage_name = f"第{number}采"
    lazy_config = {"lazy": True} if lazy else {}
    return {
        f"model_{number}": (
            "MODEL",
            {
                **lazy_config,
                "display_name": f"{stage_name}模型",
                "tooltip": f"{stage_name}负责处理其对应噪声区间的扩散模型。",
            },
        ),
        f"positive_{number}": (
            "CONDITIONING",
            {
                **lazy_config,
                "display_name": f"{stage_name}正面条件",
                "tooltip": f"传给{stage_name}模型的正面提示词条件。",
            },
        ),
        f"negative_{number}": (
            "CONDITIONING",
            {
                **lazy_config,
                "display_name": f"{stage_name}负面条件",
                "tooltip": (
                    f"传给{stage_name}模型的负面提示词条件。CFG=1 时通常不会"
                    "参与 classifier-free guidance。"
                ),
            },
        ),
        f"cfg_{number}": (
            "FLOAT",
            {
                "default": default_cfg,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "round": 0.01,
                "display_name": f"{stage_name} CFG",
                "tooltip": (
                    f"{stage_name}的提示词引导强度。Krea 2 Turbo 推荐 1.0；"
                    "普通非蒸馏模型通常使用更高值。"
                ),
            },
        ),
        f"sampler_{number}": (
            comfy.samplers.KSampler.SAMPLERS,
            {
                "default": default_sampler,
                "display_name": f"{stage_name}采样器",
                "tooltip": (
                    f"{stage_name}使用的采样算法。Krea 2 Turbo 推荐 euler；"
                    "DPM++ 等多步算法在模型交接处可能需要 Euler 过渡。"
                ),
            },
        ),
    }


def _upscale_inputs():
    return {
        "final_width": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 16384,
                "step": 16,
                "display_name": "最终宽度",
                "tooltip": (
                    "设为 0 表示保持输入宽度或按最终高度自动计算。启用第二采时，"
                    "第一采会在输入小尺寸得到干净 latent，再放大到目标尺寸；"
                    "第二采使用受控噪声继续精修。"
                ),
            },
        ),
        "final_height": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 16384,
                "step": 16,
                "display_name": "最终高度",
                "tooltip": (
                    "设为 0 表示保持输入高度或按最终宽度自动计算。宽高都为 0 "
                    "时关闭内部放大；第二采关闭时该设置自动失效。"
                ),
            },
        ),
        "upscale_method": (
            UPSCALE_METHODS,
            {
                "default": "bislerp",
                "display_name": "潜变量放大算法",
                "tooltip": (
                    "第一采到达交接点后放大干净 latent 的插值算法。节点会在"
                    "目标尺寸生成正确分布的新噪声，并自动限制重绘强度，避免"
                    "彩色噪点、双脸和构图重生。bislerp 通常最适合扩散 latent；"
                    "支持普通 4D 和带单帧维度的 5D 图像 latent。"
                ),
            },
        ),
        "redraw_strength": (
            "FLOAT",
            {
                "default": UPSCALE_RENOISE_FRACTION,
                "min": 0.10,
                "max": 1.00,
                "step": 0.05,
                "round": 0.01,
                "display_name": "重绘强度",
                "tooltip": (
                    "只在内部放大时生效。数值越高，二采重绘和补细节越强，"
                    "但不同模型之间更容易出现手指双边、轮廓重影或构图漂移；"
                    "数值越低越保留一采结构，但可能无法修正一采遗留的黏连。"
                    "默认 1.0 表示完整使用交接 sigma，仍会被交接点自动限幅，"
                    "不会从纯噪声重新生成。"
                ),
            },
        ),
    }


class ModelRelaySamplerDual:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        required.update(_stage_inputs(1, 1.0, "euler"))
        required.update(_stage_inputs(2, 1.0, "euler", lazy=True))
        required.update(
            _common_inputs(
                8,
                default_scheduler="simple",
                default_handoff_mode="percent",
                default_transition_steps=0,
            )
        )
        required["handoff_snr_db"] = (
            "FLOAT",
            {
                "default": 6.0,
                "min": -40.0,
                "max": 40.0,
                "step": 0.5,
                "display_name": "SNR 交接值（dB）",
                "tooltip": (
                    "仅在交接方式选择 snr 时生效。数值越小越早交接，"
                    "数值越大越晚交接；6 dB 约对应 sigma=0.5。"
                ),
            },
        )
        required["handoff_percent"] = (
            "FLOAT",
            {
                "default": 50.0,
                "min": 1.0,
                "max": 99.0,
                "step": 1.0,
                "display_name": "交接位置（百分比）",
                "tooltip": (
                    "仅在交接方式选择 percent 时生效。50% 表示第一、第二模型"
                    "各负责约一半总步数；Krea 2 Turbo 8 步时约为 4+4。"
                ),
            },
        )
        required["enable_stage_2"] = (
            "BOOLEAN",
            {
                "default": True,
                "display_name": "启用第二采",
                "label_on": "启用",
                "label_off": "关闭",
                "tooltip": (
                    "关闭后节点退化为单模型采样：第二采模型和条件不会执行，"
                    "所有模型交接参数自动失效。"
                ),
            },
        )
        required.update(_upscale_inputs())
        return {"required": required}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("最终结果", "第一采结果")
    OUTPUT_TOOLTIPS = (
        "完成全部模型接力后的最终 latent。",
        "第一模型完成低分辨率阶段后的干净 latent，可直接解码预览和对比。",
    )
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "两个模型共享同一条 sigma 时间轴。第一采添加噪声，第二采从交接点"
        "继续去噪，不会重复加噪；可关闭第二采退化为单采。"
    )

    def check_lazy_status(
        self,
        enable_stage_2,
        model_2=None,
        positive_2=None,
        negative_2=None,
        **kwargs,
    ):
        if not enable_stage_2:
            return []
        lazy_inputs = {
            "model_2": model_2,
            "positive_2": positive_2,
            "negative_2": negative_2,
        }
        return [name for name, value in lazy_inputs.items() if value is None]

    def sample(
        self,
        model_1,
        positive_1,
        negative_1,
        cfg_1,
        sampler_1,
        model_2,
        positive_2,
        negative_2,
        cfg_2,
        sampler_2,
        seed,
        steps,
        scheduler,
        denoise,
        compatibility,
        handoff_mode,
        transition_steps,
        memory_mode,
        latent_image,
        handoff_snr_db,
        handoff_percent,
        enable_stage_2,
        final_width,
        final_height,
        upscale_method,
        redraw_strength,
    ):
        stages = [
            {
                "model": model_1,
                "positive": positive_1,
                "negative": negative_1,
                "cfg": cfg_1,
                "sampler": sampler_1,
            },
        ]
        if enable_stage_2:
            stages.append(
                {
                    "model": model_2,
                    "positive": positive_2,
                    "negative": negative_2,
                    "cfg": cfg_2,
                    "sampler": sampler_2,
                }
            )
        handoff = (
            handoff_snr_db if handoff_mode == "snr" else handoff_percent
        )
        handoffs = [handoff] if enable_stage_2 else []
        output, stage_outputs = _run_relay(
            stages,
            latent_image,
            seed,
            steps,
            scheduler,
            denoise,
            handoff_mode,
            handoffs,
            compatibility,
            transition_steps,
            memory_mode,
            False,
            final_width,
            final_height,
            upscale_method,
            redraw_strength,
        )
        return (output, stage_outputs[0])


def _independent_stage_inputs(stage_key, stage_label, lazy=False):
    lazy_config = {"lazy": True} if lazy else {}
    return {
        f"{stage_key}_model": (
            "MODEL",
            {
                **lazy_config,
                "display_name": stage_label,
                "tooltip": f"{stage_label}负责其对应噪声区间的扩散模型。",
            },
        ),
        f"{stage_key}_positive": (
            "CONDITIONING",
            {
                **lazy_config,
                "display_name": f"{stage_label}正向",
                "tooltip": f"传给{stage_label}的正面提示词条件。",
            },
        ),
        f"{stage_key}_negative": (
            "CONDITIONING",
            {
                **lazy_config,
                "display_name": f"{stage_label}负向",
                "tooltip": f"传给{stage_label}的负面提示词条件。",
            },
        ),
        f"{stage_key}_cfg": (
            "FLOAT",
            {
                "default": 1.0,
                "min": 0.0,
                "max": 100.0,
                "step": 0.1,
                "round": 0.01,
                "display_name": f"{stage_label} CFG",
                "tooltip": f"{stage_label}提示词引导强度；Krea 2 Turbo 推荐 1.0。",
            },
        ),
        f"{stage_key}_sampler": (
            comfy.samplers.KSampler.SAMPLERS,
            {
                "default": "euler",
                "display_name": f"{stage_label}采样器",
                "tooltip": f"{stage_label}采样算法；Krea 2 Turbo 推荐 Euler。",
            },
        ),
    }


def _independent_dual_inputs(handoff_mode):
    required = {}
    optional = {}
    required.update(_independent_stage_inputs("stage_one", "模型一"))
    stage_one_sampler = required.pop("stage_one_sampler")
    required.update(
        {
            "stage_one_steps": (
                "INT",
                {
                    "default": 8,
                    "min": 2,
                    "max": 10000,
                    "display_name": "模型一步数",
                    "tooltip": (
                        "用于生成第一模型自己的完整 sigma 时间表。交接后只执行"
                        "到共同边界为止；Krea 2 Turbo 默认 8。"
                    ),
                },
            ),
            "stage_one_sampler": stage_one_sampler,
            "stage_one_scheduler": (
                comfy.samplers.KSampler.SCHEDULERS,
                {
                    "default": "simple",
                    "display_name": "模型一调度器",
                    "tooltip": "第一模型的噪声时间表；Krea 2 Turbo 推荐 simple。",
                },
            ),
        }
    )
    stage_two_inputs = _independent_stage_inputs("stage_two", "模型二", lazy=True)
    for input_name in _STAGE_TWO_CONNECTION_INPUTS:
        optional[input_name] = stage_two_inputs.pop(input_name)
        optional[input_name][1]["tooltip"] += (
            " 模型二、模型二正向、模型二负向任一未连接时，节点自动只运行模型一。"
        )
    required.update(stage_two_inputs)
    stage_two_sampler = required.pop("stage_two_sampler")
    required.update(
        {
            "stage_two_steps": (
                "INT",
                {
                    "default": 12,
                    "min": 2,
                    "max": 10000,
                    "display_name": "模型二步数",
                    "tooltip": (
                        "用于生成第二模型自己的时间表。默认 12 可在 50% 交接后"
                        "留下约 6 个高分辨率精修步，比旧版 4 步更容易消除模糊"
                        "和双边缘；这不是先完整跑 12 步再重跑。"
                    ),
                },
            ),
            "stage_two_sampler": stage_two_sampler,
            "stage_two_scheduler": (
                comfy.samplers.KSampler.SCHEDULERS,
                {
                    "default": "simple",
                    "display_name": "模型二调度器",
                    "tooltip": "第二模型的噪声时间表；Krea 2 Turbo 推荐 simple。",
                },
            ),
        }
    )
    required.update(
        {
            "seed": (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "display_name": "随机种子",
                    "tooltip": "控制两阶段使用的可复现噪声。",
                },
            ),
            "denoise": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display_name": "降噪强度",
                    "tooltip": "1.0 为完整去噪；图生图时可降低。",
                },
            ),
            "compatibility": (
                COMPATIBILITY_MODES,
                {
                    "default": "same_latent_format",
                    "display_name": "潜空间兼容检查",
                    "tooltip": (
                        "默认严格检查两个模型是否共享同类 latent。不同 VAE "
                        "潜空间不能仅靠该节点直接交接。"
                    ),
                },
            ),
            "memory_mode": (
                MEMORY_MODES,
                {
                    "default": "vram_saver",
                    "display_name": "显存管理模式",
                    "tooltip": (
                        "auto_balance：按显存压力清理跨轮残留，并在交接时卸载"
                        "放不下的第一模型；vram_saver："
                        "每阶段执行完都卸载；speed_first：优先保留缓存。"
                    ),
                },
            ),
            "latent_image": (
                "LATENT",
                {
                    "display_name": "输入潜变量",
                    "tooltip": "通常连接空潜变量或 VAE 编码结果。",
                },
            ),
        }
    )
    if handoff_mode == "percent":
        required["handoff_percent"] = (
            "FLOAT",
            {
                "default": 50.0,
                "min": 1.0,
                "max": 99.0,
                "step": 1.0,
                "display_name": "交接位置（百分比）",
                "tooltip": (
                    "只按第二采时间表选择共同 sigma。默认 50% 时，第一采 "
                    "8 步计划约执行 4 步，第二采 12 步计划约执行后 6 步。"
                ),
            },
        )
    else:
        required["handoff_snr_db"] = (
            "FLOAT",
            {
                "default": 2.4,
                "min": -40.0,
                "max": 40.0,
                "step": 0.1,
                "round": 0.01,
                "display_name": "SNR 交接值（dB）",
                "tooltip": (
                    "按物理噪声强度选择共同 sigma。Krea 2 Turbo / 8 步 "
                    "Simple 中 2.4 dB 约对应第一采 4 步；0 dB 只约 1 步，"
                    "内部放大时会由最低完成步数保护自动推迟。"
                ),
            },
        )
    required["minimum_upscale_stage_1_steps"] = (
        "INT",
        {
            "default": 4,
            "min": 1,
            "max": 100,
            "advanced": True,
            "display_name": "放大前最低一采步数",
            "tooltip": (
                "仅在内部放大时生效。若交接过早，节点会把共同 sigma 自动"
                "推迟到第一采至少完成这些步，避免未成形图被放大后出现融化、"
                "重影。Krea 2 Turbo 8 步推荐 4。"
            ),
        },
    )
    required["enable_stage_2"] = (
        STAGE_TOGGLE_OPTIONS,
        {
            "default": "开启",
            "display_name": "启用模型二",
            "tooltip": (
                "选择关闭后直接变为单采，第二模型、交接、内部放大和第二采步数"
                "全部自动失效；选择开启且模型二三项输入完整时执行双采，任一项"
                "未连接时也会自动只运行模型一。"
            ),
        },
    )
    required.update(_upscale_inputs())
    return {"required": required, "optional": optional}


class _ModelRelaySamplerDualIndependent:
    HANDOFF_MODE = None
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("最终结果", "第一采结果")
    OUTPUT_TOOLTIPS = (
        "第二模型在共同 sigma 之后完成的最终 latent。",
        "第一模型到达交接点的结果；启用内部放大时是可直接解码的干净低分辨率 latent。",
    )
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def check_lazy_status(
        self,
        enable_stage_2,
        stage_two_model=_MISSING_INPUT,
        stage_two_positive=_MISSING_INPUT,
        stage_two_negative=_MISSING_INPUT,
        **kwargs,
    ):
        if not _stage_is_enabled(enable_stage_2):
            return []
        lazy_inputs = {
            "stage_two_model": stage_two_model,
            "stage_two_positive": stage_two_positive,
            "stage_two_negative": stage_two_negative,
        }
        if any(value is _MISSING_INPUT for value in lazy_inputs.values()):
            return []
        return [name for name, value in lazy_inputs.items() if value is None]

    def _sample_impl(self, handoff_value, **kwargs):
        stage_two_inputs = {
            name: kwargs.get(name, _MISSING_INPUT)
            for name in _STAGE_TWO_CONNECTION_INPUTS
        }
        stage_2_requested = _stage_is_enabled(kwargs["enable_stage_2"])
        stage_2_complete = all(
            value is not _MISSING_INPUT and value is not None
            for value in stage_two_inputs.values()
        )
        stage_2_enabled = stage_2_requested and stage_2_complete
        if stage_2_requested and not stage_2_complete:
            missing_inputs = [
                name
                for name, value in stage_two_inputs.items()
                if value is _MISSING_INPUT or value is None
            ]
            LOGGER.info(
                "Model Relay model-two inputs incomplete; falling back to "
                "model one only (missing=%s)",
                ",".join(missing_inputs),
            )
        stages = [
            {
                "model": kwargs["stage_one_model"],
                "positive": kwargs["stage_one_positive"],
                "negative": kwargs["stage_one_negative"],
                "cfg": kwargs["stage_one_cfg"],
                "sampler": kwargs["stage_one_sampler"],
            }
        ]
        if stage_2_enabled:
            stages.append(
                {
                    "model": stage_two_inputs["stage_two_model"],
                    "positive": stage_two_inputs["stage_two_positive"],
                    "negative": stage_two_inputs["stage_two_negative"],
                    "cfg": kwargs["stage_two_cfg"],
                    "sampler": kwargs["stage_two_sampler"],
                }
            )

        output, stage_outputs = _run_dual_independent(
            stages,
            kwargs["latent_image"],
            kwargs["seed"],
            kwargs["stage_one_steps"],
            kwargs["stage_one_scheduler"],
            kwargs["stage_two_steps"],
            kwargs["stage_two_scheduler"],
            kwargs["denoise"],
            self.HANDOFF_MODE,
            handoff_value,
            kwargs["compatibility"],
            kwargs["memory_mode"],
            False,
            kwargs["final_width"],
            kwargs["final_height"],
            kwargs["upscale_method"],
            kwargs["redraw_strength"],
            kwargs["minimum_upscale_stage_1_steps"],
        )
        return output, stage_outputs[0]


class ModelRelaySamplerDualPercent(_ModelRelaySamplerDualIndependent):
    HANDOFF_MODE = "percent"
    DESCRIPTION = (
        "百分比交接专用双采节点。两套模型分别建立自己的 sigma 时间表，"
        "在第二采时间表的指定百分比处对齐，不含任何 SNR 控件。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return _independent_dual_inputs("percent")

    def sample(self, handoff_percent, **kwargs):
        return self._sample_impl(handoff_percent, **kwargs)


class ModelRelaySamplerDualSNR(_ModelRelaySamplerDualIndependent):
    HANDOFF_MODE = "snr"
    DESCRIPTION = (
        "SNR 交接专用双采节点。按目标噪声强度寻找两套时间表的共同 sigma，"
        "并在内部放大时保护第一采最低完成度，不含百分比控件。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return _independent_dual_inputs("snr")

    def sample(self, handoff_snr_db, **kwargs):
        return self._sample_impl(handoff_snr_db, **kwargs)


class ModelRelaySamplerTriple:
    @classmethod
    def INPUT_TYPES(cls):
        required = {}
        required.update(_stage_inputs(1, 6.5, "euler_ancestral"))
        required.update(_stage_inputs(2, 5.5, "dpmpp_2m", lazy=True))
        required.update(_stage_inputs(3, 4.5, "dpmpp_2m", lazy=True))
        required.update(_common_inputs(40))
        required["handoff_1_snr_db"] = (
            "FLOAT",
            {
                "default": -6.0,
                "min": -40.0,
                "max": 39.5,
                "step": 0.5,
                "display_name": "第一次 SNR 交接值（dB）",
                "tooltip": "SNR 模式第一次交接值。-6 dB 对应 sigma≈2。",
            },
        )
        required["handoff_2_snr_db"] = (
            "FLOAT",
            {
                "default": 6.0,
                "min": -39.5,
                "max": 40.0,
                "step": 0.5,
                "display_name": "第二次 SNR 交接值（dB）",
                "tooltip": "SNR 模式第二次交接值。6 dB 对应 sigma≈0.5。",
            },
        )
        required["handoff_1_percent"] = (
            "FLOAT",
            {
                "default": 40.0,
                "min": 1.0,
                "max": 98.0,
                "step": 1.0,
                "display_name": "第一次交接位置（百分比）",
                "tooltip": (
                    "percent 模式下第一模型交给第二模型的位置，按总步数百分比计算。"
                ),
            },
        )
        required["handoff_2_percent"] = (
            "FLOAT",
            {
                "default": 75.0,
                "min": 2.0,
                "max": 99.0,
                "step": 1.0,
                "display_name": "第二次交接位置（百分比）",
                "tooltip": (
                    "percent 模式下第二模型交给第三模型的位置，必须晚于第一次交接。"
                ),
            },
        )
        required["enable_stage_2"] = (
            "BOOLEAN",
            {
                "default": True,
                "display_name": "启用第二采",
                "label_on": "启用",
                "label_off": "关闭",
                "tooltip": (
                    "关闭后节点直接退化为单采；第二采、第三采及全部交接参数"
                    "自动失效，后续模型输入不会被请求执行。"
                ),
            },
        )
        required["enable_stage_3"] = (
            "BOOLEAN",
            {
                "default": True,
                "display_name": "启用第三采",
                "label_on": "启用",
                "label_off": "关闭",
                "tooltip": (
                    "关闭后节点退化为双采，第二次交接自动失效。第三采只有在"
                    "第二采同时启用时才会执行。"
                ),
            },
        )
        required.update(_upscale_inputs())
        return {"required": required}

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("最终结果", "第一采结果", "第二采结果")
    OUTPUT_TOOLTIPS = (
        "完成三模型接力后的最终 latent。",
        "第一模型完成其负责区间后的中间 latent。",
        "第二模型完成其负责区间后的中间 latent。",
    )
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "三个模型共享同一条 sigma 时间轴，在两个交接点连续采样，模型切换时"
        "不会重复加噪；可通过开关切换为单采或双采。"
    )

    def check_lazy_status(
        self,
        enable_stage_2,
        enable_stage_3,
        model_2=None,
        positive_2=None,
        negative_2=None,
        model_3=None,
        positive_3=None,
        negative_3=None,
        **kwargs,
    ):
        if not enable_stage_2:
            return []

        lazy_inputs = {
            "model_2": model_2,
            "positive_2": positive_2,
            "negative_2": negative_2,
        }
        if enable_stage_3:
            lazy_inputs.update(
                {
                    "model_3": model_3,
                    "positive_3": positive_3,
                    "negative_3": negative_3,
                }
            )
        return [name for name, value in lazy_inputs.items() if value is None]

    def sample(
        self,
        model_1,
        positive_1,
        negative_1,
        cfg_1,
        sampler_1,
        model_2,
        positive_2,
        negative_2,
        cfg_2,
        sampler_2,
        model_3,
        positive_3,
        negative_3,
        cfg_3,
        sampler_3,
        seed,
        steps,
        scheduler,
        denoise,
        compatibility,
        handoff_mode,
        transition_steps,
        memory_mode,
        latent_image,
        handoff_1_snr_db,
        handoff_2_snr_db,
        handoff_1_percent,
        handoff_2_percent,
        enable_stage_2,
        enable_stage_3,
        final_width,
        final_height,
        upscale_method,
        redraw_strength,
    ):
        stages = [
            {
                "model": model_1,
                "positive": positive_1,
                "negative": negative_1,
                "cfg": cfg_1,
                "sampler": sampler_1,
            },
        ]
        if enable_stage_2:
            stages.append(
                {
                    "model": model_2,
                    "positive": positive_2,
                    "negative": negative_2,
                    "cfg": cfg_2,
                    "sampler": sampler_2,
                }
            )
            if enable_stage_3:
                stages.append(
                    {
                        "model": model_3,
                        "positive": positive_3,
                        "negative": negative_3,
                        "cfg": cfg_3,
                        "sampler": sampler_3,
                    }
                )

        if handoff_mode == "snr":
            all_handoffs = [handoff_1_snr_db, handoff_2_snr_db]
        else:
            all_handoffs = [handoff_1_percent, handoff_2_percent]
        handoffs = all_handoffs[: max(0, len(stages) - 1)]
        output, stage_outputs = _run_relay(
            stages,
            latent_image,
            seed,
            steps,
            scheduler,
            denoise,
            handoff_mode,
            handoffs,
            compatibility,
            transition_steps,
            memory_mode,
            False,
            final_width,
            final_height,
            upscale_method,
            redraw_strength,
        )
        stage_1_output = stage_outputs[0]
        stage_2_output = (
            stage_outputs[1] if len(stage_outputs) > 1 else stage_1_output
        )
        return (output, stage_1_output, stage_2_output)


NODE_CLASS_MAPPINGS = {
    "ModelRelaySamplerDualPercent": ModelRelaySamplerDualPercent,
    "ModelRelaySamplerDualSNR": ModelRelaySamplerDualSNR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelRelaySamplerDualPercent": "模型接力采样器（双采·百分比）",
    "ModelRelaySamplerDualSNR": "模型接力采样器（双采·SNR）",
}
