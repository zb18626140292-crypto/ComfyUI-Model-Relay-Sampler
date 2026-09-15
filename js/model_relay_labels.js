import { app } from "../../scripts/app.js";

const NODE_TYPES = new Set([
    "ModelRelaySamplerDualPercent",
    "ModelRelaySamplerDualSNR",
]);

const WIDGET_LABELS = {
    stage_one_cfg: "模型一 CFG",
    stage_one_steps: "模型一步数",
    stage_one_sampler: "模型一采样器",
    stage_one_scheduler: "模型一调度器",
    stage_two_cfg: "模型二 CFG",
    stage_two_steps: "模型二步数",
    stage_two_sampler: "模型二采样器",
    stage_two_scheduler: "模型二调度器",
    seed: "种子",
    control_after_generate: "生成后控制",
    denoise: "降噪强度",
    compatibility: "潜空间兼容检查",
    memory_mode: "显存模式",
    handoff_snr_db: "SNR 交接值（dB）",
    handoff_percent: "交接位置（百分比）",
    minimum_upscale_stage_1_steps: "放大前最低一采步数",
    enable_stage_2: "启用模型二",
    final_width: "最终宽度",
    final_height: "最终高度",
    upscale_method: "放大方法",
    redraw_strength: "重绘强度",
};

const INPUT_LABELS = {
    stage_one_model: "模型一",
    stage_one_positive: "模型一正向",
    stage_one_negative: "模型一负向",
    stage_two_model: "模型二",
    stage_two_positive: "模型二正向",
    stage_two_negative: "模型二负向",
    latent_image: "输入潜变量",
};

function applyChineseLabels(node) {
    for (const widget of node?.widgets || []) {
        const label = WIDGET_LABELS[widget?.name];
        if (!label) continue;
        widget.label = label;
        widget.options = { ...(widget.options || {}), label };
        if (widget.name === "enable_stage_2") {
            widget.options.values = ["关闭", "开启"];
            if (typeof widget.value === "boolean") {
                widget.value = widget.value ? "开启" : "关闭";
            }
        }
    }

    for (const input of node?.inputs || []) {
        const label = INPUT_LABELS[input?.name];
        if (label) input.label = label;
    }
}

function installLabelOverrides(nodeType) {
    if (nodeType.prototype.__modelRelayChineseLabels) return;
    Object.defineProperty(nodeType.prototype, "__modelRelayChineseLabels", {
        value: true,
        configurable: false,
        enumerable: false,
        writable: false,
    });

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const result = originalOnNodeCreated?.apply(this, arguments);
        applyChineseLabels(this);
        return result;
    };

    const originalConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
        // Older workflows stored the removed force-offload switch immediately
        // after memory_mode. Drop that serialized value before LiteGraph maps
        // the remaining values to widgets, so following settings do not shift.
        if (
            Array.isArray(info?.widgets_values)
            && Array.isArray(this.widgets)
            && info.widgets_values.length === this.widgets.length + 1
        ) {
            const memoryIndex = this.widgets.findIndex(
                (widget) => widget?.name === "memory_mode",
            );
            if (memoryIndex >= 0) {
                info.widgets_values.splice(memoryIndex + 1, 1);
            }
        }

        // The stage-two control changed from a boolean switch to a Chinese
        // combo selector. Convert serialized values from existing workflows
        // before LiteGraph assigns them to the new widget.
        if (Array.isArray(info?.widgets_values) && Array.isArray(this.widgets)) {
            const stageTwoIndex = this.widgets.findIndex(
                (widget) => widget?.name === "enable_stage_2",
            );
            if (stageTwoIndex >= 0) {
                const oldValue = info.widgets_values[stageTwoIndex];
                if (typeof oldValue === "boolean") {
                    info.widgets_values[stageTwoIndex] = oldValue ? "开启" : "关闭";
                } else if (oldValue === 1 || oldValue === "true" || oldValue === "启用") {
                    info.widgets_values[stageTwoIndex] = "开启";
                } else if (oldValue === 0 || oldValue === "false") {
                    info.widgets_values[stageTwoIndex] = "关闭";
                }
            }
        }
        const result = originalConfigure?.apply(this, arguments);
        applyChineseLabels(this);
        requestAnimationFrame(() => applyChineseLabels(this));
        return result;
    };

    const originalOnDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function () {
        const result = originalOnDrawForeground?.apply(this, arguments);
        applyChineseLabels(this);
        return result;
    };
}

app.registerExtension({
    name: "ModelRelaySampler.ChineseLabels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (NODE_TYPES.has(nodeData?.name)) {
            installLabelOverrides(nodeType);
        }
    },
});
