import { app } from "../../scripts/app.js";

const NODE_TYPES = new Set([
    "ModelRelaySamplerDualPercent",
    "ModelRelaySamplerDualSNR",
    "QwenImage21DualPercentSampler",
    "QwenImage21DualSNRSampler"
]);
const originalOrders = new WeakMap();

function rememberOrder(node) {
    if (!originalOrders.has(node)) {
        originalOrders.set(node, (node.widgets || []).map((widget) => widget.name));
    }
    return originalOrders.get(node);
}

function restoreOriginalOrder(node) {
    const order = rememberOrder(node);
    const widgets = node.widgets || [];
    const byName = new Map(widgets.map((widget) => [widget.name, widget]));
    const ordered = order.flatMap((name) => byName.has(name) ? [byName.get(name)] : []);
    const known = new Set(order);
    ordered.push(...widgets.filter((widget) => !known.has(widget.name)));
    widgets.splice(0, widgets.length, ...ordered);
}

function applyLayout(node) {
    if (!Array.isArray(node.widgets)) return;
    rememberOrder(node);
    for (const stage of ["stage_one", "stage_two"]) {
        const cfgIndex = node.widgets.findIndex((widget) => widget.name === `${stage}_cfg`);
        const stepsIndex = node.widgets.findIndex((widget) => widget.name === `${stage}_steps`);
        if (cfgIndex >= 0 && stepsIndex > cfgIndex) {
            const [steps] = node.widgets.splice(stepsIndex, 1);
            node.widgets.splice(cfgIndex, 0, steps);
        }
    }
    const mode = node.widgets.find((widget) => widget.name === "redraw_mode");
    if (mode) {
        mode.hidden = true;
        mode.options = { ...(mode.options || {}), hidden: true };
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "ModelRelaySampler.WidgetLayout",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_TYPES.has(nodeData?.name) || nodeType.prototype.__relayWidgetLayout) return;
        Object.defineProperty(nodeType.prototype, "__relayWidgetLayout", { value: true });

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            applyLayout(this);
            return result;
        };

        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            // Keep old positional workflows in backend order while restoring.
            restoreOriginalOrder(this);
            try {
                return configure?.apply(this, arguments);
            } finally {
                applyLayout(this);
            }
        };

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = onSerialize?.apply(this, arguments);
            if (Array.isArray(info?.widgets_values)) {
                const widgets = (this.widgets || []).filter((widget) => widget.serialize !== false);
                const values = new Map(widgets.map((widget, index) => [widget.name, info.widgets_values[index]]));
                const order = rememberOrder(this);
                const names = [...order.filter((name) => values.has(name)),
                    ...widgets.map((widget) => widget.name).filter((name) => !order.includes(name))];
                info.widgets_values = names.map((name) => values.get(name));
                info.widgets_values_named = Object.fromEntries(values);
            }
            return result;
        };
    },
});
