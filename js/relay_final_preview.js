import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SAMPLERS = new Set(["ModelRelaySamplerDualPercent", "ModelRelaySamplerDualSNR"]);
const IMAGE_OUTPUTS = new Set(["SaveImage", "PreviewImage"]);
const DECODERS = new Set(["VAEDecode", "VAEDecodeTiled"]);
let executionVersion = 0;
let initialized = false;
const cachedNodes = new Set();

function inputSource(node, name) {
    const index = node?.inputs?.findIndex((input) => input.name === name) ?? -1;
    let link = index < 0 ? null : node.getInputLink?.(index);
    const visited = new Set();
    while (link) {
        const source = node.graph?.getNodeById(link.origin_id);
        if (!source || visited.has(source)) return null;
        visited.add(source);
        if ((source.comfyClass || source.type) !== "Reroute") {
            return { node: source, slot: link.origin_slot };
        }
        link = source.getInputLink?.(0);
    }
    return null;
}

function samplerFor(output) {
    if (!IMAGE_OUTPUTS.has(output?.comfyClass || output?.type)) return null;
    const decoded = inputSource(output, "images");
    if (!decoded || decoded.slot !== 0 || !DECODERS.has(decoded.node.comfyClass || decoded.node.type)) return null;
    const sampled = inputSource(decoded.node, "samples");
    if (!sampled || sampled.slot !== 0 || !SAMPLERS.has(sampled.node.comfyClass || sampled.node.type)) return null;
    return sampled.node;
}

function clearFinalThumbnail(node) {
    if (!SAMPLERS.has(node?.comfyClass || node?.type)) return;
    const output = app.nodeOutputs[node.id];
    if (output && Object.hasOwn(output, "images")) {
        const remaining = { ...output };
        delete remaining.images;
        app.nodeOutputs[node.id] = remaining;
    }
    delete node.images;
}

async function showFinalPreview({ detail }) {
    const images = detail?.output?.images;
    if (!Array.isArray(images) || !images[0]?.filename) return;
    const graph = app.graph;
    const output = graph?.getNodeById(detail.display_node ?? detail.node);
    const sampler = samplerFor(output);
    if (!sampler) return;
    const version = executionVersion;
    const first = images[0];
    const query = new URLSearchParams({
        filename: first.filename, subfolder: first.subfolder || "", type: first.type || "output",
    });
    const response = await api.fetchApi(`/view?${query}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`Image request failed (${response.status})`);
    const blob = await response.blob();
    if (version !== executionVersion || app.graph !== graph
        || graph.getNodeById(sampler.id) !== sampler || samplerFor(output) !== sampler) return;

    // Update the existing preview only; image outputs create a second gallery.
    clearFinalThumbnail(sampler);
    api.dispatchCustomEvent("b_preview_with_metadata", {
        blob, displayNodeId: String(sampler.id), jobId: detail.prompt_id,
    });
    sampler.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "ModelRelaySampler.FinalPreview",
    nodeCreated(node) {
        clearFinalThumbnail(node);
    },
    loadedGraphNode(node) {
        clearFinalThumbnail(node);
    },
    setup() {
        if (initialized) return;
        initialized = true;
        api.addEventListener("execution_start", () => { executionVersion++; cachedNodes.clear(); });
        api.addEventListener("execution_cached", ({ detail }) => {
            for (const id of detail?.nodes || []) cachedNodes.add(id);
        });
        api.addEventListener("execution_success", ({ detail }) => {
            for (const id of cachedNodes) {
                const output = app.nodeOutputs[id];
                if (output?.images?.length) {
                    showFinalPreview({ detail: { node: id, prompt_id: detail.prompt_id, output } })
                        .catch((error) => console.warn("Model Relay: cached final preview could not be loaded.", error));
                }
            }
        });
        api.addEventListener("executed", (event) => {
            showFinalPreview(event).catch((error) => {
                console.warn("Model Relay: final decoded preview could not be loaded.", error);
            });
        });
    },
});
