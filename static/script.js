const API = window.location.origin;

const els = {
    modelSelect: document.getElementById("model-select"),
    temperature: document.getElementById("temperature"),
    tempValue: document.getElementById("temp-value"),
    maxTokens: document.getElementById("max-tokens"),
    tokensValue: document.getElementById("tokens-value"),
    topK: document.getElementById("top-k"),
    topkValue: document.getElementById("topk-value"),
    modelInfo: document.getElementById("model-info"),
    prompt: document.getElementById("prompt"),
    generateBtn: document.getElementById("generate-btn"),
    btnText: document.querySelector(".btn-text"),
    btnLoading: document.querySelector(".btn-loading"),
    outputArea: document.getElementById("output-area"),
    outputText: document.getElementById("output-text"),
    copyBtn: document.getElementById("copy-btn"),
    status: document.getElementById("status"),
    statusText: document.getElementById("status-text"),
};

let modelsData = [];

function setStatus(state, text) {
    els.status.className = "status " + state;
    els.statusText.textContent = text;
}

async function fetchModels() {
    try {
        const res = await fetch(`${API}/api/models`);
        const data = await res.json();
        modelsData = data.models;

        els.modelSelect.innerHTML = "";
        if (modelsData.length === 0) {
            els.modelSelect.innerHTML = '<option value="">No models — run train.py first</option>';
            setStatus("error", "No models");
            return;
        }

        modelsData.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m.name;
            opt.textContent = `${m.name} (${m.type} · ${m.parameters})`;
            els.modelSelect.appendChild(opt);
        });

        setStatus("connected", `${modelsData.length} model(s) on ${data.device}`);
        updateModelInfo();
    } catch {
        setStatus("error", "Server offline");
        els.modelSelect.innerHTML = '<option value="">Cannot reach server</option>';
    }
}

function updateModelInfo() {
    const selected = modelsData.find((m) => m.name === els.modelSelect.value);
    if (selected) {
        els.modelInfo.textContent = `${selected.type} · ${selected.parameters} parameters · vocab ${selected.vocab_size}`;
        els.modelInfo.classList.add("visible");
    } else {
        els.modelInfo.classList.remove("visible");
    }
}

els.temperature.addEventListener("input", () => {
    els.tempValue.textContent = parseFloat(els.temperature.value).toFixed(2);
});

els.maxTokens.addEventListener("input", () => {
    els.tokensValue.textContent = els.maxTokens.value;
});

els.topK.addEventListener("input", () => {
    const v = parseInt(els.topK.value);
    els.topkValue.textContent = v === 0 ? "Off" : v;
});

els.modelSelect.addEventListener("change", updateModelInfo);

els.generateBtn.addEventListener("click", generate);
els.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) generate();
});

async function generate() {
    const model = els.modelSelect.value;
    if (!model) return;

    els.generateBtn.disabled = true;
    els.btnText.style.display = "none";
    els.btnLoading.style.display = "inline";

    try {
        const topK = parseInt(els.topK.value);
        const body = {
            model,
            prompt: els.prompt.value,
            max_tokens: parseInt(els.maxTokens.value),
            temperature: parseFloat(els.temperature.value),
        };
        if (topK > 0) body.top_k = topK;

        const res = await fetch(`${API}/api/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        const data = await res.json();

        if (data.error) {
            els.outputText.textContent = `Error: ${data.error}`;
        } else {
            const promptLen = data.prompt ? data.prompt.length : 0;
            if (promptLen > 0) {
                const promptSpan = document.createElement("span");
                promptSpan.className = "prompt-highlight";
                promptSpan.textContent = data.text.slice(0, promptLen);
                const rest = document.createTextNode(data.text.slice(promptLen));
                els.outputText.innerHTML = "";
                els.outputText.appendChild(promptSpan);
                els.outputText.appendChild(rest);
            } else {
                els.outputText.textContent = data.text;
            }
        }

        els.outputArea.style.display = "block";
    } catch (err) {
        els.outputText.textContent = `Request failed: ${err.message}`;
        els.outputArea.style.display = "block";
    } finally {
        els.generateBtn.disabled = false;
        els.btnText.style.display = "inline";
        els.btnLoading.style.display = "none";
    }
}

els.copyBtn.addEventListener("click", () => {
    navigator.clipboard.writeText(els.outputText.textContent).then(() => {
        els.copyBtn.textContent = "Copied!";
        setTimeout(() => (els.copyBtn.textContent = "Copy"), 1500);
    });
});

fetchModels();
