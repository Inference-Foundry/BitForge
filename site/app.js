// BitForge Unified Research & Interactive Lab Scripting

document.addEventListener('DOMContentLoaded', () => {
    // Initialize subsystems
    initNavigation();
    initSimulator();
    initTaxonomyTabs();
    initOutlierLab();
    initAccordions();
    initKaTeX();
});

// ==========================================
// 1. Navigation & Routing (Unified Dashboard)
// ==========================================

let rooflineChartInstance = null;
let degChartInstance = null;

function initNavigation() {
    const navBtns = document.querySelectorAll('#nav-menu button');
    const sections = document.querySelectorAll('.section-view');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const mobileNav = document.getElementById('mobile-nav');
    const mobileNavMenu = document.getElementById('mobile-nav-menu');

    // Generate mobile navigation items dynamically
    navBtns.forEach(btn => {
        const li = document.createElement('li');
        const mobileBtn = document.createElement('button');
        mobileBtn.className = 'w-full text-left px-6 py-3 border-l-4 border-transparent text-bf-text text-sm hover:bg-stone-50 transition-colors';
        mobileBtn.innerHTML = btn.innerHTML;
        mobileBtn.setAttribute('data-target', btn.getAttribute('data-target'));
        li.appendChild(mobileBtn);
        mobileNavMenu.appendChild(li);

        mobileBtn.addEventListener('click', (e) => {
            let target = e.target;
            // Handle clicking inner elements of the button
            while (target && target.tagName !== 'BUTTON') {
                target = target.parentElement;
            }
            if (target) {
                handleNavigation(target.getAttribute('data-target'));
                mobileNav.classList.add('hidden');
            }
        });
    });

    mobileMenuBtn.addEventListener('click', () => {
        mobileNav.classList.toggle('hidden');
    });

    navBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            let target = e.target;
            // Handle clicking inner elements of the button
            while (target && target.tagName !== 'BUTTON') {
                target = target.parentElement;
            }
            if (target) {
                handleNavigation(target.getAttribute('data-target'));
            }
        });
    });

    function handleNavigation(targetId) {
        sections.forEach(sec => sec.classList.add('hidden'));
        document.getElementById(targetId).classList.remove('hidden');

        // Update Desktop Navigation states
        navBtns.forEach(b => {
            b.classList.remove('nav-active');
            b.classList.add('text-bf-text');
            const icon = b.querySelector('i');
            if (icon) icon.classList.remove('text-bf-accent');

            if (b.getAttribute('data-target') === targetId) {
                b.classList.add('nav-active');
                if (icon) icon.classList.add('text-bf-accent');
            }
        });

        // Scroll to top
        document.querySelector('#main-content').scrollTop = 0;

        // Lazy load Chart.js visualizations
        if (targetId === 'sec-runtime' && !rooflineChartInstance) renderRoofline();
        if (targetId === 'sec-empirical' && !degChartInstance) renderDegradation();
    }
}

// ==========================================
// 2. Interactive Simulator
// ==========================================

const simulationData = {
    llama8b: {
        name: "Llama 3 (8B)",
        size: 8.03,
        basePPL: 6.15,
        ppl: { 16: 6.15, 8: 6.18, 4: 6.38, 2: 15.42 },
        tps: { 16: 22, 8: 34, 4: 58, 2: 74 },
        retention: { 16: 100, 8: 99.8, 4: 97.5, 2: 38.2 },
        explanations: {
            16: "At full 16-bit float precision, the model maintains its baseline pre-trained perplexity score and maximum general reasoning capabilities. No degradation.",
            8: "INT8 quantization shrinks the model size by half. Perplexity remains virtually indistinguishable from FP16 (+0.03), making it an extremely safe and cost-effective deployment choice.",
            4: "At 4-bit (using AWQ/GPTQ group-wise scaling), the model is 4x smaller and nearly 3x faster. Perplexity increases slightly (+0.23), keeping reasoning and factual recall heavily intact.",
            2: "Extreme 2-bit quantization falls off the precision cliff. Perplexity explodes (+9.27) and reasoning retention collapses to 38%. The model suffers severe hallucination and is practically unusable."
        }
    },
    mistral7b: {
        name: "Mistral (7B)",
        size: 7.24,
        basePPL: 5.92,
        ppl: { 16: 5.92, 8: 5.95, 4: 6.18, 2: 12.86 },
        tps: { 16: 25, 8: 38, 4: 65, 2: 82 },
        retention: { 16: 100, 8: 99.7, 4: 96.8, 2: 44.5 },
        explanations: {
            16: "At full 16-bit precision, Mistral 7B maintains its baseline pre-trained perplexity score and maximum capabilities. No degradation.",
            8: "INT8 reduces Mistral 7B size by 50%. Accuracy loss is marginal (+0.03 PPL), making this standard for high-performance CPU/GPU execution.",
            4: "INT4 compression (using AWQ/GPTQ) maintains over 96.8% reasoning retention while enabling 65 tokens/sec. Perplexity increase is extremely small (+0.26 PPL). Highly recommended.",
            2: "2-bit quantization introduces massive rounding errors. Perplexity balloons to 12.86 (+6.94 PPL). The model loses structural grammatical capabilities, frequently repeating symbols or words."
        }
    },
    gemma2b: {
        name: "Gemma 2 (2B)",
        size: 2.51,
        basePPL: 7.82,
        ppl: { 16: 7.82, 8: 7.92, 4: 8.94, 2: 32.14 },
        tps: { 16: 55, 8: 72, 4: 110, 2: 135 },
        retention: { 16: 100, 8: 98.9, 4: 88.5, 2: 12.4 },
        explanations: {
            16: "At full 16-bit precision, the highly compact Gemma 2B model runs fast and maintains full baseline accuracy.",
            8: "INT8 quantization introduces noticeable but acceptable perplexity degradation (+0.10 PPL) for a 2B parameter class model, reflecting its smaller capacity to absorb compression loss.",
            4: "At 4-bit, the model's capacity limit is pushed. Perplexity climbs (+1.12 PPL) and reasoning retention dips to 88.5%. Highly domain-specific factual knowledge begins to break down.",
            2: "At 2-bit, the model experiences total cognitive collapse. Accuracy drops to 12.4%. Almost all sentences generated represent incoherent garbled tokens."
        }
    },
    llama70b: {
        name: "Llama 3 (70B)",
        size: 70.56,
        basePPL: 4.10,
        ppl: { 16: 4.10, 8: 4.11, 4: 4.18, 2: 5.86 },
        tps: { 16: 4.5, 8: 7.8, 4: 14.2, 2: 21.0 },
        retention: { 16: 100, 8: 99.9, 4: 99.1, 2: 84.8 },
        explanations: {
            16: "At 16-bit, Llama 3 70B represents a state-of-the-art foundation model but requires a massive memory footprint of 140+ GB.",
            8: "INT8 quantization drops the size to 70 GB with absolutely zero practical degradation (+0.01 PPL). Perfect for dual-GPU consumer cards.",
            4: "At 4-bit, the 70B parameter model demonstrates high resilience to quantization. Memory drops to 35GB, speed triples (14.2 tok/s), and accuracy remains at 99.1% of baseline. High quantization efficiency.",
            2: "Even at 2-bit, Llama 3 70B maintains a surprising 84.8% reasoning retention and 5.86 perplexity. Its high parameter capacity absorbs rounding losses significantly better than smaller models."
        }
    }
};

function initSimulator() {
    const modelSelect = document.getElementById('model-select');
    const calibSelect = document.getElementById('calibration-select');
    const bitBtns = document.querySelectorAll('#simulator-bit-grid .bit-btn');

    let activeBits = 16;

    modelSelect.addEventListener('change', updateSimulation);
    calibSelect.addEventListener('change', updateSimulation);

    bitBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            bitBtns.forEach(b => {
                b.classList.remove('active', 'bg-bf-light-accent', 'border-bf-accent');
                b.classList.add('bg-white');
                const title = b.querySelector('span:first-child');
                if (title) {
                    title.classList.remove('text-bf-accent');
                    title.classList.add('text-bf-text');
                }
            });

            btn.classList.add('active', 'bg-bf-light-accent', 'border-bf-accent');
            btn.classList.remove('bg-white');
            const title = btn.querySelector('span:first-child');
            if (title) {
                title.classList.add('text-bf-accent');
                title.classList.remove('text-bf-text');
            }

            activeBits = parseInt(btn.getAttribute('data-bits'));
            updateSimulation();
        });
    });

    updateSimulation();

    function updateSimulation() {
        const selectedModelKey = modelSelect.value;
        const calibrationKey = calibSelect.value;
        const model = simulationData[selectedModelKey];

        // 1. Calculate static VRAM
        let vramVal = (model.size * activeBits) / 8;
        vramVal += (activeBits === 16 ? 1.5 : activeBits === 8 ? 1.1 : activeBits === 4 ? 0.8 : 0.6);
        vramVal = parseFloat(vramVal.toFixed(1));

        document.getElementById('vram-val').innerText = vramVal;

        const vramPercentage = Math.min((vramVal / 85.0) * 100, 100);
        document.getElementById('vram-bar').style.width = `${vramPercentage}%`;

        const originalVram = parseFloat(((model.size * 16) / 8 + 1.5).toFixed(1));
        if (activeBits === 16) {
            document.getElementById('vram-save').innerText = `Full baseline size`;
        } else {
            const savings = Math.round((1 - (vramVal / originalVram)) * 100);
            document.getElementById('vram-save').innerText = `Saved ${savings}% memory`;
        }

        // 2. Throughput Speed
        const speedVal = model.tps[activeBits];
        document.getElementById('speed-val').innerText = speedVal;

        const speedPercentage = Math.min((speedVal / 140.0) * 100, 100);
        document.getElementById('speed-bar').style.width = `${speedPercentage}%`;

        if (activeBits === 16) {
            document.getElementById('speed-gain').innerText = `Baseline speed`;
        } else {
            const speedup = (speedVal / model.tps[16]).toFixed(1);
            document.getElementById('speed-gain').innerText = `${speedup}x speedup ratio`;
        }

        // 3. Perplexity & Domain Calibration calibration impacts
        let ppl = model.ppl[activeBits];
        let retention = model.retention[activeBits];

        if (calibrationKey === 'medical' || calibrationKey === 'code') {
            if (activeBits === 4) {
                ppl = parseFloat((ppl - 0.05).toFixed(2));
                retention = Math.min(retention + 0.8, 99.8);
            } else if (activeBits === 2) {
                ppl = parseFloat((ppl + 1.2).toFixed(2));
                retention = Math.max(retention - 5.0, 5.0);
            }
        }

        document.getElementById('ppl-val').innerText = ppl.toFixed(2);
        document.getElementById('retention-val').innerText = `${retention.toFixed(1)}%`;

        const pplStatus = document.getElementById('ppl-status');
        pplStatus.className = 'text-3xs font-bold px-2 py-0.5 rounded border uppercase';

        if (retention >= 98.0) {
            pplStatus.innerText = 'EXCELLENT';
            pplStatus.classList.add('bg-emerald-100', 'text-emerald-800', 'border-emerald-300');
        } else if (retention >= 85.0) {
            pplStatus.innerText = 'ACCEPTABLE';
            pplStatus.classList.add('bg-amber-100', 'text-amber-800', 'border-amber-300');
        } else {
            pplStatus.innerText = 'COLLAPSED';
            pplStatus.classList.add('bg-rose-100', 'text-rose-800', 'border-rose-300');
        }

        document.getElementById('sim-explanation').innerText = model.explanations[activeBits];
    }
}

// ==========================================
// 3. Taxonomy Tabs (Architectures Section)
// ==========================================

function initTaxonomyTabs() {
    const archTabs = document.querySelectorAll('.arch-tab');
    const archContents = document.querySelectorAll('.arch-content');

    archTabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            archContents.forEach(c => c.classList.add('hidden'));
            archTabs.forEach(t => {
                t.classList.remove('tab-active', 'border-b-2');
                t.classList.add('text-bf-muted');
            });

            const target = document.getElementById(e.target.getAttribute('data-target'));
            target.classList.remove('hidden');
            e.target.classList.add('tab-active', 'border-b-2', 'border-bf-accent');
            e.target.classList.remove('text-bf-muted');
        });
    });
}

// ==========================================
// 4. Outlier suppression lab
// ==========================================

function initOutlierLab() {
    const chart = document.getElementById('activation-chart');
    const suppressBtn = document.getElementById('suppress-btn');
    const peakVal = document.getElementById('outlier-peak-val');
    const quantStatusText = document.getElementById('quant-status-text');

    const channels = [
        { id: 1, val: 0.82 },
        { id: 2, val: 1.15 },
        { id: 3, val: 0.94 },
        { id: 4, val: 15.0, isOutlier: true },
        { id: 5, val: 1.05 },
        { id: 6, val: 0.76 },
        { id: 7, val: 1.28 },
        { id: 8, val: 0.91 },
        { id: 9, val: 0.85 },
        { id: 10, val: 1.02 },
        { id: 11, val: 12.0, isOutlier: true },
        { id: 12, val: 0.98 },
        { id: 13, val: 0.72 },
        { id: 14, val: 1.14 },
        { id: 15, val: 0.88 },
        { id: 16, val: 1.01 }
    ];

    let suppressed = false;

    function drawChart() {
        chart.innerHTML = '';
        channels.forEach(ch => {
            const bar = document.createElement('div');
            bar.className = 'bar-outliers';
            if (ch.isOutlier) {
                bar.classList.add('outlier');
            }

            let heightVal = ch.val;
            if (suppressed && ch.isOutlier) {
                heightVal = 2.45;
            }

            const heightPercent = (heightVal / 15.0) * 100;
            bar.style.height = `${heightPercent}%`;
            bar.setAttribute('data-val', heightVal.toFixed(2));
            chart.appendChild(bar);
        });
    }

    suppressBtn.addEventListener('click', () => {
        suppressed = !suppressed;

        if (suppressed) {
            drawChart();
            peakVal.innerText = '2.45';
            peakVal.classList.remove('text-bf-accent');
            peakVal.style.color = '#10b981'; // emerald-500

            quantStatusText.innerHTML = '<i class="fa-solid fa-circle-check"></i> STABILIZED';
            quantStatusText.className = 'text-sm font-bold flex items-center gap-1.5 mt-1 text-emerald-600';

            suppressBtn.innerHTML = '<i class="fa-solid fa-rotate-left"></i> Reset Activations';
            suppressBtn.style.backgroundColor = '#10b981';
        } else {
            drawChart();
            peakVal.innerText = '15.0';
            peakVal.style.color = '';
            peakVal.classList.add('text-bf-accent');

            quantStatusText.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> CRUSHED';
            quantStatusText.className = 'text-sm font-bold flex items-center gap-1.5 mt-1 text-bf-accent';

            suppressBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Apply SmoothQuant Migration';
            suppressBtn.style.backgroundColor = '';
        }
    });

    drawChart();
}

// ==========================================
// 5. Accordion (Open Research Questions)
// ==========================================

function initAccordions() {
    const accordions = document.querySelectorAll('.accordion-btn');
    accordions.forEach(acc => {
        acc.addEventListener('click', function () {
            const content = this.nextElementSibling;
            const icon = this.querySelector('span');
            if (content.classList.contains('hidden')) {
                content.classList.remove('hidden');
                icon.classList.add('rotate-180');
            } else {
                content.classList.add('hidden');
                icon.classList.remove('rotate-180');
            }
        });
    });
}

// ==========================================
// 6. KaTeX Formatting Initialization
// ==========================================

function initKaTeX() {
    if (typeof renderMathInElement === "function") {
        renderMathInElement(document.body, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false }
            ]
        });
    }
}

// ==========================================
// 7. Dynamic Chart.js Renderings
// ==========================================

function renderRoofline() {
    const canvas = document.getElementById('rooflineChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    rooflineChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Memory Bound Limit',
                    data: [{ x: 0.1, y: 10 }, { x: 10, y: 1000 }],
                    borderColor: '#78716c',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Compute Bound Ceiling',
                    data: [{ x: 10, y: 1000 }, { x: 100, y: 1000 }],
                    borderColor: '#292524',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'FP16 Generation (BS=1)',
                    data: [{ x: 2, y: 200 }],
                    backgroundColor: '#292524',
                    borderColor: '#292524',
                    pointRadius: 8,
                    pointStyle: 'circle',
                    showLine: false
                },
                {
                    label: 'INT4 Generation (Roofline Shift)',
                    data: [{ x: 8, y: 800 }],
                    backgroundColor: '#ea580c',
                    borderColor: '#ea580c',
                    pointRadius: 8,
                    pointStyle: 'triangle',
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Arithmetic Intensity (FLOPs / Byte)' },
                    min: 0.1, max: 100
                },
                y: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Performance (TFLOPS)' },
                    min: 10, max: 2000
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return context.dataset.label + ': ' + context.parsed.y + ' TFLOPS';
                        }
                    }
                },
                legend: { position: 'bottom', labels: { usePointStyle: true } }
            }
        }
    });
}

function renderDegradation() {
    const canvas = document.getElementById('degradationChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    degChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['FP16 (Baseline)', 'INT8 (W8A8)', 'INT4 (W4A16)', 'INT3 (W3A16)', 'INT2 (W2A16)'],
            data: [], // populated below
            datasets: [
                {
                    label: 'Factual Recall (TriviaQA % Retained)',
                    data: [100, 99.5, 96.2, 88.5, 71.0],
                    backgroundColor: '#d6d3d1'
                },
                {
                    label: 'Abstract Reasoning (GSM8k % Retained)',
                    data: [100, 98.1, 89.0, 45.2, 12.5],
                    backgroundColor: '#ea580c'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: { display: true, text: '% Capability Retained vs Baseline' }
                }
            },
            plugins: {
                legend: { position: 'top' },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return context.dataset.label + ': ' + context.parsed.y + '%';
                        }
                    }
                }
            }
        }
    });
}
