/**
 * Yosemite CycleGAN — browser demo.
 *
 * Everything runs client-side: the ONNX generators are fetched from
 * docs/models/, executed with ONNX Runtime Web, and the result is drawn onto a
 * canvas. No image ever leaves the machine, which is also why the page works
 * unchanged on GitHub Pages with no backend at all.
 */
(() => {
  'use strict';

  const MODELS_URL = 'models/manifest.json';
  const SAMPLES_URL = 'samples/manifest.json';

  const state = {
    inputSize: 128,
    /** direction -> { file, session|null, loading:Promise|null } */
    models: new Map(),
    direction: 'summer2winter',
    /** The currently loaded source image, as an ImageBitmap or HTMLImageElement. */
    source: null,
    backend: null,
    busy: false,
  };

  const el = {};
  const ids = [
    'app', 'noModels', 'status', 'fileInput', 'dropZone', 'directionToggle',
    'backendLabel', 'result', 'beforeCanvas', 'afterCanvas', 'afterClip',
    'compare', 'handle', 'timing', 'downloadBtn', 'samples', 'sampleStrip',
    'randomSampleBtn', 'afterBadge',
  ];

  // ---------------------------------------------------------------- status
  function setStatus(message, { error = false, busy = false } = {}) {
    el.status.classList.toggle('error', error);
    el.status.innerHTML = busy ? `<span class="spinner"></span>${message}` : message;
  }

  // ---------------------------------------------------------------- models
  async function loadManifest() {
    // A 404 here is the expected state before any model has been trained and
    // committed, so it gets its own explanatory panel rather than an error.
    let response;
    try {
      response = await fetch(MODELS_URL, { cache: 'no-cache' });
    } catch {
      return null;
    }
    if (!response.ok) return null;
    try {
      return await response.json();
    } catch {
      return null;
    }
  }

  /** Load (once) and return the ORT session for a direction. */
  function getSession(direction) {
    const entry = state.models.get(direction);
    if (!entry) return Promise.reject(new Error(`No model published for ${direction}`));
    if (entry.session) return Promise.resolve(entry.session);
    if (!entry.loading) {
      const url = `models/${entry.file}`;
      entry.loading = ort.InferenceSession
        .create(url, {
          // WebGPU when the browser has it, WASM everywhere else. Both run the
          // same graph; WebGPU is roughly an order of magnitude faster.
          executionProviders: navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'],
          graphOptimizationLevel: 'all',
        })
        .then((session) => {
          entry.session = session;
          state.backend = navigator.gpu ? 'WebGPU' : 'WebAssembly';
          el.backendLabel.textContent = state.backend;
          return session;
        })
        .catch((err) => {
          entry.loading = null; // allow a retry on the next attempt
          throw err;
        });
    }
    return entry.loading;
  }

  // ----------------------------------------------------------- pre/post
  /**
   * Draw an image into a size x size canvas and return NCHW float32 in [-1, 1],
   * matching the Normalize((0.5,)*3, (0.5,)*3) used during training.
   */
  function toTensor(image, size) {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, size, size);
    const { data } = ctx.getImageData(0, 0, size, size);

    const plane = size * size;
    const array = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const p = i * 4;
      array[i] = data[p] / 127.5 - 1;                 // R
      array[plane + i] = data[p + 1] / 127.5 - 1;     // G
      array[2 * plane + i] = data[p + 2] / 127.5 - 1; // B
    }
    return new ort.Tensor('float32', array, [1, 3, size, size]);
  }

  /** Write an NCHW [-1, 1] tensor onto a canvas at its native resolution. */
  function tensorToCanvas(tensor, canvas) {
    const [, , height, width] = tensor.dims;
    const plane = width * height;
    const source = tensor.data;
    const image = new ImageData(width, height);

    for (let i = 0; i < plane; i++) {
      const p = i * 4;
      // clamp: the tanh output can drift a hair outside [-1, 1] in float32.
      image.data[p] = Math.max(0, Math.min(255, (source[i] + 1) * 127.5));
      image.data[p + 1] = Math.max(0, Math.min(255, (source[plane + i] + 1) * 127.5));
      image.data[p + 2] = Math.max(0, Math.min(255, (source[2 * plane + i] + 1) * 127.5));
      image.data[p + 3] = 255;
    }
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').putImageData(image, 0, 0);
  }

  // --------------------------------------------------------------- running
  async function translate() {
    if (!state.source || state.busy) return;
    state.busy = true;
    const label = state.direction === 'summer2winter' ? 'summer → winter' : 'winter → summer';

    try {
      setStatus(`Loading the ${label} model…`, { busy: true });
      const session = await getSession(state.direction);

      setStatus('Translating…', { busy: true });
      // Yield a frame so the spinner actually paints before the WASM backend
      // blocks the main thread.
      await new Promise((resolve) => requestAnimationFrame(resolve));

      const started = performance.now();
      const feeds = { [session.inputNames[0]]: toTensor(state.source, state.inputSize) };
      const output = await session.run(feeds);
      const elapsed = performance.now() - started;

      tensorToCanvas(output[session.outputNames[0]], el.afterCanvas);
      drawSource();
      el.result.hidden = false;
      el.afterBadge.textContent = state.direction === 'summer2winter' ? 'winter' : 'summer';
      el.timing.textContent =
        `${Math.round(elapsed)} ms · ${state.inputSize}×${state.inputSize} · ${state.backend}`;
      setPosition(50);
      setStatus('');
    } catch (err) {
      console.error(err);
      setStatus(`Could not run the model: ${err.message}`, { error: true });
    } finally {
      state.busy = false;
    }
  }

  /** Draw the original at the model's aspect ratio so both layers line up. */
  function drawSource() {
    const size = state.inputSize;
    el.beforeCanvas.width = size;
    el.beforeCanvas.height = size;
    el.beforeCanvas.getContext('2d').drawImage(state.source, 0, 0, size, size);
  }

  async function useImageSource(blobOrUrl) {
    try {
      setStatus('Reading image…', { busy: true });
      const blob = typeof blobOrUrl === 'string'
        ? await (await fetch(blobOrUrl)).blob()
        : blobOrUrl;
      state.source = await createImageBitmap(blob);
      await translate();
    } catch (err) {
      console.error(err);
      setStatus(`Could not read that image: ${err.message}`, { error: true });
    }
  }

  // ------------------------------------------------------- compare slider
  function setPosition(percent) {
    const clamped = Math.max(0, Math.min(100, percent));
    el.afterClip.style.width = `${clamped}%`;
    el.handle.style.left = `${clamped}%`;
    el.handle.setAttribute('aria-valuenow', Math.round(clamped));
    // The clipped canvas must stay the width of the *container*, not the clip,
    // or the two halves would show different scales of the same image.
    el.afterCanvas.style.width = `${el.compare.clientWidth}px`;
  }

  function initCompare() {
    let dragging = false;
    const positionFrom = (clientX) => {
      const rect = el.compare.getBoundingClientRect();
      return ((clientX - rect.left) / rect.width) * 100;
    };

    el.compare.addEventListener('pointerdown', (event) => {
      dragging = true;
      el.compare.setPointerCapture(event.pointerId);
      setPosition(positionFrom(event.clientX));
    });
    el.compare.addEventListener('pointermove', (event) => {
      if (dragging) setPosition(positionFrom(event.clientX));
    });
    el.compare.addEventListener('pointerup', (event) => {
      dragging = false;
      el.compare.releasePointerCapture(event.pointerId);
    });

    el.handle.addEventListener('keydown', (event) => {
      const current = parseFloat(el.handle.style.left) || 50;
      const step = event.shiftKey ? 10 : 2;
      if (event.key === 'ArrowLeft') setPosition(current - step);
      else if (event.key === 'ArrowRight') setPosition(current + step);
      else return;
      event.preventDefault();
    });

    window.addEventListener('resize', () => {
      if (!el.result.hidden) setPosition(parseFloat(el.handle.style.left) || 50);
    });
  }

  // --------------------------------------------------------------- samples
  async function initSamples() {
    let list = [];
    try {
      const response = await fetch(SAMPLES_URL, { cache: 'no-cache' });
      if (response.ok) list = (await response.json()).images || [];
    } catch {
      return; // samples are optional
    }
    if (!list.length) return;

    for (const name of list) {
      const thumb = new Image();
      thumb.src = `samples/${name}`;
      thumb.alt = `Sample photo ${name}`;
      thumb.tabIndex = 0;
      thumb.loading = 'lazy';
      const pick = () => useImageSource(`samples/${name}`);
      thumb.addEventListener('click', pick);
      thumb.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); }
      });
      el.sampleStrip.appendChild(thumb);
    }
    el.samples.hidden = false;
    el.randomSampleBtn.hidden = false;
    el.randomSampleBtn.addEventListener('click', () => {
      useImageSource(`samples/${list[Math.floor(Math.random() * list.length)]}`);
    });
    // Start with something on screen rather than an empty page.
    useImageSource(`samples/${list[0]}`);
  }

  // ------------------------------------------------------------------ init
  function initInputs() {
    el.fileInput.addEventListener('change', (event) => {
      const [file] = event.target.files;
      if (file) useImageSource(file);
    });

    for (const type of ['dragenter', 'dragover']) {
      el.dropZone.addEventListener(type, (event) => {
        event.preventDefault();
        el.dropZone.classList.add('dragover');
      });
    }
    for (const type of ['dragleave', 'drop']) {
      el.dropZone.addEventListener(type, (event) => {
        event.preventDefault();
        el.dropZone.classList.remove('dragover');
      });
    }
    el.dropZone.addEventListener('drop', (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file && file.type.startsWith('image/')) useImageSource(file);
      else setStatus('That does not look like an image file.', { error: true });
    });

    el.directionToggle.addEventListener('click', (event) => {
      const button = event.target.closest('button[data-direction]');
      if (!button || button.dataset.direction === state.direction) return;
      state.direction = button.dataset.direction;
      for (const other of el.directionToggle.querySelectorAll('button')) {
        const active = other === button;
        other.classList.toggle('active', active);
        other.setAttribute('aria-checked', String(active));
      }
      translate();
    });

    el.downloadBtn.addEventListener('click', () => {
      el.afterCanvas.toBlob((blob) => {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${state.direction}.png`;
        link.click();
        URL.revokeObjectURL(url);
      }, 'image/png');
    });
  }

  async function init() {
    for (const id of ids) el[id] = document.getElementById(id);

    const manifest = await loadManifest();
    if (!manifest || !manifest.models?.length) {
      el.noModels.hidden = false;
      return;
    }

    state.inputSize = manifest.input_size || 128;
    for (const model of manifest.models) state.models.set(model.direction, { ...model, session: null, loading: null });

    // Offer only the directions that were actually exported.
    for (const button of el.directionToggle.querySelectorAll('button')) {
      if (!state.models.has(button.dataset.direction)) button.remove();
    }
    const first = el.directionToggle.querySelector('button');
    if (first) {
      state.direction = first.dataset.direction;
      first.classList.add('active');
      first.setAttribute('aria-checked', 'true');
    }

    el.app.hidden = false;
    el.backendLabel.textContent = navigator.gpu ? 'WebGPU (on first run)' : 'WebAssembly';
    initInputs();
    initCompare();
    setStatus('Pick a photo to translate.');
    await initSamples();
  }

  if (typeof ort === 'undefined') {
    document.getElementById('status').textContent =
      'ONNX Runtime failed to load — check your network connection and reload.';
  } else {
    document.addEventListener('DOMContentLoaded', init);
  }
})();
