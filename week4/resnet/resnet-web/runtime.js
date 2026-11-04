export class ModelRuntime {
  constructor(catalog) {
    this.catalog = catalog;
    this.models = new Map();
    this.big = null;
    this.backend = null;
  }

  async initialize(preferred = 'webgpu', onStatus = () => {}) {
    if (this.backend && (!preferred || this.backend === preferred)) return this.backend;
    this.dispose();
    const candidates = preferred === 'webgpu' ? ['webgpu','webgl','wasm'] : [preferred];
    const failures = [];
    tf.wasm?.setWasmPaths(new URL('./assets/', location.href).href);
    tf.wasm?.setThreadsCount(1);
    for (const backend of candidates) {
      try {
        onStatus(`${backend} 초기화 및 실제 모델 확인 중…`);
        if (backend === 'webgpu' && !navigator.gpu) throw new Error('WebGPU API를 사용할 수 없습니다.');
        if (!await tf.setBackend(backend)) throw new Error('backend 초기화 실패');
        await tf.ready();
        this.backend = backend;
        const model = await this.loadGraph('a');
        const input = tf.zeros([1,160,160,3]);
        let output;
        try {
          output = model.execute({ [this.catalog.graphs.a.input]: input }, this.catalog.graphs.a.layers.at(-1).node);
          await output.data();
        } finally {
          output?.dispose(); input.dispose();
        }
        return backend;
      } catch (error) {
        failures.push(`${backend}: ${error.message}`);
        this.dispose();
      }
    }
    throw new Error(`모델 실행 backend를 초기화하지 못했습니다. ${failures.join(' / ')}`);
  }

  async loadGraph(id, onProgress) {
    if (this.models.has(id)) return this.models.get(id);
    const info = this.catalog.graphs[id];
    if (!info) throw new Error(`catalog.json에 ${id} 모델 정보가 없습니다.`);
    if (['a','b','c'].includes(id) && this.big && this.big !== id) {
      this.models.get(this.big)?.dispose();
      this.models.delete(this.big);
    }
    const model = await tf.loadGraphModel(info.url, { onProgress });
    this.models.set(id, model);
    if (['a','b','c'].includes(id)) this.big = id;
    return model;
  }

  path(modelId) {
    const baseId = ['d','e'].includes(modelId) ? 'a' : modelId;
    const base = this.catalog.graphs[baseId];
    let layers = base.layers;
    if (modelId === 'd' || modelId === 'e') {
      const end = modelId === 'd' ? base.pooled : base.spatial;
      layers = layers.slice(0, layers.findIndex(layer => layer.id === end) + 1);
    }
    const path = [{ graph: baseId, layers }];
    if (['d','e'].includes(modelId)) path.push({ graph: `${modelId}_head`, layers: this.catalog.graphs[`${modelId}_head`].layers });
    return path;
  }

  async run(modelId, rgb, { onLayer = async () => {}, onStatus = () => {}, trace = true } = {}) {
    const started = performance.now(), tensors = new Set(), snapshots = [];
    const runId = crypto.randomUUID();
    let input = tf.tensor(rgb, [1,160,160,3], 'float32');
    tensors.add(input);
    const path = this.path(modelId);
    const total = path.reduce((sum, part) => sum + part.layers.length, 0);
    try {
      for (const part of path) {
        onStatus(`${part.graph.toUpperCase()} 모델 로딩 중…`);
        const info = this.catalog.graphs[part.graph];
        const model = await this.loadGraph(part.graph);
        const feeds = { [info.input]: input };
        const layers = trace ? part.layers : [part.layers.at(-1)];
        for (const spec of layers) {
          let result = trace ? feeds[spec.feed] : null;
          if (!result) {
            result = model.execute(feeds, spec.node);
            if (Array.isArray(result)) [result] = result;
            tensors.add(result);
          }
          if (trace) feeds[spec.feed] = result;
          const data = Float32Array.from(await result.data());
          const snapshot = { ...spec, key: `${part.graph}/${spec.id}`, graph: part.graph,
            runId, data, shape: result.shape,
            timeMs: performance.now() - started };
          snapshots.push(snapshot);
          await onLayer(snapshot, snapshots.length, total);
          input = result;
        }
      }
      return { runId, modelId, snapshots, output: snapshots.at(-1).data,
        backend: tf.getBackend(), elapsedMs: performance.now() - started,
        snapshotBytes: snapshots.reduce((sum, item) => sum + item.data.byteLength, 0) };
    } finally {
      for (const tensor of tensors) if (!tensor.isDisposedInternal) tensor.dispose();
    }
  }

  async verifyGraph(id) {
    const root = `verification/${id}`;
    const [reference, inputBuffer, valuesBuffer] = await Promise.all([
      fetch(`${root}.json`).then(response => response.json()),
      fetch(`${root}-input.bin`).then(response => response.arrayBuffer()),
      fetch(`${root}-layers.bin`).then(response => response.arrayBuffer()),
    ]);
    const info = this.catalog.graphs[id], model = await this.loadGraph(id);
    const input = tf.tensor(new Float32Array(inputBuffer), reference.inputShape, 'float32');
    const owned = new Set([input]), feeds = { [info.input]: input };
    const failures = [], layers = [];
    try {
      for (const spec of info.layers) {
        const expected = reference.layers.find(layer => layer.id === spec.id);
        let result = feeds[spec.feed];
        if (!result) { result = model.execute(feeds, spec.node); owned.add(result); }
        feeds[spec.feed] = result;
        const actual = await result.data();
        const target = new Float32Array(valuesBuffer, expected.offset, expected.length);
        let maxAbs = 0, sumAbs = 0, outside = 0;
        if (actual.length !== target.length || JSON.stringify(result.shape) !== JSON.stringify(expected.shape)) throw new Error(`${id}/${spec.name}: shape mismatch`);
        for (let index = 0; index < actual.length; index++) {
          const diff = Math.abs(actual[index] - target[index]);
          maxAbs = Math.max(maxAbs, diff); sumAbs += diff;
          if (!Number.isFinite(diff) || diff > 1e-4 + 1e-3 * Math.abs(target[index])) outside++;
        }
        const row = { id: spec.id, name: spec.name, shape: result.shape, maxAbs, meanAbs: sumAbs / actual.length, outside, count: actual.length };
        layers.push(row);
        if (outside) failures.push(row);
      }
      return { id, backend: tf.getBackend(), layers, failures, passed: failures.length === 0 };
    } finally {
      for (const tensor of owned) if (!tensor.isDisposedInternal) tensor.dispose();
    }
  }

  dispose() {
    for (const model of this.models.values()) model.dispose();
    this.models.clear(); this.big = null; this.backend = null;
  }
}
