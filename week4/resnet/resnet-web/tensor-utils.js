export function summarize(values) {
  let min = Infinity, max = -Infinity, sum = 0, finite = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value); max = Math.max(max, value); sum += value; finite++;
  }
  return { min: finite ? min : NaN, max: finite ? max : NaN, mean: finite ? sum / finite : NaN, finite, count: values.length };
}

export function layout(shape) {
  const dims = shape[0] === 1 ? shape.slice(1) : shape;
  if (dims.length === 3) return { height: dims[0], width: dims[1], channels: dims[2], spatial: true };
  if (dims.length === 2) return { height: dims[0], width: dims[1], channels: 1, spatial: true };
  const count = dims.reduce((a, b) => a * b, 1);
  return { height: Math.ceil(Math.sqrt(count)), width: Math.ceil(Math.sqrt(count)), channels: 1, spatial: false, count };
}

export function valueIndex(shape, channel, row, column) {
  const grid = layout(shape);
  if (row < 0 || column < 0 || row >= grid.height || column >= grid.width || channel < 0 || channel >= grid.channels) return -1;
  const index = grid.spatial ? (row * grid.width + column) * grid.channels + channel : row * grid.width + column;
  return !grid.spatial && index >= grid.count ? -1 : index;
}

export function channelValues(values, shape, channel) {
  const grid = layout(shape), output = [];
  for (let y = 0; y < grid.height; y++) for (let x = 0; x < grid.width; x++) {
    const index = valueIndex(shape, channel, y, x);
    if (index >= 0) output.push(values[index]);
  }
  return output;
}

export function softmax(values) {
  const max = Math.max(...values), exp = Array.from(values, x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(value => value / sum);
}

export function pixelColor(value, min, max) {
  if (!Number.isFinite(value)) return [237, 78, 93, 255];
  const t = max === min ? 0.5 : Math.max(0, Math.min(1, (value - min) / (max - min)));
  return [Math.round(22 + 211*t), Math.round(33 + 173*t), Math.round(60 + 67*t), 255];
}
