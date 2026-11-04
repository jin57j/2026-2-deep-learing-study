const BLOCK = /^conv([2-5])_block(\d+)(?:_([0-3]))?_(conv|bn|relu|add|out)$/;
const localName = layer => layer.name.split('/').at(-1);
const keyOf = layer => `${layer.graph}/${layer.id}`;
const dimensions = shape => shape.slice(1).join(' × ');
const operationName = layer => {
  const name = ({
  Conv2D: 'Conv', BatchNormalization: 'BN', Activation: 'ReLU', Add: 'Add',
  InputLayer: '입력', ResNetPreprocess: '전처리', ZeroPadding2D: 'Padding',
  MaxPooling2D: 'MaxPool', GlobalAveragePooling2D: 'Global average pooling',
  Dense: 'Dense', LayerNormalization: 'LayerNorm', Flatten: 'Flatten', OrderedBox: 'OrderedBox',
}[layer.type] || layer.type);
  return ['Conv2D', 'Dense'].includes(layer.type) && /\/Relu(?::0)?$/.test(layer.feed) ? `${name} + ReLU` : name;
};

export function describeLayerStructure(layer, path) {
  const graph = path.filter(item => item.graph === layer.graph);
  const name = localName(layer), match = BLOCK.exec(name);
  const residual = graph.filter(item => BLOCK.test(localName(item)));
  const prefixes = [...new Set(residual.map(item => localName(item).match(/^conv\d_block\d+/)[0]))];
  const stages = [...new Set(path.map(item => item.name.startsWith('head/') ? 'head'
    : /^conv[2-5]_block/.test(localName(item)) ? localName(item).split('_')[0]
    : /^(conv1|pool1)_/.test(localName(item)) ? 'conv1' : 'input'))];
  if (match) {
    const [, stage, number, step] = match;
    const prefix = `conv${stage}_block${number}`;
    const find = suffix => graph.find(item => localName(item) === `${prefix}_${suffix}`);
    const groups = [1, 2, 3].map(index => ({
      index, conv: find(`${index}_conv`),
      layers: ['conv', 'bn', 'relu'].map(op => find(`${index}_${op}`)).filter(Boolean),
    }));
    const main = groups.flatMap(group => group.layers);
    const shortcut = ['0_conv', '0_bn'].map(find).filter(Boolean);
    const merge = ['add', 'out'].map(find).filter(Boolean);
    const branch = step === '0' ? shortcut : step ? main : merge;
    const position = branch.findIndex(item => item.id === layer.id) + 1;
    const branchName = step === '0' ? 'Shortcut' : step ? '주 경로' : '합류';
    const convPosition = step && step !== '0' ? ` · 합성곱 ${step}/3${layer.type === 'Conv2D' ? '' : ' 뒤'}` : '';
    return {
      kind: 'residual', stages, stage: `conv${stage}`, groups, shortcut, merge,
      title: `conv${stage} · 잔차 블록 ${number} / ${prefixes.filter(p => p.startsWith(`conv${stage}_`)).length}`,
      tags: ['Residual block', 'Bottleneck'],
      context: `전체 잔차 블록 ${prefixes.indexOf(prefix) + 1} / ${prefixes.length}`,
      current: `${branchName} 연산 ${position}/${branch.length}${convPosition} · ${operationName(layer)}`,
    };
  }
  let kind, title, context, members;
  if (layer.name.startsWith('head/')) {
    kind = 'head'; title = 'Task Head'; context = '문제에 맞는 최종 출력을 계산합니다.';
    members = graph.filter(item => item.name.startsWith('head/'));
  } else if (/^(conv1|pool1)_/.test(name)) {
    kind = 'conv1'; title = 'conv1 · 입력 특징 추출'; context = 'Stem · 잔차 블록에 들어가기 전의 합성곱과 풀링입니다.';
    members = graph.filter(item => /^(conv1|pool1)_/.test(localName(item)));
  } else if (layer.type === 'GlobalAveragePooling2D') {
    kind = 'pooling'; title = '공간 특징 → 특징 벡터'; context = '각 채널의 공간 위치를 평균 내어 Head에 전달합니다.';
    members = [layer];
  } else {
    kind = 'input'; title = '입력과 전처리'; context = 'RGB 이미지를 ResNet 입력으로 준비합니다.';
    members = graph.filter(item => !item.name.startsWith('head/') && ['InputLayer', 'ResNetPreprocess'].includes(item.type));
  }
  return { kind, title, context, stages, stage: kind, members, tags: [],
    current: `연산 ${members.findIndex(item => item.id === layer.id) + 1}/${members.length} · ${operationName(layer)}` };
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function kernelIcon(shape) {
  const [height, width] = shape;
  const icon = element('span', 'kernel-icon');
  icon.style.setProperty('--kernel-width', width);
  icon.setAttribute('role', 'img'); icon.setAttribute('aria-label', `${height} × ${width} 공간 커널`);
  for (let i = 0; i < height * width; i++) icon.append(element('i'));
  return icon;
}

export function renderLayerStructure(target, layer, path, available, onSelect) {
  const info = describeLayerStructure(layer, path);
  target.replaceChildren(); target.hidden = false;
  const stages = element('div', 'stage-trail'); stages.setAttribute('aria-label', '모델 안의 현재 단계');
  for (const stage of info.stages) {
    const active = stage === info.stage || info.kind === 'pooling' && stage === 'conv5';
    const item = element('span', `stage-pill${active ? ' active' : ''}`, stage === 'input' ? '입력' : stage === 'head' ? 'Head' : stage);
    if (active) item.setAttribute('aria-current', 'step');
    stages.append(item);
  }
  target.append(stages);
  const heading = element('div', 'structure-heading'); heading.append(element('h4', '', info.title));
  for (const tag of info.tags) heading.append(element('span', 'structure-tag', tag));
  target.append(heading, element('p', 'structure-context', info.context));
  const current = element('p', 'structure-current', `현재 · ${info.current}`);
  current.setAttribute('role', 'status'); target.append(current);

  function operation(item) {
    const selected = keyOf(item) === keyOf(layer);
    const button = element('button', `operation-node${selected ? ' active' : ''}`, operationName(item));
    button.type = 'button'; button.title = item.name;
    button.disabled = !available.has(keyOf(item));
    button.setAttribute('aria-label', `${item.name}${selected ? ' · 현재 선택' : ''}`);
    if (selected) button.setAttribute('aria-current', 'step');
    button.addEventListener('click', () => onSelect(keyOf(item)));
    return button;
  }
  function sequence(items, className = 'operation-sequence') {
    const row = element('div', className);
    for (const [index, item] of items.entries()) {
      if (index) row.append(element('span', 'flow-arrow', '→'));
      row.append(operation(item));
    }
    return row;
  }
  if (info.kind === 'residual') {
    const diagram = element('div', 'residual-diagram');
    diagram.append(element('div', 'block-input', '블록 입력 x'));
    const main = element('div', 'main-branch'); main.append(element('p', 'path-label', '주 경로 F(x) ↓'));
    for (const group of info.groups) {
      const selected = group.layers.some(item => item.id === layer.id);
      const card = element('div', `conv-group${selected ? ' active' : ''}`);
      const kernel = group.conv.convolution?.kernelShape;
      const heading = element('div', 'conv-heading');
      if (kernel) heading.append(kernelIcon(kernel));
      const labels = element('div'); labels.append(element('strong', '', `Conv ${group.index}/3 · ${kernel ? kernel.slice(0, 2).join('×') : '크기 미확인'}`));
      if (kernel) labels.append(element('small', '', `입력 ${kernel[2]}ch → 필터 ${kernel[3]}개`));
      heading.append(labels); card.append(heading, sequence(group.layers)); main.append(card);
    }
    const shortcut = element('div', 'shortcut-branch');
    shortcut.append(element('p', 'path-label', 'Shortcut ↓'));
    if (info.shortcut.length) {
      const conv = info.shortcut[0].convolution;
      if (conv) shortcut.append(kernelIcon(conv.kernelShape), element('strong', '', `${conv.kernelShape.slice(0, 2).join('×')} Conv`),
        element('small', '', `${conv.kernelShape[2]} → ${conv.kernelShape[3]}ch`), element('small', '', `stride ${conv.strides.join('×')}`));
      shortcut.append(sequence(info.shortcut, 'shortcut-operations'));
      shortcut.append(element('p', 'shortcut-note', '공간 크기·채널 맞춤'));
    } else {
      shortcut.append(element('strong', '', 'Identity'), element('p', 'shortcut-note', '입력 x를 그대로 전달'));
    }
    shortcut.append(element('span', 'shortcut-arrow', '↓'));
    const merge = element('div', 'merge-branch'); merge.append(element('span', 'path-label', '두 경로 합류 ↓'), sequence(info.merge));
    diagram.append(main, shortcut, merge); target.append(diagram);
  } else {
    target.append(sequence(info.members, 'operation-sequence plain-sequence'));
  }

  const details = element('dl', 'operation-details');
  const add = (label, value) => details.append(element('dt', '', label), element('dd', '', value));
  if (layer.convolution) {
    const { kernelShape: [h, w, input, filters], strides, padding } = layer.convolution;
    if (info.kind !== 'residual') {
      const preview = element('div', 'standalone-kernel');
      preview.append(kernelIcon([h, w]), element('strong', '', `${h} × ${w} 공간 커널`));
      target.append(preview);
    }
    add('현재 커널', `${h} × ${w} · 필터 1개 ${h} × ${w} × ${input}`);
    add('필터 개수', `${filters}개 → 출력 ${filters}채널`);
    add('Stride / Padding', `${strides.join(' × ')} / ${padding}`);
    target.append(element('p', 'kernel-explanation', '필터 하나가 입력 채널 전체를 사용해 출력 채널 하나를 만듭니다.'));
  } else if (layer.pooling) {
    add('풀링 창', `${layer.pooling.size.join(' × ')} · stride ${layer.pooling.strides.join(' × ')}`);
    add('현재 커널', '학습하는 합성곱 커널 없음 · 창 안의 최댓값 선택');
  } else {
    add('현재 커널', '해당 없음 · 현재 연산은 합성곱이 아닙니다.');
  }
  const parents = layer.parents.map(id => path.find(item => item.graph === layer.graph && item.id === id)).filter(Boolean);
  add('입력 → 출력', `${parents.length ? parents.map(item => dimensions(item.shape)).join(' + ') : dimensions(layer.shape)} → ${dimensions(layer.shape)}`);
  target.append(details, element('p', 'structure-hint', '계산된 연산을 누르면 해당 레이어의 출력으로 이동합니다.'));
}
