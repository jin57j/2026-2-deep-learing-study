const OPERATIONS = {
  conv: '학습한 필터로 각 위치의 입력 값을 조합해 특징을 추출합니다.',
  bn: 'Batch Normalization. 학습 때 저장한 통계와 학습한 계수로 채널별 값의 크기를 정리합니다.',
  relu: 'ReLU. 음수는 0으로 바꾸고 양수는 그대로 통과시킵니다.',
  add: '새로 계산한 특징과 shortcut으로 전달된 특징을 같은 위치끼리 더합니다. Skip Connection이 합쳐지는 지점입니다.',
  out: '덧셈 뒤 ReLU를 적용한 잔차 블록의 출력입니다. 다음 블록으로 전달됩니다.',
  pad: '가장자리에 0을 덧붙여 뒤따르는 합성곱이나 pooling을 준비합니다.',
  pool: 'Max Pooling. 주변 영역에서 큰 값을 골라 공간 크기를 줄입니다.',
};

const TYPES = {
  InputLayer: '이미지 또는 앞부분에서 계산한 특징을 이 그래프의 입력으로 받습니다.',
  ResNetPreprocess: 'RGB 채널을 BGR 순서로 바꾸고 채널별 평균값을 빼서, 사전학습 ResNet에 맞는 입력으로 변환합니다.',
  ZeroPadding2D: OPERATIONS.pad,
  Conv2D: OPERATIONS.conv,
  BatchNormalization: OPERATIONS.bn,
  Activation: OPERATIONS.relu,
  Add: OPERATIONS.add,
  MaxPooling2D: OPERATIONS.pool,
  GlobalAveragePooling2D: '채널마다 모든 공간 위치의 값을 평균 내어 하나의 특징 벡터로 모읍니다.',
  Dense: '입력 특징을 학습한 가중치로 합하고 편향을 더해, 다음 특징이나 최종 예측 점수를 만듭니다.',
  LayerNormalization: '각 공간 위치에서 채널들의 평균과 분산을 구해 값을 정규화하고, 학습한 계수로 조정합니다.',
  Flatten: '공간과 채널에 놓인 값을 순서대로 펼쳐 하나의 벡터로 만듭니다. 값 자체는 바꾸지 않습니다.',
  OrderedBox: 'Sigmoid와 좌표 조합으로 이미지 안의 왼쪽 위·오른쪽 아래 좌표를 만듭니다. 좌우·상하 순서가 뒤집히지 않게 합니다.',
};

const NAMED = {
  rgb: ['RGB', '빨강·초록·파랑 채널을 가진 입력 이미지입니다.'],
  ResNetPreprocess: ['ResNetPreprocess', TYPES.ResNetPreprocess],
  input: ['input', '앞부분에서 얻은 특징을 이 Head의 입력으로 받습니다.'],
  input_layer: ['input_layer', TYPES.InputLayer],
  global_average_pooling: ['global_average_pooling', TYPES.GlobalAveragePooling2D],
  dense: ['dense', TYPES.Dense],
  conv2d: ['conv2d', '2차원 합성곱입니다. 이 위치 예측 Head에서는 각 위치의 채널 특징을 조합한 뒤 ReLU를 적용합니다.'],
  layer_normalization: ['layer_normalization', TYPES.LayerNormalization],
  flatten: ['flatten', TYPES.Flatten],
  ordered_box: ['ordered_box', TYPES.OrderedBox],
};

/** Explain the Keras ResNet50 names used by the exported classroom graphs. */
export function describeLayer({name, type}) {
  const localName = name.split('/').at(-1);
  const terms = name.startsWith('head/')
    ? [['head', '공통 특징을 받아 이번 문제의 답을 만드는, Task별 마지막 부분입니다.']]
    : [];
  const block = /^conv(\d+)_block(\d+)(?:_([0-3]))?_(conv|bn|relu|add|out)$/.exec(localName);
  if (block) {
    const [, stage, number, step, operation] = block;
    terms.push(
      ['conv', '합성곱 중심의 특징 추출 단계(stage)를 가리키는 이름입니다. 이 레이어 자체의 연산 종류는 마지막 단어로 구분합니다.'],
      [stage, `특징 추출 단계 conv${stage}의 번호입니다. 전체 모델의 ${stage}번째 레이어라는 뜻은 아닙니다.`],
      ['block', '특징을 새로 계산하는 경로와 shortcut 경로를 묶은 잔차 블록입니다.'],
      [number, `conv${stage} 단계 안에서 ${number}번째 잔차 블록입니다.`],
    );
    if (step !== undefined) terms.push([step, step === '0'
      ? 'shortcut 경로입니다. 덧셈 전에 두 경로의 공간 크기와 채널 수를 맞추는 데 사용합니다.'
      : `새 특징을 계산하는 주 경로의 ${step}번째 합성곱 단계입니다. 전체 레이어 번호나 채널 수가 아닙니다.`]);
    terms.push([operation, OPERATIONS[operation]]);
    const branch = step === '0' ? '의 shortcut 경로' : step ? `의 주 경로 ${step}단계` : '';
    return {summary: `conv${stage} 단계의 ${number}번째 잔차 블록${branch}: ${OPERATIONS[operation]}`, terms};
  }
  const stem = /^(conv|pool)(\d+)_(pad|conv|bn|relu|pool)$/.exec(localName);
  if (stem) {
    const [, family, number, operation] = stem;
    terms.push(
      [family, family === 'conv' ? '잔차 블록들에 들어가기 전, 처음 특징을 추출하는 합성곱 부분입니다.' : '입력 쪽에서 공간 크기를 줄이는 pooling 부분입니다.'],
      [number, `${family}${number} 부분을 구분하는 번호입니다. 전체 모델의 레이어 순서가 아닙니다.`],
      [operation, OPERATIONS[operation]],
    );
    return {summary: OPERATIONS[operation], terms};
  }
  const numbered = /^(input_layer|dense|conv2d|layer_normalization|flatten|ordered_box)_(\d+)$/.exec(localName);
  const base = numbered ? numbered[1] : localName;
  if (NAMED[base]) terms.push(NAMED[base]);
  else terms.push([localName, TYPES[type] || `${type} 연산으로 입력을 다음 표현으로 변환합니다.`]);
  if (numbered) terms.push([numbered[2], 'Keras가 이름의 중복을 피하려고 붙인 식별 번호입니다. 뉴런 수나 모델 안의 레이어 순서를 뜻하지 않습니다.']);
  return {summary: TYPES[type] || `${type} 연산으로 입력을 다음 표현으로 변환합니다.`, terms};
}
