import { ModelRuntime } from './runtime.js';
import { MODEL_INFO, DATASET_NAMES } from './model-info.js';
import { loadDatasets } from './datasets.js';
import { describeLayer } from './layer-help.js';
import { renderLayerStructure } from './layer-structure.js';
import { layout, valueIndex, channelValues, pixelColor, softmax, summarize } from './tensor-utils.js';

const $ = id => document.getElementById(id);
const state = { model: 'a', row: null, rgb: null, image: null, snapshots: [], path: [], selected: -1, channel: 0, page: 0, running: false, follow: true, result: null };
const format = value => Number.isFinite(value) ? Number(value).toPrecision(5) : String(value);
let catalog, datasets, runtime;

function message(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
function options(select, values) { select.replaceChildren(...values.map(([value, title]) => new Option(title, value))); }
function fields(target, values) {
  target.replaceChildren();
  for (const [label, value] of values) {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = label; dd.textContent = value; target.append(dt, dd);
  }
}
function lock(busy) {
  state.running = busy;
  for (const id of ['model','dataset','sample','run']) $(id).disabled = busy;
}
function resetTrace() {
  state.snapshots = []; state.result = null; state.selected = -1; state.follow = true;
  $('progress').value = 0; $('trace-count').textContent = '실행 전';
  $('result').replaceChildren(Object.assign(document.createElement('p'), {className:'muted',textContent:'실행하면 모델의 예측을 보여줍니다.'}));
  $('layer-name').textContent = '실행 후 레이어를 선택하세요';
  $('layer-help').hidden = true;
  $('layer-structure').replaceChildren(); $('layer-structure').hidden = true;
  $('layer-operation').textContent = ''; $('layer-terms').replaceChildren();
  for (const id of ['layer-shape','layer-parents','numbers','color-scale']) $(id).textContent = '';
  for (const id of ['overview','zoom']) $(id).getContext('2d').clearRect(0,0,512,512);
  $('cell-value').textContent = '셀을 가리키거나 눌러 실제 값을 확인하세요.';
  buildLayerList();
}
function buildLayerList() {
  state.path = runtime.path(state.model).flatMap(part => part.layers.map(layer => ({...layer,key:`${part.graph}/${layer.id}`,graph:part.graph})));
  $('layers').replaceChildren();
  state.path.forEach((layer,index) => {
    const button = document.createElement('button'); button.type='button'; button.className='layer-item pending'; button.disabled=true;
    button.dataset.index=index;
    const name=document.createElement('span'), info=document.createElement('small');
    name.className='layer-name'; name.textContent=`${String(index+1).padStart(3,'0')} · ${layer.name}`;
    info.textContent=`${layer.type} · [${layer.shape.join(', ')}]`;
    button.append(name,info); button.addEventListener('click',()=>{state.follow=false;selectLayer(index);});
    $('layers').append(button);
  });
  filterLayers();
}
function filterLayers() {
  const term=$('layer-search').value.trim().toLowerCase();
  for(const button of $('layers').children) button.hidden=!button.textContent.toLowerCase().includes(term);
}

async function renderModel() {
  state.model=$('model').value;
  const info=MODEL_INFO[state.model];
  $('problem').textContent=info.problem;
  fields($('model-summary'),[['입력',info.input],['출력',info.output],['출발 모델',info.start],['학습 범위',info.scope]]);
  fields($('training'),[['학습 데이터',info.data],['Train / Valid / Test',info.split],['Epoch / Batch',`${info.epochs} / ${info.batch}`],['Optimizer',`Adam · learning rate ${info.learningRate} · beta1=0.9, beta2=0.999, epsilon=1e-7, amsgrad=False${state.model==='e'?' · clipnorm=1':''}`],['Loss',info.loss],['전역 seed / 저장','42 · 정해진 마지막 epoch (early stopping·scheduler 없음)'],['구조','ResNet50 · 분류 Head Dense64/ReLU → Dense6 또는 Dense25. 위치 Head는 위 학습 범위 참조.'],['증강 / 초기화',info.augmentation],['데이터 출처','Oxford-IIIT Pet의 교육용 재분할 · RGB 160×160 letterbox · CC BY-SA 4.0']]);
  const source=catalog.graphs[['d','e'].includes(state.model)?`${state.model}_head`:state.model];
  $('weight-source').textContent=`제공 가중치: ${source.source} · SHA-256 ${source.sha256}${['d','e'].includes(state.model)?' · 공통 특징: A 가중치':''}`;
  options($('dataset'),info.datasets.map(id=>[id,DATASET_NAMES[id]]));
  await renderDataset();
}
async function renderDataset() {
  const rows=datasets[$('dataset').value];
  options($('sample'),rows.map((row,index)=>[String(index),`${row.id} · ${state.model==='e'?'머리 box':labels()[row.label]}`]));
  await renderSample();
}
function labels() { return state.model==='d'?catalog.dogLabels:catalog.breedLabels; }
async function renderSample() {
  state.row=datasets[$('dataset').value][Number($('sample').value)];
  resetTrace(); state.rgb=null;
  const row=state.row, image=new Image(); image.src=row.url;
  await image.decode();
  if(state.row!==row) return;
  state.image=image;
  const source=document.createElement('canvas');source.width=160;source.height=160;
  const ctx=source.getContext('2d',{willReadFrequently:true});ctx.drawImage(image,0,0);
  const rgba=ctx.getImageData(0,0,160,160).data;
  state.rgb=new Float32Array(160*160*3);
  for(let i=0,j=0;i<rgba.length;i+=4){state.rgb[j++]=rgba[i];state.rgb[j++]=rgba[i+1];state.rgb[j++]=rgba[i+2];}
  $('sample-id').textContent=`${row.id} · ${row.dataset} / Test · 160×160`;
  $('expected').textContent=state.model==='e'?`머리 xyxy = (${row.box.map(x=>x.toFixed(4)).join(', ')})`:`${labels()[row.label]} · label ${row.label}`;
  drawInput();
  message(`Test 이미지와 정답을 확인했습니다. 실행하면 ${state.path.length}개 레이어의 실제 값을 계산합니다.`);
}
function drawInput(predicted) {
  const ctx=$('image').getContext('2d');ctx.clearRect(0,0,160,160);if(!state.image)return;ctx.drawImage(state.image,0,0);
  if(state.model!=='e')return;
  for(const [box,color] of [[state.row.box,'#2da56a'],[predicted,'#ef425a']]){
    if(!box)continue;ctx.strokeStyle=color;ctx.lineWidth=1.7;
    ctx.strokeRect(box[0]*160,box[1]*160,(box[2]-box[0])*160,(box[3]-box[1])*160);
  }
}
async function execute() {
  if(!state.rgb || state.running)return;
  lock(true);resetTrace();const expectedRow=state.row;
  try {
    $('run').textContent='계산 중…';
    state.result=await runtime.run(state.model,state.rgb,{
      onStatus:message,
      onLayer:async(snapshot,count,total)=>{
        state.snapshots.push(snapshot);
        const button=$('layers').children[count-1];button.disabled=false;button.classList.remove('pending');
        $('progress').value=count/total;$('trace-count').textContent=`${count} / ${total} 레이어`;
        message(`${snapshot.name} 계산 완료 · shape [${snapshot.shape.join(', ')}]`);
        if(state.follow)selectLayer(count-1);
        await new Promise(requestAnimationFrame);
      },
    });
    if(state.row!==expectedRow)throw new Error('입력이 변경됐습니다. 다시 실행해 주세요.');
    renderResult();
    if (state.selected >= 0) renderStructure(state.snapshots[state.selected]);
    message(`추론 완료 · ${state.result.snapshots.length}개 레이어 · ${(state.result.elapsedMs/1000).toFixed(2)}초 · 값 ${(state.result.snapshotBytes/1e6).toFixed(1)} MB 보존. 레이어를 선택해 다시 확인할 수 있습니다.`);
  } catch(error){message(`실행 실패: ${error.message}`,true);console.error(error);}
  finally{$('run').textContent='추론 실행';lock(false);}
}
function renderResult() {
  $('result').replaceChildren();const output=state.result.output;
  const title=document.createElement('p');title.className='prediction';
  if(state.model==='e'){
    title.textContent='머리 위치 예측 완료';$('result').append(title);
    const text=document.createElement('p');text.className='mono';text.textContent=`예측 xyxy = (${Array.from(output,x=>x.toFixed(4)).join(', ')})`;
    $('result').append(text,Object.assign(document.createElement('p'),{className:'footnote',textContent:'초록: 정답 · 빨강: 예측'}));drawInput(output);return;
  }
  const probabilities=softmax(output), order=probabilities.map((value,index)=>({value,index})).sort((a,b)=>b.value-a.value);
  title.textContent=`${labels()[order[0].index]} · ${order[0].index===state.row.label?'정답':'오답'}`;$('result').append(title);
  const details=document.createElement('details');details.open=order.length<=6;
  const summary=document.createElement('summary');summary.textContent=`전체 ${order.length}개 클래스 확률`;details.append(summary);
  for(const {value,index} of order){const row=document.createElement('div');row.className='prob-row';const name=document.createElement('span');name.textContent=labels()[index];const bar=document.createElement('progress');bar.max=1;bar.value=value;const number=document.createElement('span');number.textContent=`${(value*100).toFixed(2)}%`;row.append(name,bar,number);details.append(row);}
  $('result').append(details);
}

function selectLayer(index) {
  const snapshot=state.snapshots[index];if(!snapshot)return;
  state.selected=index;state.channel=0;state.page=0;
  for(const button of $('layers').children)button.classList.toggle('active',Number(button.dataset.index)===index);
  $('layer-name').textContent=`${snapshot.name} · ${snapshot.type}`;
  $('layer-shape').textContent=`shape [${snapshot.shape.join(', ')}] · ${snapshot.data.length.toLocaleString()}개 값 · ${snapshot.graph.toUpperCase()}`;
  const help=describeLayer(snapshot);
  $('layer-operation').textContent=help.summary;
  fields($('layer-terms'),help.terms);$('layer-help').hidden=false;
  renderStructure(snapshot);
  const graph=catalog.graphs[snapshot.graph];
  $('layer-parents').textContent=`입력 연결: ${snapshot.parents.map(id=>graph.layers.find(layer=>layer.id===id)?.name||id).join(' + ')||'그래프 입력'} · 현재 이미지의 저장된 값`;
  const grid=layout(snapshot.shape);for(const id of ['channel','channel-slider']){$(id).max=grid.channels-1;$(id).value=0;}
  renderGrids();
}
function renderStructure(snapshot) {
  renderLayerStructure($('layer-structure'), snapshot, state.path,
    new Set(state.snapshots.map(item => item.key)), key => {
      const index = state.snapshots.findIndex(item => item.key === key);
      if (index >= 0) { state.follow = false; selectLayer(index); }
    });
}
function makeTile(snapshot,channel) {
  const grid=layout(snapshot.shape), values=channelValues(snapshot.data,snapshot.shape,channel),stats=summarize(values),n=Math.max(grid.height,grid.width);
  const canvas=document.createElement('canvas');canvas.width=n;canvas.height=n;
  const context=canvas.getContext('2d'),pixels=context.createImageData(n,n);
  for(let y=0;y<n;y++)for(let x=0;x<n;x++){
    const index=valueIndex(snapshot.shape,channel,y,x),offset=(y*n+x)*4;
    const color=index<0?[237,241,246,255]:pixelColor(snapshot.data[index],stats.min,stats.max);
    pixels.data.set(color,offset);
  }
  context.putImageData(pixels,0,0);return{canvas,stats,grid};
}
function renderGrids() {
  const snapshot=state.snapshots[state.selected];if(!snapshot)return;
  const grid=layout(snapshot.shape),ctx=$('overview').getContext('2d');ctx.clearRect(0,0,512,512);ctx.imageSmoothingEnabled=false;
  const begin=state.page*64,end=Math.min(begin+64,grid.channels);
  for(let channel=begin;channel<end;channel++){
    const i=channel-begin,x=(i%8)*64,y=Math.floor(i/8)*64,tile=makeTile(snapshot,channel);
    ctx.drawImage(tile.canvas,x+2,y+2,60,48);ctx.fillStyle='#526476';ctx.font='10px monospace';ctx.fillText(`ch ${channel}`,x+3,y+61);
    if(channel===state.channel){ctx.strokeStyle='#18869b';ctx.lineWidth=2;ctx.strokeRect(x+1,y+1,62,62);}
  }
  const tile=makeTile(snapshot,state.channel),zoom=$('zoom').getContext('2d');zoom.clearRect(0,0,512,512);zoom.imageSmoothingEnabled=false;zoom.drawImage(tile.canvas,0,0,512,512);
  $('page-label').textContent=`채널 ${begin}–${end-1} / ${grid.channels}개`;
  $('previous').disabled=begin===0;$('next').disabled=end===grid.channels;
  $('color-scale').textContent=`선택 채널 ${state.channel} · 색 범위 ${format(tile.stats.min)} ~ ${format(tile.stats.max)} · 채널별 정규화. 밝은 빈칸은 값이 없습니다.`;
  if($('numbers').parentElement.open)renderNumbers(snapshot,grid);
}
function renderNumbers(snapshot,grid) {
  const table=document.createElement('table'),body=document.createElement('tbody');
  for(let y=0;y<grid.height;y++){
    const row=document.createElement('tr');
    for(let x=0;x<grid.width;x++){
      const index=valueIndex(snapshot.shape,state.channel,y,x),cell=document.createElement('td');
      cell.textContent=index<0?'—':format(snapshot.data[index]);cell.title=index<0?'빈칸':String(snapshot.data[index]);
      if(index<0)cell.className='empty';row.append(cell);
    }
    body.append(row);
  }
  table.append(body);$('numbers').replaceChildren(table);
}
function setChannel(value) {
  const snapshot=state.snapshots[state.selected];if(!snapshot)return;
  state.channel=Math.max(0,Math.min(layout(snapshot.shape).channels-1,Math.trunc(Number(value)||0)));
  state.page=Math.floor(state.channel/64);
  for(const id of ['channel','channel-slider'])$(id).value=state.channel;
  renderGrids();
}
function inspectCell(event) {
  const snapshot=state.snapshots[state.selected];if(!snapshot)return;
  const grid=layout(snapshot.shape),rect=$('zoom').getBoundingClientRect(),n=Math.max(grid.width,grid.height);
  const column=Math.min(n-1,Math.floor((event.clientX-rect.left)/rect.width*n)),row=Math.min(n-1,Math.floor((event.clientY-rect.top)/rect.height*n));
  const index=valueIndex(snapshot.shape,state.channel,row,column);
  $('cell-value').textContent=index<0?`(${row}, ${column}) · 빈칸`:`ch ${state.channel} · (${row}, ${column}) · 원소 ${index} = ${snapshot.data[index]}`;
}

async function init() {
  lock(true);
  try{
    [catalog,datasets]=await Promise.all([fetch('catalog.json').then(r=>{if(!r.ok)throw new Error('catalog.json을 찾을 수 없습니다. 웹 폴더에 파일이 있는지 확인해 주세요.');return r.json();}),loadDatasets()]);
    runtime=new ModelRuntime(catalog);
    options($('model'),Object.entries(MODEL_INFO).map(([id,info])=>[id,info.title]));
    for(const option of $('model').options)option.disabled=!catalog.graphs[['d','e'].includes(option.value)?`${option.value}_head`:option.value];
    await renderModel();
    const backend=await runtime.initialize('webgpu',message);$('backend').textContent=`${backend.toUpperCase()} · 로컬 추론`;
    lock(false);message(`준비 완료 · ${state.path.length}개 레이어. Test 이미지를 선택하고 실행하세요.`);
    window.resnetLab={runtime,catalog,datasets,get state(){return state;},execute};
  }catch(error){message(`준비 실패: ${error.message}`,true);console.error(error);}
}
$('model').addEventListener('change',()=>renderModel().catch(error=>message(error.message,true)));
$('dataset').addEventListener('change',()=>renderDataset().catch(error=>message(error.message,true)));
$('sample').addEventListener('change',()=>renderSample().catch(error=>message(error.message,true)));
$('run').addEventListener('click',execute);$('layer-search').addEventListener('input',filterLayers);
for(const id of ['channel','channel-slider'])$(id).addEventListener('input',event=>setChannel(event.target.value));
$('previous').addEventListener('click',()=>setChannel(Math.max(0,(state.page-1)*64)));
$('next').addEventListener('click',()=>setChannel((state.page+1)*64));
$('overview').addEventListener('click',event=>{const rect=$('overview').getBoundingClientRect();setChannel(state.page*64+Math.floor((event.clientY-rect.top)/rect.height*8)*8+Math.floor((event.clientX-rect.left)/rect.width*8));});
$('zoom').addEventListener('pointermove',inspectCell);$('zoom').addEventListener('click',inspectCell);
$('numbers').parentElement.addEventListener('toggle',()=>{const snapshot=state.snapshots[state.selected];if(snapshot && $('numbers').parentElement.open)renderNumbers(snapshot,layout(snapshot.shape));});
await init();
