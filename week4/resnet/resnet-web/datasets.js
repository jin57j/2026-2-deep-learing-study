export function parseCSV(text) {
  const rows=[];let row=[],field='',quoted=false;
  text=text.replace(/^\uFEFF/,'');
  for(let i=0;i<=text.length;i++){
    const char=text[i];
    if(char==='"'){
      if(quoted && text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;
    }else if(!quoted && (char===','||char==='\n'||char===undefined)){
      row.push(field);field='';
      if(char!==','){if(row.some(value=>value!==''))rows.push(row);row=[];}
    }else if(char!=='\r'||quoted)field+=char;
  }
  if(quoted)throw new Error('CSV의 따옴표가 닫히지 않았습니다.');
  const headers=rows.shift()||[];
  return rows.map(values=>Object.fromEntries(headers.map((key,index)=>[key,values[index]??''])));
}

export async function loadDatasets() {
  const result={};
  await Promise.all(['base','domain','dog_breeds'].map(async dataset=>{
    const base=`../resnet-classroom/data/${dataset}/`;
    const response=await fetch(`${base}manifest.csv`);
    if(!response.ok)throw new Error(`학생 데이터 CSV를 찾지 못했습니다: ${dataset}`);
    result[dataset]=parseCSV(await response.text()).filter(row=>row.split==='test').map(row=>({
      id:row.image_id,dataset,url:base+row.path,label:Number(row.breed_id),
      box:row.xmin!==undefined?['xmin','ymin','xmax','ymax'].map(key=>Number(row[key])):null,
    }));
  }));
  return result;
}
