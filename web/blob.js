import {decodeBundle} from './core.js';

export function publicBlobUrl(value) {
  const url = new URL(value);
  if(url.protocol!=='https:' || !/^[a-z0-9-]+\.public\.blob\.vercel-storage\.com$/.test(url.hostname) || url.username || url.password || url.port || url.search || url.hash) throw new Error('Configure a public HTTPS Vercel Blob URL without credentials');
  return url;
}
async function boundedBytes(response, limit) {
  if(!response.ok) throw new Error(`Modeling dataset request failed (${response.status})`);
  if(Number(response.headers.get('content-length'))>limit) throw new Error('Modeling dataset exceeds size limit');
  const reader=response.body.getReader(), chunks=[];let length=0;
  try {
    while(true){const {value,done}=await reader.read();if(done)break;length+=value.byteLength;if(length>limit)throw new Error('Modeling dataset exceeds size limit');chunks.push(value);}
  } finally { await reader.cancel(); }
  const bytes=new Uint8Array(length);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}return bytes;
}
export async function loadBlobDataset(manifestUrl, fetcher=fetch) {
  const origin=publicBlobUrl(manifestUrl);
  const signal=AbortSignal.timeout(30000);
  const options={credentials:'omit',redirect:'error',signal};
  const manifest=JSON.parse(new TextDecoder().decode(await boundedBytes(await fetcher(origin.href,{...options,cache:'no-cache'}),100_000)));
  if(manifest.format!=='wti-modeling-deployment' || manifest.version!==1 || manifest.encoding!=='gzip' || !/^[a-f0-9]{64}$/.test(manifest.sha256) || !Number.isInteger(manifest.bytes) || manifest.bytes<=0 || manifest.bytes>30_000_000 || !Number.isInteger(manifest.decoded_bytes) || manifest.decoded_bytes<=0 || manifest.decoded_bytes>30_000_000)throw new Error('Invalid modeling deployment manifest');
  const url=publicBlobUrl(manifest.data_url);
  if(url.origin!==origin.origin || !url.pathname.endsWith(`/versions/${manifest.sha256}.json.gz`))throw new Error('Dataset URL must be a versioned object in the configured Blob store');
  const bytes=await boundedBytes(await fetcher(url.href,{...options,cache:'force-cache'}),manifest.bytes);
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
  if(bytes.length!==manifest.bytes || hash!==manifest.sha256)throw new Error('Modeling dataset integrity check failed');
  const raw=await boundedBytes(new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))),manifest.decoded_bytes);
  if(raw.length!==manifest.decoded_bytes)throw new Error('Decoded dataset size mismatch');
  const bundle=decodeBundle(JSON.parse(new TextDecoder().decode(raw)));
  if(bundle.metadata.dataset_id!==manifest.dataset_id || bundle.dates[0]!==manifest.first_date || bundle.dates.at(-1)!==manifest.last_date || !manifest.table_rows || Object.entries(manifest.table_rows).some(([name,count])=>!Array.isArray(bundle[name])||bundle[name].length!==count))throw new Error('Dataset coverage does not match manifest');
  return bundle;
}
