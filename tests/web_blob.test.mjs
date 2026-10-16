import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {gzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {loadBlobDataset, publicBlobUrl} from '../web/blob.js';
const raw=await readFile(new URL('../web/data/modeling-demo.json',import.meta.url));
const payload=JSON.parse(raw), compressed=gzipSync(raw), sha=createHash('sha256').update(compressed).digest('hex');
const origin='https://example.public.blob.vercel-storage.com';
const dates=payload.tables.panel.data.map(row=>row[payload.tables.panel.columns.indexOf('date_')].slice(0,10)).sort();
const manifest={format:'wti-modeling-deployment',version:1,encoding:'gzip',bytes:compressed.length,decoded_bytes:raw.length,sha256:sha,data_url:`${origin}/commodity-factors/modeling/wti/versions/${sha}.json.gz`,dataset_id:'synthetic',first_date:dates[0],last_date:dates.at(-1),table_rows:Object.fromEntries(Object.entries(payload.tables).map(([k,v])=>[k,v.data.length]))};
const stub=(meta=manifest,bytes=compressed)=>async url=>new Response(url.endsWith('latest.json')?JSON.stringify(meta):bytes);

test('Blob loader validates gzip, full coverage and checksum with separate cache policies',async()=>{
 const calls=[];const fetcher=async(url,options)=>{calls.push(options);return stub()(url);};
 const bundle=await loadBlobDataset(`${origin}/latest.json`,fetcher);
 assert.equal(bundle.panel.length,payload.tables.panel.data.length);
 assert.equal(calls[0].cache,'no-cache');assert.equal(calls[1].cache,'force-cache');
 assert.ok(calls.every(c=>c.credentials==='omit'&&c.redirect==='error'&&c.signal));
});
test('private, credentialed and foreign Blob URLs are rejected',async()=>{
 for(const url of ['http://example.public.blob.vercel-storage.com/a','https://example.private.blob.vercel-storage.com/a','https://token@example.public.blob.vercel-storage.com/a','https://example.public.blob.vercel-storage.com/a?token=x'])assert.throws(()=>publicBlobUrl(url));
 await assert.rejects(loadBlobDataset(`${origin}/latest.json`,stub({...manifest,data_url:manifest.data_url.replace('example.','other.')})),/configured Blob store/);
});
test('missing, corrupted, oversized and mismatched datasets fail explicitly',async()=>{
 await assert.rejects(loadBlobDataset(`${origin}/latest.json`,async()=>new Response('',{status:404})),/404/);
 await assert.rejects(loadBlobDataset(`${origin}/latest.json`,stub(manifest,Buffer.from('bad'))),/integrity/);
 await assert.rejects(loadBlobDataset(`${origin}/latest.json`,stub({...manifest,decoded_bytes:30_000_001})),/manifest/);
 await assert.rejects(loadBlobDataset(`${origin}/latest.json`,stub({...manifest,last_date:'2099-01-01'})),/coverage/);
 await assert.rejects(loadBlobDataset(`${origin}/latest.json`,stub({...manifest,decoded_bytes:100})),/size limit/);
});
