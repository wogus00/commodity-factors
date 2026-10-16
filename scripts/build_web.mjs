// Static production build: validate the checked-in modeling payload, copy assets.
import {readFile, mkdir, cp, readdir, writeFile} from 'node:fs/promises';
import {resolve, relative} from 'node:path';
import {decodeBundle} from '../web/core.js';
import {publicBlobUrl} from '../web/blob.js';
const root = resolve(import.meta.dirname, '..');
const allowed = ['blob.js','modeling-source.json','app.js','core.js','views.js','styles.css','index.html','favicon.svg','vendor','data'];
const entries = await readdir(resolve(root,'web'));
if(entries.some(name=>!allowed.includes(name))) throw new Error('Unexpected frontend asset; review before publishing');
const dataFiles = await readdir(resolve(root,'web/data'));
if(dataFiles.some(name=>name!=='modeling-demo.json')) throw new Error('Keep empirical/local exports outside the public tree');
const data = decodeBundle(JSON.parse(await readFile(resolve(root,'web/data/modeling-demo.json'),'utf8')));
if(data.metadata.dataset_id!=='synthetic') throw new Error('Public example must remain synthetic');
const html = await readFile(resolve(root,'web/index.html'),'utf8');
for(const match of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
  if(match[1]==='./')continue;
  if(!allowed.includes(match[1].split('/')[0])) throw new Error(`Unexpected asset reference: ${match[1]}`);
  await readFile(resolve(root,'web',match[1]));
}
if(/id="horizon-select"|daily-target|data-view="daily"|daily\.css|remote\.js/.test(html)) throw new Error('Legacy daily forecast interface still present');
const destination = resolve(root,'dist');
await mkdir(destination,{recursive:true});
// Delete no user files; fail rather than leave unrecognized stale deploy artifacts.
async function files(directory) {
  const entries=await readdir(directory,{withFileTypes:true});
  return (await Promise.all(entries.map(entry=>entry.isDirectory()?files(resolve(directory,entry.name)):[resolve(directory,entry.name)]))).flat();
}
const sourceFiles=new Set((await files(resolve(root,'web'))).map(path=>relative(resolve(root,'web'),path)));
if((await files(destination)).some(path=>!sourceFiles.has(relative(destination,path)))) throw new Error('Unexpected files in dist; use a clean build directory');
await cp(resolve(root,'web'),destination,{recursive:true});
const config=JSON.parse(await readFile(resolve(root,'web/modeling-source.json'),'utf8'));
const manifest=process.env.MODELING_MANIFEST_URL ?? config.manifest_url;
if(typeof manifest!=='string')throw new Error('Invalid modeling source configuration');
if(manifest)publicBlobUrl(manifest);
await writeFile(resolve(destination,'modeling-source.json'),JSON.stringify({manifest_url:manifest})+'\n');
console.log(`Built modeling dashboard in ${destination}`);
