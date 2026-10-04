"""Original 3D avatar and streamed speech player."""
import json
import streamlit.components.v1 as components

from avatar_renderer import render_script


def render_ai_visualization(audio_b64="", text="", autoplay=True,
                            resume_listener_on_end=True, queue_id=None,
                            request_id=None, audio_mime="audio/wav", thinking=False):
    settings = json.dumps({
        "audio": audio_b64, "text": text[:4000], "autoplay": autoplay,
        "resume": resume_listener_on_end, "queueId": queue_id or "",
        "requestId": request_id or "", "mime": audio_mime, "thinking": thinking,
    }, ensure_ascii=False).replace("</", "<\\/")
    html = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{color-scheme:dark;font-family:Inter,Segoe UI,Arial,sans-serif;color:#f4f8f5}
*{box-sizing:border-box}html,body{margin:0;height:100%;overflow:hidden;background:transparent}
.visual{position:relative;height:100%;overflow:hidden;border-radius:22px;background:
 radial-gradient(circle at 50% 43%,rgba(53,227,122,.1),transparent 34%),
 radial-gradient(circle at 50% 48%,#0d1710,#070a08 49%,#050505 78%)}
.visual:before{content:"";position:absolute;inset:0;opacity:.22;background-image:
 linear-gradient(rgba(59,163,96,.14) 1px,transparent 1px),
 linear-gradient(90deg,rgba(59,163,96,.14) 1px,transparent 1px);background-size:44px 44px;
 mask-image:radial-gradient(circle at center,#000,transparent 70%)}
.visual:after{content:"";position:absolute;inset:0;border:1px solid rgba(117,255,157,.1);border-radius:22px;pointer-events:none}
#avatarScene{position:absolute;inset:0;width:100%;height:100%}
#avatarScene canvas{display:block;width:100%;height:100%}
.model-status{position:absolute;top:50%;left:8%;right:8%;transform:translateY(-50%);text-align:center;font-size:13px;color:#9ba9a0}
.model-status.loaded{display:none}
.credit{position:absolute;left:16px;bottom:12px;color:#627c6b;font-size:10px}.credit a{color:inherit}
.status{position:absolute;left:0;right:0;bottom:6%;display:flex;justify-content:center;align-items:center;gap:9px;
 color:#a9b6ad;font-size:12px;letter-spacing:.18em;text-transform:uppercase}
.dot{width:6px;height:6px;border-radius:50%;background:#6cbd83;box-shadow:0 0 11px rgba(102,255,153,.55)}
.audio{position:absolute;left:6%;right:6%;bottom:14%;width:88%;height:30px;opacity:0;pointer-events:none;transition:opacity .25s}
.visual[data-has-audio="true"] .audio{opacity:.8;pointer-events:auto}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
</style><script type="importmap">__IMPORT_MAP__</script></head><body>
<div class="visual" id="visual" data-state="idle" data-has-audio="false">
 <div id="avatarScene" role="img" aria-label="3D AI avatar"></div>
 <div id="modelStatus" class="model-status">Loading avatar...</div>
 <div class="credit"><a href="https://sketchfab.com/3d-models/male04-face-rigged-c18ce40998d943e690f758986fbd298d" target="_blank" rel="noopener">male04 face rigged</a> by <a href="https://sketchfab.com/Professor_E12" target="_blank" rel="noopener">photon</a> &middot; <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener">CC BY 4.0</a></div>
 <audio id="audio" class="audio" controls playsinline preload="auto"></audio>
 <div class="status" role="status" aria-live="polite"><span class="dot"></span><span id="statusText">Ready</span></div></div>
<script>
const config=__SETTINGS__, visual=document.getElementById("visual"),
 audio=document.getElementById("audio"),statusText=document.getElementById("statusText");
const activityKey="texmin_ai_activity",speakingKey="texmin_voice_avatar_speaking_at",
 interruptKey="texmin_voice_interrupt_at",startedAt=Date.now();
const ready=new Map(),clips=[];let next=0,final=null,interrupted=false,playing=false,heartbeat=null,lastActivity=0,resumed=false;
let firstPlayback=true;const visualStarted=performance.now();
function phase(value,label){
 if(playing&&value!=="speaking")return;
 visual.dataset.state=value;
 statusText.textContent=label||({idle:"Ready",listening:"Listening",transcribing:"Transcribing",
 retrieving:"Retrieving",generating:"Generating",speaking:"Speaking",
 interrupted:"Interrupted",error:"Something went wrong"}[value]||"Ready");
}
function speaking(on){
 if(heartbeat)clearInterval(heartbeat);heartbeat=null;
 if(on){const mark=()=>localStorage.setItem(speakingKey,String(Date.now()));mark();heartbeat=setInterval(mark,600)}
 else localStorage.removeItem(speakingKey);
}
function resume(){if(config.resume&&!resumed){resumed=true;localStorage.setItem("texmin_voice_resume_at",String(Date.now()))}}
function playNext(){
 if(interrupted)return;
 if(!clips.length){playing=false;if(final!==null&&next>=final){phase("idle");resume()}return}
 const item=clips.shift();visual.dataset.hasAudio="true";
 audio.src="data:"+(item.mime||config.mime)+";base64,"+item.audio;audio.load();
 audio.play().catch(()=>{playing=false;speaking(false);phase("idle");resume()});
}
function interrupt(){
 const at=Number(localStorage.getItem(interruptKey)||0);
 if(!at||at<startedAt||interrupted)return;
 interrupted=true;clips.length=0;ready.clear();audio.pause();audio.removeAttribute("src");audio.load();
 visual.dataset.hasAudio="false";speaking(false);playing=false;phase("interrupted");resume();
}
audio.addEventListener("play",()=>{
 if(firstPlayback){
  firstPlayback=false;
  console.info(JSON.stringify({event:"voice_first_playback",request_id:config.requestId,
   elapsed_from_avatar_ms:Math.round(performance.now()-visualStarted)}));
 }
 playing=true;speaking(true);phase("speaking");
});
audio.addEventListener("pause",()=>{if(!audio.ended){playing=false;speaking(false);phase("idle")}});
audio.addEventListener("ended",()=>{playing=false;speaking(false);if(config.queueId)playNext();else{phase("idle");resume()}});
window.addEventListener("message",event=>{
 const item=event.data||{};
 if(item.type!=="texmin:avatar-queue"||item.queueId!==config.queueId||interrupted)return;
 if((item.audio||item.skip)&&item.sequence>=next){
  ready.set(item.sequence,item);
  while(ready.has(next)){const current=ready.get(next);if(current.audio)clips.push(current);ready.delete(next++)}
 }
 if(item.final)final=item.finalSequence;
 if(!playing&&audio.paused)playNext();
});
function sync(){
 interrupt();if(playing||interrupted)return;
 let value;try{value=JSON.parse(localStorage.getItem(activityKey)||"null")}catch(_){value=null}
 if(value&&value.at!==lastActivity&&Date.now()-value.at<120000){lastActivity=value.at;phase(value.phase,value.label)}
 else if(!value||Date.now()-value.at>=120000)phase("idle");
}
window.addEventListener("storage",sync);setInterval(sync,250);
if(config.audio){
 visual.dataset.hasAudio="true";audio.src="data:"+config.mime+";base64,"+config.audio;
 if(config.autoplay)audio.play().catch(()=>phase("idle"));
}else if(config.thinking)phase("generating");
sync();
</script>__RENDERER_SCRIPT__</body></html>'''
    import_map, renderer_script = render_script()
    html = (html.replace("__SETTINGS__", settings)
            .replace("__IMPORT_MAP__", import_map)
            .replace("__RENDERER_SCRIPT__", renderer_script))
    components.html(html, height=620)

