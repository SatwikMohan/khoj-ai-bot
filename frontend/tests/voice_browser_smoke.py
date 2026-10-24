"""Headless browser smoke for the hands-free Streamlit component.

Run directly: python frontend/tests/voice_browser_smoke.py
The browser microphone, recorder, and server bridge are deterministic fakes.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "voice_query_component" / "index.html"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
MOCKS = r"""
<script>
const events = [];
let begins = 0, audios = 0, earlyBegin = false, earlyAudio = false;
let speaking = false, speechEndAt = 0, completeHeaders = true, recorderStarts = 0, mediaAcquires = 0;
const result = document.createElement("output");
result.id = "result";
document.addEventListener("DOMContentLoaded", () => document.body.appendChild(result));
Object.defineProperty(window, "isSecureContext", {value:true});
Object.defineProperty(navigator, "mediaDevices", {value:{
  getUserMedia:async()=>{
    mediaAcquires++;
    const track={label:"fake microphone",readyState:"live",stop(){this.readyState="ended";}};
    return {getTracks:()=>[track],getAudioTracks:()=>[track]};
  }
}});
class FakeRecorder {
  static isTypeSupported() {return true;}
  constructor() {this.mimeType="audio/webm";this.state="inactive";}
  start() {
    this.state="recording";
    recorderStarts++;
    this.ondataavailable?.({data:new Blob(["HEADER"])});
    this.timer=setInterval(()=>this.ondataavailable?.({data:new Blob(["data"])}),10);
  }
  stop() {
    if(this.state==="inactive") return;
    this.state="inactive";clearInterval(this.timer);
    this.ondataavailable?.({data:new Blob(["last"])});
    setTimeout(()=>this.onstop?.(),0);
  }
}
window.MediaRecorder=FakeRecorder;
const NativeBlob = window.Blob;
window.Blob = class extends NativeBlob {
  constructor(parts, options) {
    super(parts, options);
    this.hasHeader = parts[0] === "HEADER" || parts[0]?.hasHeader === true;
  }
};
class FakeAudioContext {
  async resume() {}
  async close() {}
  createAnalyser() {return {fftSize:512,getByteTimeDomainData(values) {
    values.fill(128);
    if(speaking) for(let i=0;i<values.length;i++) values[i]=i%2?140:116;
  }};}
  createMediaStreamSource() {return {connect(){}};}
}
window.AudioContext=FakeAudioContext;
window.FileReader=class {
  readAsDataURL(blob){
    completeHeaders = completeHeaders && blob.hasHeader;
    this.result="data:audio/webm;base64,ZGF0YQ==";
    setTimeout(()=>this.onload?.(),0);
  }
};
function render(accepted="") {
  window.postMessage({type:"streamlit:render",args:{
    session_id:"smoke-session",accepted_request_id:accepted,
    speech_confirm_ms:100,silence_ms:300,max_recording_seconds:5,recording_buffer_ms:50
  }},"*");
}
function finishTurn(id, awaitAudio) {
  const turnKey="texmin_voice_turn:smoke-session";
  const audioKey="texmin_voice_audio_state:smoke-session";
  if(awaitAudio) localStorage.setItem(audioKey,JSON.stringify({
    requestId:id,state:"pending",at:Date.now()
  }));
  localStorage.setItem(turnKey,JSON.stringify({
    requestId:id,state:"finished",awaitAudio,at:Date.now()
  }));
  if(awaitAudio) {
    const before=begins;
    setTimeout(()=>{
      if(begins!==before) earlyBegin=true;
      localStorage.setItem(audioKey,JSON.stringify({
        requestId:id,state:"completed",at:Date.now()
      }));
    },250);
  }
}
window.addEventListener("message",e=>{
  const item=e.data||{};
  if(item.type==="streamlit:componentReady") {
    render();
    setTimeout(()=>document.getElementById("micButton").click(),40);
  }
  if(item.type!=="streamlit:setComponentValue") return;
  const value=item.value||{};
  events.push(value.kind);
  if(value.kind==="begin") {
    begins++;
    render(value.request_id);
    if(begins<=3) {
      speechEndAt=0;
      setTimeout(()=>{speaking=true;},150);
      setTimeout(()=>{speaking=false;},450);
      setTimeout(()=>{speaking=true;},590);
      setTimeout(()=>{speaking=false;speechEndAt=Date.now();},950);
    }
  }
  if(value.kind==="audio") {
    audios++;
    if (!speechEndAt || Date.now() < speechEndAt) earlyAudio=true;
    const id=value.request_id;
    setTimeout(()=>finishTurn(id,audios<3),70);
  }
});
setTimeout(()=>{
  const at=Date.now();
  localStorage.setItem("texmin_voice_interrupt_at",String(at));
  window.dispatchEvent(new StorageEvent("storage",{
    key:"texmin_voice_interrupt_at",newValue:String(at)
  }));
},5100);
setTimeout(()=>{
  const text="begins="+begins+" audios="+audios+" early="+earlyBegin+" earlyAudio="+earlyAudio+
    " headers="+completeHeaders+" recorderStarts="+recorderStarts+" acquires="+mediaAcquires+" cancels="+events.filter(x=>x==="cancel").length+
    " label="+document.getElementById("micLabel").textContent;
  result.dataset.pass=String(begins===4&&audios===3&&completeHeaders&&!earlyBegin&&!earlyAudio&&
    mediaAcquires===1&&events.filter(x=>x==="cancel").length===0&&
    document.getElementById("micLabel").textContent==="Start conversation");
  result.textContent=text;
},6000);
</script>
"""

def main():
    component = COMPONENT.read_text(encoding="utf-8")
    browser = EDGE if EDGE.exists() else shutil.which("msedge")
    if not browser:
        raise SystemExit("Microsoft Edge is unavailable")
    for mode in ("typed_interrupt", "stalled_turn", "silent_restart", "recoverable_error"):
        mocks = MOCKS
        if mode == "silent_restart":
            mocks = mocks.replace("if(begins<=3)", "if(false)")
            mocks = mocks.replace("speech_confirm_ms:100,silence_ms:300,",
                                  "speech_confirm_ms:100,silence_ms:300,no_speech_timeout_seconds:0.5,")
            mocks = mocks.replace("begins===4&&audios===3&&completeHeaders",
                                  "begins===1&&audios===0&&recorderStarts>=3")
        if mode == "recoverable_error":
            original = "setTimeout(()=>finishTurn(id,audios<3),70);"
            retry = """if(audios===1) setTimeout(()=>window.postMessage({
      type:"streamlit:render",args:{
        session_id:"smoke-session",accepted_request_id:"",
        voice_error_request_id:id,voice_retry:true,
        voice_error:"Only background noise was detected."
      }
    },"*"),70);
    else setTimeout(()=>finishTurn(id,false),70);"""
            assert original in mocks
            mocks = mocks.replace(original, retry)
        if mode == "stalled_turn":
            start = mocks.index("setTimeout(()=>{\n  const at=Date.now();")
            end = mocks.index("setTimeout(()=>{\n  const text=", start)
            stale = """setTimeout(()=>{
  const key="texmin_voice_turn:smoke-session";
  const turn=JSON.parse(localStorage.getItem(key));
  turn.state="transcribing";
  turn.at=Date.now()-2000;
  localStorage.setItem(key,JSON.stringify(turn));
},5100);
"""
            mocks = mocks[:start] + stale + mocks[end:]
            mocks = mocks.replace("speech_confirm_ms:100,silence_ms:300,",
                                  "speech_confirm_ms:100,silence_ms:300,voice_turn_watchdog_seconds:1,")
            mocks = mocks.replace('events.filter(x=>x==="cancel").length===0',
                                  'events.filter(x=>x==="cancel").length===1')
        html = component.replace("<script>", mocks + "<script>", 1)
        with tempfile.TemporaryDirectory(prefix="texmin-voice-browser-") as folder:
            page = Path(folder) / "smoke.html"
            page.write_text(html, encoding="utf-8")
            command = [str(browser), "--headless=new", "--no-first-run",
                       "--no-default-browser-check", "--disable-extensions",
                       "--disable-gpu-sandbox", "--enable-unsafe-swiftshader",
                       "--disable-background-networking",
                       "--user-data-dir=" + str(Path(folder) / "profile"),
                       "--virtual-time-budget=6500", "--dump-dom", page.as_uri()]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        marker = result.stdout.find('<output id="result"')
        print(mode, result.stdout[marker:result.stdout.find("</output>", marker)+9])
        if 'data-pass="true"' not in result.stdout:
            print(result.stderr[-1000:])
            raise SystemExit("Voice browser smoke failed: " + mode)
    print("PASS")

if __name__ == "__main__":
    main()

