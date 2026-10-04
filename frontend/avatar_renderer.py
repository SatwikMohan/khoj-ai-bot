"""Package the original GLTF avatar and Three.js modules into an offline iframe."""

import base64
import json
import mimetypes
from functools import lru_cache
from pathlib import Path

import config


ASSETS = Path(__file__).resolve().parent / "assets"


def _uri(path: Path, mime: str | None = None) -> str:
    kind = mime or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{kind};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


@lru_cache(maxsize=2)
def model_uri(configured_path: str) -> str:
    path = Path(configured_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if path.suffix.lower() == ".glb":
        return _uri(path, "model/gltf-binary")
    if path.suffix.lower() != ".gltf":
        raise ValueError("AVATAR_MODEL_FILE must point to a GLTF or GLB model")
    model = json.loads(path.read_text(encoding="utf-8"))
    for resource in [*model.get("buffers", []), *model.get("images", [])]:
        source = resource.get("uri", "")
        if source and not source.startswith("data:"):
            resource["uri"] = _uri(path.parent / source)
    return "data:model/gltf+json;base64," + base64.b64encode(
        json.dumps(model, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


@lru_cache(maxsize=1)
def import_map() -> str:
    def javascript_uri(source: str) -> str:
        return "data:text/javascript;base64," + base64.b64encode(source.encode()).decode("ascii")

    loader = (ASSETS / "GLTFLoader.js").read_text(encoding="utf-8").replace(
        "'../utils/BufferGeometryUtils.js'", "'three/BufferGeometryUtils.js'"
    )
    return json.dumps({"imports": {
        "three": javascript_uri((ASSETS / "three.module.js").read_text(encoding="utf-8")),
        "three/GLTFLoader.js": javascript_uri(loader),
        "three/BufferGeometryUtils.js": javascript_uri(
            (ASSETS / "BufferGeometryUtils.js").read_text(encoding="utf-8")
        ),
    }})


def render_script() -> tuple[str, str]:
    """Return import map and renderer script; audio playback remains in visualization."""
    try:
        source = model_uri(config.AVATAR_MODEL_FILE)
        modules = import_map()
        error = ""
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        message = json.dumps(f"Avatar renderer unavailable: {exc}").replace("</", "<\\/")
        return "{}", f'<script>document.getElementById("modelStatus").textContent={message};</script>'
    script = r'''
<script type="module">
import * as THREE from "three";
import {GLTFLoader} from "three/GLTFLoader.js";
const host=document.getElementById("avatarScene"),notice=document.getElementById("modelStatus");
const stage=document.getElementById("visual"),audio=document.getElementById("audio");
const modelUri=__MODEL__;
try {
 if(!modelUri)throw Error(__ERROR__||"Avatar model is missing");
 const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});
 renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
 renderer.outputColorSpace=THREE.SRGBColorSpace;
 host.appendChild(renderer.domElement);
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(35,1,.01,100);
 scene.add(new THREE.HemisphereLight(0xffffff,0x315442,3.2));
 const key=new THREE.DirectionalLight(0xffffff,2.8);key.position.set(2,3,4);scene.add(key);
 const rim=new THREE.DirectionalLight(0x65ffae,1.2);rim.position.set(-3,1,-3);scene.add(rim);
 let root=null,mixer=null,jaw=null,jawRest=0,head=null,headRest=0,baseY=0;
 let center=new THREE.Vector3(),halfWidth=1,halfHeight=1,halfDepth=.3,context=null,analyser=null,samples=null,mouth=0;
 function resize(){
  const width=Math.max(1,host.clientWidth),height=Math.max(1,host.clientHeight);
  renderer.setSize(width,height,false);camera.aspect=width/height;
  const tangent=Math.tan(THREE.MathUtils.degToRad(camera.fov)/2);
  const distance=Math.max(halfHeight/tangent,halfWidth/(tangent*camera.aspect))*1.16+halfDepth;
  camera.position.set(center.x,center.y+halfHeight*.04,center.z+distance);
  camera.lookAt(center);camera.updateProjectionMatrix();
 }
 new ResizeObserver(resize).observe(host);
 new GLTFLoader().load(modelUri,gltf=>{
  root=gltf.scene;root.traverse(node=>{
   const name=(node.name||"").toLowerCase();
   if(node.isBone&&name.includes("jaw")&&!name.includes("unused")&&!jaw){jaw=node;jawRest=node.rotation.z}
   if(node.isBone&&/root[_ ]head/.test(name)&&!head){head=node;headRest=node.rotation.z}
  });
  const box=new THREE.Box3().setFromObject(root);
  if(box.isEmpty()){notice.textContent="Avatar model has no visible geometry";return}
  const size=box.getSize(new THREE.Vector3()),origin=box.getCenter(new THREE.Vector3());
  root.scale.setScalar(2.4/Math.max(size.x,size.y,size.z,.001));
  root.position.copy(origin.multiplyScalar(-root.scale.x));baseY=root.position.y;scene.add(root);
  const fitted=new THREE.Box3().setFromObject(root);
  center=fitted.getCenter(new THREE.Vector3());
  const fittedSize=fitted.getSize(new THREE.Vector3());
  halfWidth=fittedSize.x/2;halfHeight=fittedSize.y/2;halfDepth=fittedSize.z/2;
  if(gltf.animations.length){mixer=new THREE.AnimationMixer(root);mixer.clipAction(gltf.animations[0]).play()}
  resize();notice.classList.add("loaded");
 },undefined,error=>{notice.textContent="Avatar could not load: "+(error.message||error);console.error(error)});
 async function prepareAudioGraph(){
  try{
   if(!context){context=new AudioContext();analyser=context.createAnalyser();analyser.fftSize=1024;
    samples=new Uint8Array(analyser.fftSize);
    const source=context.createMediaElementSource(audio);
    source.connect(analyser);analyser.connect(context.destination)}
   if(context.state==="suspended")await context.resume();
  }catch(error){console.warn("Audio analysis unavailable; jaw stays neutral",error)}
 }
 audio.addEventListener("play",prepareAudioGraph);
 if(!audio.paused)prepareAudioGraph();
 const clock=new THREE.Clock();
 function frame(now){
  requestAnimationFrame(frame);const delta=clock.getDelta(),time=now/1000,state=stage.dataset.state;
  if(mixer)mixer.update(delta);
  if(root){
   const thinking=state==="retrieving"||state==="generating";
   root.rotation.y=.5+Math.sin(time*(state==="listening"?1.1:.65))*.045+(thinking?Math.sin(time*1.1)*.04:0);
   root.position.y=baseY+Math.sin(time*(state==="listening"?1.8:1.1))*.013;
   if(head)head.rotation.z=headRest+(thinking?Math.sin(time*1.3)*.018:0);
  }
  let energy=0;
  if(state==="speaking"&&analyser&&context?.state==="running"){
   analyser.getByteTimeDomainData(samples);let sum=0;
   for(const value of samples){const sample=(value-128)/128;sum+=sample*sample}
   energy=Math.min(1,Math.max(0,(Math.sqrt(sum/samples.length)-.025)*7));
  }
  mouth+=(energy-mouth)*(energy>mouth?.55:.35);
  if(jaw)jaw.rotation.z=jawRest-mouth*.15;
  renderer.render(scene,camera);
 }
 requestAnimationFrame(frame);
}catch(error){notice.textContent="Avatar renderer unavailable: "+error.message;console.error(error)}
</script>'''
    return modules, script.replace("__MODEL__", json.dumps(source)).replace("__ERROR__", json.dumps(error))

