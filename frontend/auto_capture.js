/* Snap&Style automatic camera guide — browser-only, no backend. */
(function (window) {
  'use strict';

  var ASSETS = 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/';
  var SETTINGS = {
    interval: 115, stableMs: 1350,
    minWidth: .20, maxWidth: .70, minHeight: .32, maxHeight: .86,
    centerX: .12, centerY: .13, maxRoll: 13, maxYaw: .26,
    minPitch: .22, maxPitch: .72, minLight: 40, maxLight: 232,
    maxMovement: .027, maxScaleChange: .035
  };
  var detector = null, handsDetector = null, video = null, canvas = null, ctx = null;
  var hands=[], handsReady=false, handLastAt=0, frameCounter=0;
  var active = false, busy = false, captured = false, frameId = 0;
  var lastTime = 0, lastValidTime = 0, startedStable = 0;
  var previous = null, handlers = {}, token = 0;

  var sampling = document.createElement('canvas');
  sampling.width = 96; sampling.height = 72;
  var sampleCtx = sampling.getContext('2d', { willReadFrequently:true });
  function clamp(v,a,b) { return Math.max(a,Math.min(b,v)); }
  function emit(state,message,detail,progress) {
    if (handlers.onStatus) handlers.onStatus({state:state,message:message,detail:detail || '',progress:clamp(progress || 0,0,100)});
  }
  function reset() { startedStable = 0; previous = null; }
  function pointDistance(a,b) { return Math.hypot(a.x-b.x,a.y-b.y); }
  function bbox(landmarks) {
    // Face oval anchors avoid iris / forehead artefacts when checking fit.
    var indices = [10,152,234,454,127,356,132,361,136,365,172,397,58,288,93,323];
    var xs=indices.map(function(i){return landmarks[i].x;});
    var ys=indices.map(function(i){return landmarks[i].y;});
    var left=Math.min.apply(null,xs), right=Math.max.apply(null,xs);
    var top=Math.min.apply(null,ys), bottom=Math.max.apply(null,ys);
    return {left:left,right:right,top:top,bottom:bottom,
      w:right-left,h:bottom-top,cx:(left+right)/2,cy:(top+bottom)/2};
  }
  function faceBrightness(b) {
    if (!video || !sampleCtx) return 128;
    sampleCtx.drawImage(video,0,0,96,72);
    var x0=clamp(Math.floor((b.left+.15*b.w)*96),0,95);
    var x1=clamp(Math.ceil((b.right-.15*b.w)*96),x0+1,96);
    var y0=clamp(Math.floor((b.top+.17*b.h)*72),0,71);
    var y1=clamp(Math.ceil((b.bottom-.17*b.h)*72),y0+1,72);
    var pixels=sampleCtx.getImageData(x0,y0,x1-x0,y1-y0).data;
    var total=0, count=0;
    for(var i=0;i<pixels.length;i+=16) {
      total+=pixels[i]*.2126+pixels[i+1]*.7152+pixels[i+2]*.0722;count++;
    }
    return count ? total/count:128;
  }
  function backlightInfo(face) {
    if (!sampleCtx) return {background:128,excess:0};
    // The last frame was drawn by faceBrightness(). Sample background corners.
    var pixels=sampleCtx.getImageData(0,0,96,72).data;
    var sum=0, bright=0, count=0;
    for(var y=0;y<72;y+=3) for(var x=0;x<96;x+=3){
      var nx=x/96,ny=y/72;
      if(nx>face.left-.05 && nx<face.right+.05 && ny>face.top-.05 && ny<face.bottom+.05)continue;
      var pos=(y*96+x)*4;
      var l=.2126*pixels[pos]+.7152*pixels[pos+1]+.0722*pixels[pos+2];
      sum+=l;if(l>215)bright++;count++;
    }
    return {background:count?sum/count:128,excess:count?bright/count:0};
  }
  function handOverlapsFace(face) {
    if (!handsDetector) return false; // FaceMesh can run without optional Hands support.
    if (!handsReady || performance.now()-handLastAt>2200) return null;
    // MediaPipe Hands: reject any clear hand landmark inside an expanded oval.
    var cx=face.cx,cy=face.cy;
    var rx=face.w*.66,ry=face.h*.64;
    for (var i=0;i<hands.length;i++){
      var inside=0;
      for(var j=0;j<hands[i].length;j++){
        var p=hands[i][j];
        var dx=(p.x-cx)/Math.max(rx,.01),dy=(p.y-cy)/Math.max(ry,.01);
        if(dx*dx+dy*dy<1)inside++;
      }
      if(inside>=2)return true;
    }
    return false;
  }
  function pose(landmarks) {
    var a=landmarks[33],b=landmarks[263],n=landmarks[1],chin=landmarks[152];
    var vw=(video&&video.videoWidth)||1,vh=(video&&video.videoHeight)||1;
    var ax=a.x*vw,ay=a.y*vh,bx=b.x*vw,by=b.y*vh;
    var nx=n.x*vw,ny=n.y*vh,cx=chin.x*vw,cy=chin.y*vh;
    var midX=(ax+bx)/2, midY=(ay+by)/2;
    var eyeDistance=Math.max(Math.hypot(bx-ax,by-ay),1);
    var rawRoll=Math.abs(Math.atan2(by-ay,bx-ax)*180/Math.PI);
    return {roll:Math.min(rawRoll,Math.abs(180-rawRoll)),
      yaw:Math.abs(nx-midX)/eyeDistance,
      pitch:(ny-midY)/Math.max(cy-midY,1)};
  }
  function downwardGaze(points) {
    if(points.length<478) return false;
    var eyes=[[468,159,145],[473,386,374]], ratios=[];
    for (var i=0;i<eyes.length;i++) {
      var iris=points[eyes[i][0]],upper=points[eyes[i][1]],lower=points[eyes[i][2]];
      if(!iris||!upper||!lower) return false;
      var opening=lower.y-upper.y;
      if(opening<.009) return false;
      ratios.push((iris.y-upper.y)/opening);
    }
    return ratios[0]>.78 && ratios[1]>.78;
  }
  function check(points, time) {
    var b=bbox(points), p=pose(points), light=faceBrightness(b);
    var ambient=backlightInfo(b), handOverlap=handOverlapsFace(b);
    var mirrored=handlers.mirrorCapture!==false;
    var displayX=mirrored?1-b.cx:b.cx;
    var result={state:'adjust',valid:false,progress:0,box:b};
    function reject(message,detail) {
      reset(); result.message=message;result.detail=detail;return result;
    }
    if (b.left<.015||b.right>.985||b.top<.015||b.bottom>.985)
      return reject('Show your full face','Make sure your forehead, chin and cheeks are visible.');
    if(b.w<SETTINGS.minWidth||b.h<SETTINGS.minHeight)
      return reject('Come a little closer','Move nearer without cutting off your face.');
    if(b.w>SETTINGS.maxWidth||b.h>SETTINGS.maxHeight)
      return reject('Move a little farther back','Leave some space around your whole face.');
    if(Math.abs(displayX-.5)>SETTINGS.centerX)
      return reject(displayX>.5?'Move a little left':'Move a little right','Bring the face outline toward the dotted oval.');
    if(Math.abs(b.cy-.51)>SETTINGS.centerY)
      return reject(b.cy>.51?'Raise your face a little':'Lower your face a little','Bring your face toward the middle.');
    if(handOverlap===null)
      return reject('Checking for obstructions','Keep your face clear until the camera is ready.');
    if(handOverlap)
      return reject('Remove obstructions off the face','Keep your whole forehead, nose, mouth and chin visible.');
    if(p.roll>SETTINGS.maxRoll)
      return reject('Straighten your head','Keep your eyes level.');
    if(p.yaw>SETTINGS.maxYaw)
      return reject('Face the camera directly','Turn toward the camera lens.');
    if(p.pitch<SETTINGS.minPitch)
      return reject('Lower your chin slightly','Keep the camera near eye level.');
    if(p.pitch>SETTINGS.maxPitch)
      return reject('Raise your chin slightly','Keep the camera near eye level.');
    if(light<97 && ambient.background>light+56 && (ambient.background>152 || ambient.excess>.15))
      return reject('Reduce bright background light','Face a soft light source or move away from bright windows behind you.');
    if(light<SETTINGS.minLight)
      return reject('Find brighter, even light','Face a soft light source.');
    if(light>SETTINGS.maxLight)
      return reject('Reduce strong light','Avoid harsh glare on your face.');
    if(downwardGaze(points))
      return reject('Look at the camera lens','Keep your eyes raised, not on the text.');

    if(previous) {
      var movement=Math.hypot(b.cx-previous.cx,b.cy-previous.cy);
      var changed=Math.abs(b.w-previous.w);
      if(movement>SETTINGS.maxMovement || changed>SETTINGS.maxScaleChange) {
        reset(); previous={cx:b.cx,cy:b.cy,w:b.w};
        result.state='ready';result.message='Hold still';result.detail='Position looks good. Keep your eyes on the lens.';return result;
      }
    }
    previous={cx:b.cx,cy:b.cy,w:b.w};
    if(!startedStable) startedStable=time;
    var elapsed=Math.max(0,time-startedStable);
    result.valid=true;result.state='ready';result.progress=clamp(elapsed/SETTINGS.stableMs*100,0,100);
    result.stable=elapsed>=SETTINGS.stableMs;
    result.message=result.stable?'Capturing now…':'Look at the lens · hold still';
    result.detail='Keep your face level and your eyes forward.';
    result.metrics={box:b,pose:p,brightness:light,backgroundBrightness:ambient.background};
    return result;
  }

  function overlaySize() {
    if(!canvas||!ctx) return null;
    var rect=canvas.getBoundingClientRect();
    var scale=Math.min(window.devicePixelRatio||1,2);
    var w=Math.max(1,Math.round(rect.width*scale));
    var h=Math.max(1,Math.round(rect.height*scale));
    if(canvas.width!==w||canvas.height!==h) {canvas.width=w;canvas.height=h;}
    ctx.setTransform(scale,0,0,scale,0,0);
    return {w:rect.width,h:rect.height};
  }
  function render(box,state) {
    if(!ctx) return;
    var bounds=overlaySize();if(!bounds)return;
    var w=bounds.w,h=bounds.h;
    ctx.clearRect(0,0,w,h);
    // Dotted target is the same coordinate space as face-position checks.
    ctx.save();ctx.lineWidth=2;ctx.setLineDash([8,7]);ctx.strokeStyle='rgba(255,255,255,.65)';
    ctx.beginPath();ctx.ellipse(w*.5,h*.51,w*.29,h*.39,0,0,Math.PI*2);ctx.stroke();ctx.restore();
    if(!box)return;
    var mirror=handlers.mirrorCapture!==false;
    var xc=(mirror?1-box.cx:box.cx)*w;
    var yc=box.cy*h;
    var fw=box.w*w,fh=box.h*h;
    ctx.save();ctx.lineWidth=3;ctx.strokeStyle=state==='ready'?'#7fe2a8':'#ffca81';
    ctx.shadowColor=ctx.strokeStyle;ctx.shadowBlur=8;
    ctx.beginPath();ctx.ellipse(xc,yc,Math.max(10,fw*.55),Math.max(10,fh*.54),0,0,Math.PI*2);ctx.stroke();
    ctx.restore();
    // Point marking eye-level: draws on the actual video, not the entire modal.
    ctx.save();ctx.fillStyle='#fff';ctx.globalAlpha=.8;
    ctx.beginPath();ctx.arc(w*.5,h*.09,3.5,0,Math.PI*2);ctx.fill();ctx.restore();
  }
  function softenBackground(ctx,w,h,box,mirrored){
    if(!box)return;
    // A visual highlight-reduction effect only. This cannot correct camera exposure.
    // Preserve face pixels for LAB skin / hair measurements in Python.
    var pixels=ctx.getImageData(0,0,w,h),data=pixels.data;
    var cx=(mirrored?1-box.cx:box.cx)*w,cy=box.cy*h;
    var rx=Math.max(1,box.w*w*.78),ry=Math.max(1,box.h*h*.78);
    for(var y=0;y<h;y++)for(var x=0;x<w;x++){
      var nx=(x-cx)/rx,ny=(y-cy)/ry,rad=Math.hypot(nx,ny);
      if(rad<1)continue;
      var alpha=Math.min(1,(rad-1)/.35);
      var i=(y*w+x)*4,r=data[i],g=data[i+1],b=data[i+2];
      var lum=.2126*r+.7152*g+.0722*b;
      if(lum<154)continue;
      var strength=Math.min(.42,Math.max(0,(lum-154)/100*.42))*alpha;
      var factor=1-strength;
      data[i]=r*factor;data[i+1]=g*factor;data[i+2]=b*factor;
    }
    ctx.putImageData(pixels,0,0);
  }
  function grab(metrics) {
    if(captured||!active||!video||!video.videoWidth)return;
    captured=true;
    var output=document.createElement('canvas');
    output.width=video.videoWidth;output.height=video.videoHeight;
    var outCtx=output.getContext('2d');
    if(handlers.mirrorCapture!==false){outCtx.translate(output.width,0);outCtx.scale(-1,1);}
    outCtx.drawImage(video,0,0,output.width,output.height);
    // Keep analysis pixels unchanged. Any visual background treatment belongs
    // on a separate preview layer, never in the JPEG sent to the CV API.
    var dataUrl=output.toDataURL('image/jpeg',.92);
    emit('ready','Captured automatically','Your photo passed the alignment checks.',100);
    if(handlers.onCapture) handlers.onCapture({dataUrl:dataUrl,width:output.width,height:output.height,metrics:metrics});
  }
  function onResults(results) {
    if(!active||captured)return;
    var faces=results&&results.multiFaceLandmarks||[];
    if(faces.length!==1){reset();render(null,'searching');
      if(!faces.length)emit('searching','Find your face','Keep one full face in the camera view.',0);
      else emit('adjust','Only one person in frame','Move other people out of view.',0);
      return;
    }
    var result=check(faces[0],performance.now());
    render(result.box,result.state);
    if(!handsDetector && result.valid){
      result.detail='Face alignment is being checked. Obstruction detection is limited.';
    }
    emit(result.state,result.message,result.detail,result.progress);
    if(result.valid&&result.stable)grab(result.metrics);
  }
  function initialize() {
    if (detector) return detector;
    if (!window.FaceMesh) return null;

    try {
      detector = new window.FaceMesh({locateFile:function(f){return ASSETS+f;}});
      detector.setOptions({maxNumFaces:2,refineLandmarks:true,minDetectionConfidence:.6,minTrackingConfidence:.6});
      detector.onResults(onResults);
    } catch (err) {
      console.error('FaceMesh initialization failed:', err);
      detector = null;
      return null;
    }

    // Hands adds obstruction detection, but is NOT required for face guidance.
    // Some browsers block one CDN script while loading the other successfully.
    if (window.Hands) {
      try {
        handsDetector = new window.Hands({locateFile:function(f){
          return 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/'+f;
        }});
        handsDetector.setOptions({maxNumHands:2,modelComplexity:0,minDetectionConfidence:.6,minTrackingConfidence:.6});
        handsDetector.onResults(function(result){
          hands=result&&result.multiHandLandmarks||[];
          handsReady=true;
          handLastAt=performance.now();
        });
      } catch (err) {
        console.warn('Hands detector unavailable; continuing with face alignment:', err);
        handsDetector = null;
      }
    } else {
      console.warn('MediaPipe Hands did not load; continuing with FaceMesh guidance.');
    }
    return detector;
  }
  async function tick(t) {
    if(!active)return;
    frameId=window.requestAnimationFrame(tick);
    if(busy || !video || video.readyState<2 || t-lastTime<SETTINGS.interval)return;
    lastTime=t;busy=true;var myToken=token;
    try {
      await detector.send({image:video});
      // Keep the face guide active even if the optional Hands model fails.
      if (++frameCounter%3===0 && handsDetector && active && token===myToken) {
        try {
          await handsDetector.send({image:video});
        } catch (handError) {
          console.warn('Hands detection failed; continuing with FaceMesh:', handError);
          handsDetector = null;
          hands = [];
          handsReady = false;
        }
      }
    }
    catch(e){if(active&&token===myToken){
      stop();if(handlers.onError) handlers.onError('Face/obstruction detection failed: '+(e&&e.message||'check internet/CDN permissions'));}}
    finally{busy=false;}
  }
  function stop() {
    active=false;captured=false;reset();token++;
    hands=[];handsReady=false;handLastAt=0;frameCounter=0;
    if(frameId)window.cancelAnimationFrame(frameId);
    frameId=0;video=null;
    if(ctx&&canvas)ctx.clearRect(0,0,canvas.width,canvas.height);
    canvas=null;ctx=null;
  }
  function start(videoEl,options) {
    stop(); handlers=options||{};video=videoEl;
    canvas=document.getElementById('cameraOverlay');ctx=canvas&&canvas.getContext('2d');
    if(!video||!canvas||!ctx){if(handlers.onError)handlers.onError('Camera video or guidance canvas missing.');return;}
    if(!initialize()){if(handlers.onError)handlers.onError('MediaPipe FaceMesh did not load. Check the network tab or try a different connection.');return;}
    active=true;captured=false;lastTime=0;lastValidTime=0;startedStable=0;
    emit('searching','Looking for one face',handsDetector ? 'Center your face and look at the camera lens.' : 'Face guidance active. Hand obstruction detection is unavailable; keep your face clear.',0);
    render(null,'searching');frameId=window.requestAnimationFrame(tick);
  }
  window.SnapStyleAutoCapture={
    isSupported:function(){return !!(window.FaceMesh&&navigator.mediaDevices&&navigator.mediaDevices.getUserMedia);},
    start:start,stop:stop
  };
})(window);
