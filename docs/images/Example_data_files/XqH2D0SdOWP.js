if (self.CavalryLogger) { CavalryLogger.start_js(["WMFuJ"]); }

__d("CometButtonStyles_DEPRECATED.react",["React","stylex"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g={disabled:{backgroundImage:"mf7ej076",cursor:"t5a262vz",":hover":{backgroundImage:"ljzjr9fn",cursor:"tkdm7zml"},":focus":{boxShadow:"mi62g4hq"}},expanded:{display:"pq6dq46d",justifyContent:"taijpn5t",minHeight:"sn7ne77z",minWidth:"oqhjfihn"},large:{fontSize:"a5q79mjw",lineHeight:"g1cxx5fr"},medium:{fontSize:"jq4qci2q",lineHeight:"m5l1wtfr"},primary:{backgroundColor:"s1i5eluu",color:"bwm1u5wc",":active":{backgroundColor:"id6903cd"}},primaryDeemphasized:{backgroundColor:"oo1teu6h",color:"knomaqxo",":active":{backgroundColor:"rudrce6k",color:"qzg4r8h7"}},primaryDeemphasizedDisabled:{backgroundColor:"c98fg2ug",color:"pipptul6"},primaryDisabled:{backgroundColor:"c98fg2ug",color:"erlsw9ld"},root:{alignItems:"bp9cbjyn",borderTopStartRadius:"beltcj47",borderTopEndRadius:"p86d2i9g",borderBottomEndRadius:"aot14ch1",borderBottomStartRadius:"kzx2olss",borderTopWidth:"rt8b4zig",borderEndWidth:"n8ej3o3l",borderBottomWidth:"agehan2d",borderStartWidth:"sk4xxmp2",boxSizing:"rq0escxv",cursor:"nhd2j8a9",display:"pq6dq46d",fontWeight:"lrazzd5p",outline:"lzcic4wl",paddingTop:"cxgpxx05",paddingEnd:"d1544ag0",paddingBottom:"sj5x9vvc",paddingStart:"tw6a2znq",position:"l9j0dhe7",textAlign:"oqcyycmt",textDecoration:"esuyzwwr",textShadow:"gigivrx4",verticalAlign:"sf5mxxl7",whiteSpace:"g0qnabr5",":hover":{backgroundImage:"ehryuci6",textDecoration:"p8dawk7l"},":focus":{boxShadow:"lrwzeq9o",outline:"iqfcb0g7"},":active":{transform:"lsqurvkf"}},secondary:{backgroundColor:"tdjehn4e",color:"oo9gr5id",":active":{backgroundColor:"kca3o15f"}},secondaryDeemphasized:{backgroundColor:"g5ia77u1",color:"knomaqxo",":active":{backgroundColor:"cq6j33a1",color:"qzg4r8h7"}},secondaryDeemphasizedDisabled:{backgroundColor:"g5ia77u1",color:"erlsw9ld"},secondaryDisabled:{backgroundColor:"c98fg2ug",color:"erlsw9ld"},shadow:{boxShadow:"rdkkywzo"},white:{backgroundColor:"q2y6ezfg",color:"oo9gr5id",":active":{backgroundColor:"cq6j33a1"}},whiteDeemphasized:{backgroundColor:"g5ia77u1",color:"knomaqxo",":active":{backgroundColor:"cq6j33a1",color:"qzg4r8h7"}},whiteDeemphasizedDisabled:{backgroundColor:"g5ia77u1",color:"erlsw9ld"},whiteDisabled:{backgroundColor:"c98fg2ug",color:"bwm1u5wc"}};function a(a){__p&&__p();var b=a.children,c=a.deemphasized;c=c===void 0?!1:c;var d=a.disabled;d=d===void 0?!1:d;var e=a.expanded;e=e===void 0?!1:e;var f=a.shadow;f=f===void 0?!1:f;var h=a.size;h=h===void 0?"medium":h;a=a.use;a=a===void 0?"primary":a;return b([g.root,d&&g.disabled,h==="large"&&g.large,h==="medium"&&g.medium,e===!0&&g.expanded,f===!0&&g.shadow,a==="primary"&&g.primary,a==="primary"&&d===!0&&g.primaryDisabled,a==="primary"&&c===!0&&g.primaryDeemphasized,a==="primary"&&c===!0&&d===!0&&g.primaryDeemphasizedDisabled,a==="secondary"&&g.secondary,a==="secondary"&&d===!0&&g.secondaryDisabled,a==="secondary"&&c===!0&&g.secondaryDeemphasized,a==="secondary"&&c===!0&&d===!0&&g.secondaryDeemphasizedDisabled,a==="white"&&g.white,a==="white"&&d===!0&&g.whiteDisabled,a==="white"&&c===!0&&g.whiteDeemphasized,a==="white"&&c===!0&&d===!0&&g.whiteDeemphasizedDisabled])}e.exports=a}),null);
__d("CometButton_DEPRECATED.react",["BaseButtonOrLink_DEPRECATED.react","CometButtonStyles_DEPRECATED.react","React"],(function(a,b,c,d,e,f){"use strict";__p&&__p();function a(a,c){__p&&__p();var d=a.children,e=a.deemphasized;e=e===void 0?!1:e;var f=a.expanded;f=f===void 0?!1:f;a.label;var g=a.shadow;g=g===void 0?!1:g;var h=a.size;h=h===void 0?"medium":h;var i=a.use;i=i===void 0?"primary":i;var j=babelHelpers.objectWithoutPropertiesLoose(a,["children","deemphasized","expanded","label","shadow","size","use"]);a=a.disabled;a=a===void 0?!1:a;return b("React").jsx(b("CometButtonStyles_DEPRECATED.react"),{deemphasized:e,disabled:a,expanded:f,shadow:g,size:h,use:i,children:function(a){return b("React").jsx(b("BaseButtonOrLink_DEPRECATED.react"),babelHelpers["extends"]({},j,{ref:c,xstyle:a,children:d}))}})}e.exports=b("React").forwardRef(a)}),null);
__d("CometThrottle",["clearTimeout","setTimeout"],(function(a,b,c,d,e,f){"use strict";__p&&__p();function a(a,c,d){__p&&__p();d=d===void 0?{}:d;var e=d.leading,f=d.trailing,g,h,i,j=null,k=0,l=function(){k=e===!1?0:new Date(),j=null,i=a.apply(g,h)};d=function(){j!=null&&(b("clearTimeout")(j),j=null)};function m(){var d=new Date();!k&&e===!1&&(k=d);var m=c-(d-k);g=this;h=arguments;m<=0?(b("clearTimeout")(j),j=null,k=d,i=a.apply(g,h)):!j&&f!==!1&&(j=b("setTimeout")(l,m));return i}m.cancel=d;return m}e.exports=a}),null);
__d("useCometWindowSize",["CometThrottle","React"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g=b("React").useEffect,h=b("React").useState;function i(){return{innerHeight:window.innerHeight,innerWidth:window.innerWidth,outerHeight:window.outerHeight,outerWidth:window.outerWidth}}function a(){var a=h(i()),c=a[0],d=a[1];g(function(){var a=b("CometThrottle")(function(){return d(i())},500);window.addEventListener("resize",a);return function(){window.removeEventListener("resize",a)}},[]);return c}e.exports=a}),null);
__d("useFeedClickEventHandler",["React","useStoryClickEventLogger"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g=b("React").useCallback;function a(a,c){var d=b("useStoryClickEventLogger")();return g(function(b){a&&a(b);var e=b.type;if(e==="click"||e==="contextmenu"||e==="mousedown"&&typeof b.button==="number"&&(b.button===1||b.button===2)||e==="keydown"&&(b.key==="Enter"||b.key===" ")){e=typeof b.button==="number"?b.button:0;d(b.timeStamp,e,c)}},[a,d,c])}e.exports=a}),null);
__d("DOMRectIsEqual",["DOMRectReadOnly"],(function(a,b,c,d,e,f){"use strict";function a(a,b){if(!a&&!b)return!0;else if(!a||!b)return!1;return a.x===b.x&&a.y===b.y&&a.width===b.width&&a.height===b.height}e.exports=a}),null);
__d("VideoPlayerSphericalFallbackCover.react",["fbt","React","SphericalMediaGyroOverlay.react","TetraText.react","stylex","unrecoverableViolation","useCometRouterDispatcher","useFeedClickEventHandler"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();var h=b("React").useState;function a(a){__p&&__p();var c=a.videoTahoeUrl,d=b("useCometRouterDispatcher")();a=h(!1);var e=a[0],f=a[1];if(d==null)throw b("unrecoverableViolation")("Missing CometRouterDispatcher","comet_video_player");a=b("useFeedClickEventHandler")(function(){d.go(c,{})});return b("React").jsxs("div",{className:"bkfpd7mw cbu4d94t j83agx80 nhd2j8a9 bp9cbjyn kr520xx4 j9ispegn pmk7jnqg n7fi1qx3 rq0escxv i09qtzwb",onClick:a,onMouseEnter:function(){return f(!0)},onMouseLeave:function(){return f(!1)},role:"link",tabIndex:0,children:[b("React").jsx(b("SphericalMediaGyroOverlay.react"),{isActive:!0}),b("React").jsx("div",{className:"l9j0dhe7 i7orit0i",children:e?b("React").jsx(b("TetraText.react"),{color:"white",type:"bodyLink3",children:g._("Click to expand")}):null})]})}e.exports=a}),null);
__d("VideoPlayerSpinner.react",["CometLoadingAnimation.react","React","VideoPlayerHooks","stylex","useDebouncedValue"],(function(a,b,c,d,e,f){"use strict";var g,h=b("VideoPlayerHooks").useStalling,i=36;function a(){var a=h();a=a;var c=b("useDebouncedValue")(a,a?200:500);return b("React").jsx("div",{className:(g||(g=b("stylex"))).dedupe({"height-1":"tv7at329","opacity-1":"pedkr2u6","position-1":"pmk7jnqg","start-1":"kfkz5moi","top-1":"rk01pc8j","transform-0.1":"py2didcb","transition-delay-1":"chkx7lpg","transition-duration-1":"kmdw4o4n","transition-property-1":"art1omkt","transition-timing-function-1":"e4zzj2sf","width-1":"thwo4zme"},a?null:{"opacity-1":"b5wmifdl","transition-delay-1":"hwaazqwg","transition-duration-1":"kmdw4o4n","transition-property-1":"l23jz15m","transition-timing-function-1":"e4zzj2sf","visibility-1":"kr9hpln1"}),children:b("React").jsx(b("CometLoadingAnimation.react"),{animationPaused:!c,size:i})})}e.exports=a}),null);
__d("convertToViewabilityPercentage",[],(function(a,b,c,d,e,f){"use strict";__p&&__p();function a(a){__p&&__p();if(a>=.99)return 100;else if(a>=.75)return 75;else if(a>=.5)return 50;else if(a>=.25)return 25;else if(a>=0)return 0;else return-2}e.exports=a}),null);
__d("CometVideoHomeFeedUnitPositionContext",["React"],(function(a,b,c,d,e,f){"use strict";a=b("React").createContext(0);e.exports=a}),null);
__d("CometVideoHomeShowSurfacesLoggingContext",["React"],(function(a,b,c,d,e,f){"use strict";a=b("React").createContext({entrypoint:"",pageID:null,sectionTypeName:"",surface:""});e.exports=a}),null);
__d("VideoHomeLoggingReactionVideoChannelTypeContext",["React"],(function(a,b,c,d,e,f){"use strict";a=b("React").createContext(null);e.exports=a}),null);
__d("VideoPlayerLoggingSuboriginContext",["React"],(function(a,b,c,d,e,f){"use strict";a=b("React").createContext(null);e.exports=a}),null);
__d("isEmptyObject",[],(function(a,b,c,d,e,f){"use strict";function a(a){for(var b in a)return!1;return!0}e.exports=a}),null);
__d("useDebounced",["React","debounce"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g=b("React").useEffect,h=b("React").useMemo,i=b("React").useRef;function a(a,c){__p&&__p();c===void 0&&(c=100);var d=i(a);g(function(){d.current=a},[a]);var e=h(function(){return b("debounce")(function(){return d.current.apply(d,arguments)},c)},[c]);g(function(){return e.reset},[e]);return e}e.exports=a}),null);
__d("requireDeferredForDisplay",["requireDeferred"],(function(a,b,c,d,e,f){"use strict";function a(a){return b("requireDeferred").call(null,a)}e.exports=a}),null);
__d("WebSessionExtender",["WebSession","clearInterval","setInterval"],(function(a,b,c,d,e,f){"use strict";var g=3e4,h=new Set(),i=null;a={subscribe:function(a){h.add(a),i==null&&(b("WebSession").extend(Date.now()+g+2e3),i=b("setInterval")(function(){b("WebSession").extend(Date.now()+g+2e3)},g))},unsubscribe:function(a){h["delete"](a);a=h.size;a===0&&i!=null&&(b("clearInterval")(i),i=null)}};e.exports=a}),null);
__d("URLSearchParams",[],(function(a,b,c,d,e,f){__p&&__p();var g=/\+/g,h=/[!\'()*]/g,i=/%20/g;function j(a){return encodeURIComponent(a).replace(i,"+").replace(h,function(a){return"%"+a.charCodeAt(0).toString(16)})}function k(a){return decodeURIComponent((a=a)!=null?a:"").replace(g," ")}var l=typeof Symbol==="function"?Symbol.iterator:"@@iterator";a=function(){"use strict";__p&&__p();function a(a){a===void 0&&(a="");a=a;a[0]==="?"&&(a=a.substr(1));this.$1=a.length?a.split("&").map(function(a){a=a.split("=");var b=a[0];a=a[1];return[k(b),k(a)]}):[]}var b=a.prototype;b.append=function(a,b){this.$1.push([a,String(b)])};b["delete"]=function(a){for(var b=0;b<this.$1.length;b++)this.$1[b][0]===a&&(this.$1.splice(b,1),b--)};b.entries=function(){return this.$1[typeof Symbol==="function"?Symbol.iterator:"@@iterator"]()};b.get=function(a){for(var b=0,c=this.$1.length;b<c;b++)if(this.$1[b][0]===a)return this.$1[b][1];return null};b.getAll=function(a){var b=[];for(var c=0,d=this.$1.length;c<d;c++)this.$1[c][0]===a&&b.push(this.$1[c][1]);return b};b.has=function(a){for(var b=0,c=this.$1.length;b<c;b++)if(this.$1[b][0]===a)return!0;return!1};b.keys=function(){var a=this.$1.map(function(a){var b=a[0];a[1];return b});return a[typeof Symbol==="function"?Symbol.iterator:"@@iterator"]()};b.set=function(a,b){var c=!1;for(var d=0;d<this.$1.length;d++)this.$1[d][0]===a&&(c?(this.$1.splice(d,1),d--):(this.$1[d][1]=String(b),c=!0));c||this.$1.push([a,String(b)])};b.toString=function(){return this.$1.map(function(a){var b=a[0];a=a[1];return j(b)+"="+j(a)}).join("&")};b.values=function(){var a=this.$1.map(function(a){a[0];a=a[1];return a});return a[typeof Symbol==="function"?Symbol.iterator:"@@iterator"]()};b[l]=function(){return this.entries()};return a}();e.exports=a}),null);
__d("XLiveScheduleSubscriptionController",["XController"],(function(a,b,c,d,e,f){e.exports=b("XController").create("/live_video_schedule/subscription/",{video_broadcast_schedule_id:{type:"FBID"},video_id:{type:"FBID"},subscribe:{type:"Bool",defaultValue:!1},origin:{type:"String"}})}),null);