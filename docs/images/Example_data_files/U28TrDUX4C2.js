if (self.CavalryLogger) { CavalryLogger.start_js(["G\/G2G"]); }

__d("XUICarouselDeferredReactElement",["invariant","DOMContainer.react","React","isNode"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();a=function(){__p&&__p();function a(a,b,c){if(!a)throw new Error("XUICarouselDeferredReactElement requires a ReactClass Factory");this.$1=null;this.$2=b;this.$3=a;this.$4=c}var c=a.prototype;c.getElement=function(a){if(!this.$1){a=babelHelpers["extends"]({},this.$2,a);var c=this.$3;this.$1=b("React").jsx(c,babelHelpers["extends"]({},a,{children:this.$5(a.childIndex)}))}return this.$1};c.$5=function(a){var b=this.$4;return b?Array.isArray(b)?b.map(this.$6):[this.$6(b,a?a:0)]:[]};c.$6=function(a,c){return b("isNode")(a)?b("React").jsx(b("DOMContainer.react"),{children:a},"child_"+c):a.getElement()};c.$7=function(c){var d=c instanceof a;d||b("isNode")(c)||g(0,3136)};return a}();e.exports=a}),null);
__d("XUICarouselWrapper",["DOM","React","ReactDOM","XUICarousel.react"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(a,c){__p&&__p();var d=a.items,e=a.links,f=a.linksRoot;a=babelHelpers.objectWithoutPropertiesLoose(a,["items","links","linksRoot"]);a.arrows&&a.arrows.length?a.arrows=a.arrows.map(function(a){return a.getElement()}):a.arrows=null;for(var g in a)a[g]==null&&delete a[g];a=b("React").jsx(b("XUICarousel.react"),babelHelpers["extends"]({},a,{children:d.map(function(a,b){return a.getElement({key:"child_"+b})})}));this.$1=b("ReactDOM").render(a,c);if(!e)return;d=f;d||(d=document.createElement("div"),b("DOM").appendContent(c,d));b("ReactDOM").render(e.getElement({carousel:this.$1}),d)}var c=a.prototype;c.setAutoplayEnabled=function(){this.$1.setAutoplayEnabled.apply(this.$1,arguments)};c.setIndex=function(){this.$1.setIndex.apply(this.$1,arguments)};c.unsubscribe=function(){return this.$1.unsubscribe.apply(this.$1,arguments)};c.subscribe=function(){return this.$1.subscribe.apply(this.$1,arguments)};c.next=function(){this.$1.next()};c.prev=function(){this.$1.prev()};c.getInstance=function(){return this.$1};c.forceLayoutUpdate=function(){this.$1.forceLayoutUpdate()};return a}();e.exports=a}),null);
__d("XGroupsRHCSuggestionXoutController",["XController"],(function(a,b,c,d,e,f){e.exports=b("XController").create("/groups/xout_suggested_group/",{id:{type:"Int",required:!0}})}),null);
__d("GroupSuggestionXoutHandler",["csx","Arbiter","AsyncRequest","CSS","DOM","Event","FRXURI","XGroupsRHCSuggestionXoutController"],(function(a,b,c,d,e,f,g){__p&&__p();var h={onclick:function(a,c,d,e,f){b("Event").listen(a,"click",function(){h.hideItem(c,d,e,f)})},hideItem:function(a,c,d,e){__p&&__p();var f=b("XGroupsRHCSuggestionXoutController").getURIBuilder().setInt("id",a).getURI();new(b("AsyncRequest"))().setURI(f).send();f=document.getElementById(c);if(f!==null){var g=f.parentElement;b("DOM").remove(f);b("DOM").remove(g);b("Arbiter").inform("x-out-ego-suggestions",{card_id:c})}if(d==="group_mall_rhc"){f=document.querySelectorAll("._1spx");if(f.length===0){g=document.getElementById("GroupsRHCSuggestionSection");g!==null&&b("DOM").remove(g)}}e&&h.openReportingFlow(a,c,d)},openReportingFlow:function(a,c,d){var e=b("FRXURI").withEntReportable({reportable_ent_token:a.toString(),story_location:"unknown",entry_point:d});new(b("AsyncRequest"))().setURI(e).send();h.hideItem(a,c,d,!1)},hide:function(a,c){b("Event").listen(c,"mouseover",function(){b("CSS").show(a)}),b("Event").listen(c,"mouseleave",function(){b("CSS").hide(a)})}};e.exports=h}),null);
__d("NewsFeedDedupeStoryTypedLogger",["Banzai","GeneratedLoggerUtils","nullthrows"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(){this.$1={}}var c=a.prototype;c.log=function(){b("GeneratedLoggerUtils").log("logger:NewsFeedDedupeStoryLoggerConfig",this.$1,b("Banzai").BASIC)};c.logVital=function(){b("GeneratedLoggerUtils").log("logger:NewsFeedDedupeStoryLoggerConfig",this.$1,b("Banzai").VITAL)};c.logImmediately=function(){b("GeneratedLoggerUtils").log("logger:NewsFeedDedupeStoryLoggerConfig",this.$1,{signal:!0})};c.clear=function(){this.$1={};return this};c.getData=function(){return babelHelpers["extends"]({},this.$1)};c.updateData=function(a){this.$1=babelHelpers["extends"]({},this.$1,a);return this};c.setIsWorkplaceUser=function(a){this.$1.is_workplace_user=a;return this};c.setPositionOfDuplicate=function(a){this.$1.position_of_duplicate=a;return this};c.setTime=function(a){this.$1.time=a;return this};c.setVC=function(a){this.$1.vc=a;return this};c.setWeight=function(a){this.$1.weight=a;return this};return a}();c={is_workplace_user:!0,position_of_duplicate:!0,time:!0,vc:!0,weight:!0};e.exports=a}),null);
__d("NewsFeedTailloadPlaceholderTypedLogger",["Banzai","GeneratedLoggerUtils","nullthrows"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(){this.$1={}}var c=a.prototype;c.log=function(){b("GeneratedLoggerUtils").log("logger:NewsFeedTailloadPlaceholderLoggerConfig",this.$1,b("Banzai").BASIC)};c.logVital=function(){b("GeneratedLoggerUtils").log("logger:NewsFeedTailloadPlaceholderLoggerConfig",this.$1,b("Banzai").VITAL)};c.logImmediately=function(){b("GeneratedLoggerUtils").log("logger:NewsFeedTailloadPlaceholderLoggerConfig",this.$1,{signal:!0})};c.clear=function(){this.$1={};return this};c.getData=function(){return babelHelpers["extends"]({},this.$1)};c.updateData=function(a){this.$1=babelHelpers["extends"]({},this.$1,a);return this};c.setDisplayDone=function(a){this.$1.display_done=a;return this};c.setTailLoadPosition=function(a){this.$1.tail_load_position=a;return this};c.setTailLoadStatus=function(a){this.$1.tail_load_status=a;return this};c.setTime=function(a){this.$1.time=a;return this};c.setVC=function(a){this.$1.vc=a;return this};c.setVisibleDuration=function(a){this.$1.visible_duration=a;return this};c.setWeight=function(a){this.$1.weight=a;return this};return a}();c={display_done:!0,tail_load_position:!0,tail_load_status:!0,time:!0,vc:!0,visible_duration:!0,weight:!0};e.exports=a}),null);
__d("EgoUnitSlideInsert",["csx","cx","Animation","CSS","DataStore","DOM","Ease","Event","Parent","TidyArbiterMixin","tidyEvent"],(function(a,b,c,d,e,f,g,h){__p&&__p();var i="sliding",j="EgoSlider/End",k=babelHelpers["extends"]({isSliding:function(a){return b("DataStore").get(a,i)},runAfterSlide:function(a,c){var d=b("tidyEvent")(k.subscribe(j,function(b,e){e===a&&(d&&d.unsubscribe(),c())}))},registerSlide:function(a,c){__p&&__p();b("Event").listen(a,"click",function(d){__p&&__p();d=b("Parent").bySelector(d.getTarget(),"._5cl_");if(!d)return;var e=b("Parent").byClass(a,"ego_unit");d=0;var f=b("Parent").byClass(e,"ego_unit_container"),g=b("DOM").scry(f,".ego_unit")[0];g===e&&(e.nextSibling&&(e.nextSibling.style.paddingTop="0px",e.nextSibling.style.borderTop="0px"));b("CSS").addClass(e,"_5cl-");b("DataStore").set(e,i,!0);new(b("Animation"))(e).to("height",0).to("padding-top",d).to("padding-bottom",0).to("margin",0).from("opacity",1).to("opacity",0).ease(b("Ease").circOut).duration(300).checkpoint(1,function(){b("DOM").appendContent(f,e),b("DOM").prependContent(e,c),b("DataStore").remove(e,i)}).to("height",12).to("opacity",1).to("margin-bottom",10).duration(0).checkpoint(2,function(){k.inform(j,e)}).go()})}},b("TidyArbiterMixin"));e.exports=k}),null);
__d("NetEgo",["csx","Animation","Arbiter","CSS","DOM","EgoUnitSlideInsert","Parent","URI","ge","gkx"],(function(a,b,c,d,e,f,g){__p&&__p();var h,i={setup:function(a){__p&&__p();var c=["liked"];b("gkx")("929262")?b("Arbiter").subscribe(["FriendRequest/sending"],function(c,d){__p&&__p();if(a==d.profile_id&&d.origin=="hovercard"||a==d.uid){c=b("ge")(document.body,".ego_unit_container");if(c){d=b("DOM").scry(c,".ego_unit");c=d.length;for(var e=0;e<c;e++){var f=d[e].getAttribute("data-ego-fbid");if(f==a){d[e].remove();break}}}}}):c.push("FriendRequest/sending");b("Arbiter").subscribe(c,function(c,d){__p&&__p();if(a==d.profile_id&&d.origin=="hovercard"||a==d.uid){c=b("ge")(document.body,".ego_unit_container");if(c){d=b("DOM").scry(c,".ego_unit");c=d.length;for(var e=0;e<c;e++){var f=d[e].getAttribute("data-ego-fbid");if(f==a){f=b("DOM").scry(d[e],".ego_action a")[0];f&&f.click();break}}}}})},updateXids:function(a,c){__p&&__p();if(a.length==0&&c.length==0)return;a=function(a){return function(c){c=c.getAttribute(a);if(!c)return!1;c=new(h||(h=b("URI")))(c).getQueryData();return!!c.xids}};var d=b("DOM").scry(document.body,".ego_section a");d=d.filter(a("ajaxify"));if(d.length==0)return;var e=new(h||(h=b("URI")))(d[0].getAttribute("ajaxify")),f=e.getQueryData();if(!f.xids)return;var g=null;try{g=JSON.parse(f.xids)}catch(a){return}for(var i=0;i<c.length;++i)g[c[i]]=1;var j=JSON.stringify(g);c=function(a,c){e=new(h||(h=b("URI")))(a.getAttribute(c)),f=e.getQueryData(),f.xids=j,e.setQueryData(f),a.setAttribute(c,e.toString())};for(i=0;i<d.length;++i)c(d[i],"ajaxify");g=b("DOM").scry(document.body,".ego_unit form");g=g.filter(a("action"));for(i=0;i<g.length;++i)c(g[i],"action")},replaceUnit:function(a,b,c,d){i.replaceUnitCheckParent(a,b,c,d,"")},replaceUnitCheckParent:function(a,c,d,e,f){var g=b("Parent").byClass(a,"ego_unit_container");if(g&&b("EgoUnitSlideInsert").isSliding(a)){g=b("DOM").appendContent(g,c);g.forEach(b("CSS").hide);b("EgoUnitSlideInsert").runAfterSlide(a,i._replaceUnitElement.bind(null,a,g,f))}else i._replaceUnit(a,c,d,e,f)},_replaceUnit:function(a,c,d,e,f){var g=b("DOM").insertAfter(a,c);g.forEach(b("CSS").hide);e!==void 0&&e!==null?setTimeout(function(){i._replaceUnitFadeout(a,g,d,f)},e):i._replaceUnitFadeout(a,g,d,f)},_replaceUnitFadeout:function(a,c,d,e){d?new(b("Animation"))(a).from("opacity",1).to("opacity",0).duration(d).checkpoint(1,function(){i._replaceUnitElement(a,c,e)}).go():i._replaceUnitElement(a,c,e)},_replaceUnitElement:function(a,c,d){var e=b("CSS").hasClass(a,"ego_unit")?a.parentNode:null;e&&e.tagName==="LI"&&(e=b("DOM").scry(a.parentNode,"^ul")[0]);b("DOM").remove(a);c.length&&c.forEach(b("CSS").show);b("Arbiter").inform("netego_replacedUnit",{serializedData:d,numUnitsRemained:e.childNodes.length});i.clearHeader()},clearHeader:function(){__p&&__p();var a=b("DOM").scry(document.body,".ego_column"),c=[];for(var d=0;d<a.length;++d)c=c.concat(b("DOM").scry(a[d],".uiHeader"));for(d=0;d<c.length;++d){a=c[d].nextSibling;var e=b("DOM").find(a,"._2xq");e||(e=a);if(!e||e.childNodes.length===0)b("DOM").remove(c[d]);else if(e.childNodes.length===1){a=e.childNodes[0];b("CSS").hasClass(a,"ego_appended_units")&&a.childNodes.length===0&&b("DOM").remove(c[d])}}}};e.exports=i}),null);
__d("NetEgoLogger",["AsyncSignal"],(function(a,b,c,d,e,f){e.exports={log:function(a){var c=a.uri,d=a.sid,e=a.fbid;a=a.element;new(b("AsyncSignal"))(c,{sid:d,fbid:e,height:a.offsetHeight}).send()}}}),null);
__d("FeedConsumptionEvents",["keyMirror"],(function(a,b,c,d,e,f){"use strict";a=Object.freeze(b("keyMirror")({PLACEHOLDER_HIDDEN:null,PLACEHOLDER_VISIBLE:null,REQUEST_CANCELLED:null,REQUEST_COMPLETE:null,REQUEST_ERROR:null,REQUEST_STARTED:null,STORY_DISPLAY:null}));e.exports=a}),null);
__d("FeedConsumptionController",["EventEmitter","FeedConsumptionEvents","IntersectionObserver","PageletEventConstsJS","PageletEventsHelper"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(a){this.$3=null,this.$4=null,this.$6=0,this.$2=a,this.$1=new(b("EventEmitter"))()}var c=a.prototype;c.setPlaceholder=function(a){var c=this;!this.$3?this.$3=new(b("IntersectionObserver"))(function(a){return c.$7(a)},{threshold:.1}):this.$5&&this.$3.unobserve(this.$5);this.$5=a;this.$3.observe(a)};c.start=function(){var a=this;this.$6=0;this.$4||(this.$4=b("PageletEventsHelper").subscribeToPageletEvents(function(c,d){d===b("PageletEventConstsJS").DISPLAY_END&&c.startsWith("hyperfeed_story")&&a.$1.emit(b("FeedConsumptionEvents").STORY_DISPLAY,a.$6++)}));this.$1.emit(b("FeedConsumptionEvents").REQUEST_STARTED)};c.complete=function(){this.$1.emit(b("FeedConsumptionEvents").REQUEST_COMPLETE)};c.error=function(){this.$1.emit(b("FeedConsumptionEvents").REQUEST_ERROR)};c.cancel=function(){this.$1.emit(b("FeedConsumptionEvents").REQUEST_CANCELLED)};c.destroy=function(){this.$3&&this.$3.disconnect(),this.$3=null,this.$4&&this.$4.remove(),this.$4=null};c.addListener=function(a,b){return this.$1.addListener(a,b)};c.once=function(a,b){return this.$1.once(a,b)};c.$7=function(a){var c=this,d=this.$5,e=this.$3&&this.$3.thresholds,f=Array.isArray(e)&&e[0]||0;a.forEach(function(a){a.target===d&&c.$1.emit(a.intersectionRatio>f?b("FeedConsumptionEvents").PLACEHOLDER_VISIBLE:b("FeedConsumptionEvents").PLACEHOLDER_HIDDEN,c.$2())})};return a}();e.exports=a}),null);
__d("FeedConsumptionRestorationLogger",["FeedConsumptionEvents","QuickPerformanceLogger","SubscriptionsHandler"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(a){this.$2=0,this.$1=a}var c=a.prototype;c.destroy=function(){this.$3&&this.$3.release(),this.$4&&this.$4.remove()};c.$5=function(a){b("QuickPerformanceLogger").markerEnd(655575,a),this.reset()};c.reset=function(){__p&&__p();var a=this;this.destroy();this.$2=0;this.$4=this.$1.once(b("FeedConsumptionEvents").PLACEHOLDER_VISIBLE,function(c){b("QuickPerformanceLogger").markerStart(655575);b("QuickPerformanceLogger").annotateMarkerInt(655575,"STORY_POSITION",c);a.$3=new(b("SubscriptionsHandler"))();a.$3.addSubscriptions(a.$1.once((c=b("FeedConsumptionEvents")).STORY_DISPLAY,function(c){b("QuickPerformanceLogger").annotateMarkerInt(655575,"TAIL_LOAD_STORY_POSITION",c),a.$5(2)}),a.$1.once(c.PLACEHOLDER_HIDDEN,function(){return a.$5(4)}),a.$1.once(c.REQUEST_CANCELLED,function(){return a.$5(4)}),a.$1.once(c.REQUEST_ERROR,function(){return a.$5(3)}))})};return a}();e.exports=a}),null);
__d("NewsFeedDedupeStoryController",["CSS","CurrentUser","DOMQuery","NewsFeedDedupeStoryTypedLogger","queryThenMutateDOM"],(function(a,b,c,d,e,f){__p&&__p();a={dedupe:function(a,c){__p&&__p();b("queryThenMutateDOM")(function(){var d={};return b("DOMQuery").scry(a,c).filter(function(a){var c=a.getAttribute("data-dedupekey");if(c&&b("CSS").shown(a)){if(d[c])return!0;d[c]=!0}return!1})},function(a){a.forEach(function(a){new(b("NewsFeedDedupeStoryTypedLogger"))().setIsWorkplaceUser(b("CurrentUser").isWorkUser()).setPositionOfDuplicate(a.getAttribute("data-insertion-position")).log(),b("CSS").hide(a),a.removeAttribute("style")})},"NewsFeedDedupeStoryControllerDedupe")}};e.exports=a}),null);
__d("NewsFeedOffscreenController",["cx","Arbiter","CSS","DOMQuery","IntersectionObserver","LitestandShareAttachment","NavigationMessage","Run","Style","SubscriptionsHandler","debounce","getOrCreateDOMID","intersectionObserverEntryIsIntersecting","queryThenMutateDOM"],(function(a,b,c,d,e,f,g){__p&&__p();a=400;var h=10,i="_5qqr",j={},k=null,l={rootMargin:a+"px 0px "+a+"px 0px"},m=null,n={};function o(){for(var a in n)b("DOMQuery").scry(n[a],"div[data-fullscreen]").length?delete n[a]:j[a]=n[a].offsetHeight}function p(){var a;for(var c in n)a=j[c],a!==void 0&&a>0&&(b("Style").apply(n[c],{height:a+"px",marginBottom:h+"px"}),a=n[c].firstElementChild,a&&b("CSS").hide(a),delete j[c]),delete n[c]}var q=b("debounce")(function(){b("queryThenMutateDOM")(o,p,"NewsFeedOffscreenController")});function r(a){delete n[b("getOrCreateDOMID")(a)];b("Style").apply(a,{height:"",marginBottom:""});a=a.firstElementChild;if(a){b("CSS").show(a);a=b("DOMQuery").scry(a,"."+i);a.forEach(function(a){b("LitestandShareAttachment").init(a)})}}function s(a){var c=[];a.forEach(function(a){b("intersectionObserverEntryIsIntersecting")(a)?c.push(a.target):n[b("getOrCreateDOMID")(a.target)]=a.target});q();c.forEach(r)}function t(){k&&k.disconnect(),k=null,m&&m.release(),m=null}function u(){if(k)return k;k=new(b("IntersectionObserver"))(s,l);var a=new(b("SubscriptionsHandler"))();a.addSubscriptions(b("Arbiter").subscribe(b("NavigationMessage").NAVIGATION_BEGIN,t),b("Run").onLeave(t));return k}c={attachSubstream:function(a){u().observe(a)}};e.exports=c}),null);
__d("TailLoadLogger",["invariant","Arbiter","BigPipe","FBLogger","IntersectionObserver","PageEvents","SubscriptionsHandler","performanceNow"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();var h,i=function(){__p&&__p();function a(a,b,c){this.$2=0,this.$3=0,this.$4=a,this.$4.setTailLoadPosition(c),this.$1=c,this.$5=b}var c=a.prototype;c.getTailLoadPosition=function(){return this.$1};c.visibleStart=function(a){this.$3=a;return this};c.visibleEnd=function(a){if(this.$3>0){var c=a-this.$3;c>0&&(this.$2+=c,c>1e6&&b("FBLogger")(this.$5).warn("Logging a placeholder visible duration of over 1M! Duration: %s Start time: %s, End time: %s",this.$2,this.$3,a),this.$3=0)}return this};c.setDisplayDone=function(a){this.$4.setDisplayDone(Math.round(a));return this};c.setTailLoadStatus=function(a){this.$4.setTailLoadStatus(a);return this};c.log=function(){this.$4.setVisibleDuration(Math.round(this.$2)).log()};return a}();a=function(){__p&&__p();function a(a,c,d){__p&&__p();var e=this;this.$1={};this.$5={};this.$7=a;this.$4=d;this.$3=new(b("IntersectionObserver"))(function(a){return e.$8(a)},{threshold:c});this.$6=new(b("SubscriptionsHandler"))();this.$6.addSubscriptions(b("Arbiter").subscribe(b("BigPipe").Events.init,function(a,b){return e.$9(b)}))}var c=a.prototype;c.createLogger=function(){g(0,4408)};c.getStoryCount=function(){g(0,4409)};c.setQPLEndMarker=function(a){};c.setQPLMarkerPoint=function(){};c.setQPLStartMarker=function(){};c.destroy=function(){this.$3.disconnect(),this.$6.release()};c.tailLoadStarted=function(){this.setQPLStartMarker(),this.$2||(this.$3.observe(this.$4),this.$2=new i(this.createLogger(),this.$7,this.getStoryCount()))};c.tailLoadSuccessful=function(a){this.setQPLEndMarker(2);var c=this.$2;c&&(c.visibleEnd((h||(h=b("performanceNow")))()).setTailLoadStatus("success"),this.$1[a]=c,this.$10(a),this.$11())};c.tailLoadFailed=function(){this.setQPLEndMarker(3);var a=this.$2;a&&(a.visibleEnd((h||(h=b("performanceNow")))()).setDisplayDone(-1).setTailLoadStatus("failed").log(),this.$11())};c.tailLoadCancelled=function(){this.setQPLEndMarker(4);var a=this.$2;a&&(a.visibleEnd((h||(h=b("performanceNow")))()).setDisplayDone(-1).setTailLoadStatus("cancelled").log(),this.$11())};c.$11=function(){this.$3.unobserve(this.$4),this.$2=null};c.$8=function(a){var b=this,c=this.$2;if(!c)return;var d=this.$3.thresholds,e=Array.isArray(d)&&d[0]||0;a.forEach(function(a){a.target===b.$4&&(a.intersectionRatio>e?c.visibleStart(a.time):(b.setQPLMarkerPoint(),c.visibleEnd(a.time)))})};c.$9=function(a){var c=this,d=a.arbiter;a=a.lid;var e={requestStart:0};this.$5[a]=e;d.subscribe(b("PageEvents").AJAXPIPE_SEND,function(a,b){a=b.ts;return e.requestStart=a});d.subscribe(b("BigPipe").Events.displayed,function(a,b){return c.$12(b)})};c.$12=function(a){var c=a.lid;a=a.ts;var d=this.$5[c];if(!d)return;var e=a-d.requestStart;e<=0&&b("FBLogger")(this.$7).warn("Tail load display done is invalid! Start: %s, Display Done: %s",d.requestStart,a);d.displayDone=e;this.$10(c)};c.$10=function(a){var b=this.$5[a],c=this.$1[a];b&&c&&b.displayDone!=null&&(b.displayDone>0&&c.setDisplayDone(b.displayDone).log(),delete this.$1[a],delete this.$5[a])};return a}();e.exports=a}),null);
__d("NewsFeedTailLoadPlaceholderLogger",["DOM","LitestandStream","NewsFeedTailloadPlaceholderTypedLogger","TailLoadLogger"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(a){babelHelpers.inheritsLoose(c,a);function c(b,c){return a.call(this,"feed",b,c)||this}var d=c.prototype;d.createLogger=function(){return new(b("NewsFeedTailloadPlaceholderTypedLogger"))()};d.getStoryCount=function(){return b("DOM").scry(b("LitestandStream").getStreamRoot(),b("LitestandStream").getStoriesSelector()).length};return c}(b("TailLoadLogger"));e.exports=a}),null);
__d("FeedRequestWaterfallTypedLoggerLite",["Banzai","generateLiteTypedLogger"],(function(a,b,c,d,e,f){"use strict";e.exports=b("generateLiteTypedLogger")("logger:FeedRequestWaterfallLoggerConfig")}),null);
__d("WebFeedRequestWaterfallLogger",["csx","CurrentUser","DOM","FBLogger","FeedRequestWaterfallTypedLoggerLite","LitestandStream","gkx"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();var h=b("FeedRequestWaterfallTypedLoggerLite").log;a=function(){__p&&__p();a.create=function(c,d){d===void 0&&(d=!1);return!b("gkx")("927948")?null:new a(c,d)};function a(a,b){this.$1=0,this.$2="none",this.$3=a,b&&(this.$2="started")}var c=a.prototype;c.$4=function(){try{return b("DOM").scry(b("LitestandStream").getStreamRoot(),"._7cqq").length}catch(a){b("FBLogger")("web_feed_request_waterfall").mustfix("Unable to determine number of stories in DOM: %s",a.message);return 0}};c.$5=function(){return this.$4()-this.$1};c.start=function(){if(this.$2!=="none"){b("FBLogger")("feed_request_waterfall").mustfix("start() was called with an invalid state: %s",this.$2);return}this.$2="started";this.$6({event:"client_sent_request"});this.$1=this.$4()};c.success=function(){if(this.$2!=="started"){b("FBLogger")("feed_request_waterfall").mustfix("success() was called with an invalid state: %s.",this.$2);return}this.$2="complete";var a=this.$5();this.$6({event:"client_received_response",number_of_stories:a});this.$6({event:"client_rendered_success",number_of_stories:a})};c.failed=function(a){if(this.$2!=="started"){b("FBLogger")("feed_request_waterfall").mustfix("failed() was called with an invalid state: %s.",this.$2);return}this.$2="complete";a=a!=null&&a.error!==0?a.toError():null;this.$6({event:this.$7(a),number_of_stories:this.$5(),failure_reason:a==null?void 0:a.message})};c.$7=function(a){__p&&__p();if(a!=null)if(a.code===408)return"client_network_timeout";else if(a.code>=400&&a.code<=599)return"client_failed_to_receive_response";else if(a.code>=1001&&a.code<=1008)return"client_failed_to_receive_response";else if(a.code===1011||a.code===1013)return"client_rendered_cancelled";return"client_rendered_failed"};c.$6=function(a){h(babelHelpers["extends"]({client_query_id:this.$3,is_employee:b("CurrentUser").isEmployee()},a))};return a}();e.exports=a}),null);
__d("WebFeedUnitDetailsTypedLoggerLite",["Banzai","generateLiteTypedLogger"],(function(a,b,c,d,e,f){"use strict";e.exports=b("generateLiteTypedLogger")("logger:WebFeedUnitDetailsLoggerConfig")}),null);
__d("WebFeedUnitDetailsLogger",["csx","DOM","FBLogger","LitestandStream","WebFeedUnitDetailsTypedLoggerLite","gkx"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();var h=b("WebFeedUnitDetailsTypedLoggerLite").log,i=new Set();function j(){try{return b("DOM").scry(b("LitestandStream").getStreamRoot(),"._7cqq").map(function(a){return a.parentElement})}catch(a){b("FBLogger")("web_feed_unit_details").catching(a).mustfix("Unable to query rendered stories in DOM.");return[]}}function k(){try{return b("DOM").scry(b("LitestandStream").getStreamRoot(),b("LitestandStream").getStoriesSelector())}catch(a){b("FBLogger")("web_feed_unit_details").catching(a).mustfix("Unable to query all stories in DOM.");return[]}}function l(a,b,c){var d=a.getAttribute("data-dedupekey");if(d==null||i.has(d))return;c=c.indexOf(a);h({position:c!==-1?c:null,absolute_position:b,dedup_key:d,is_rendered:c!==-1});i.add(d)}function a(){if(!b("gkx")("970508"))return;var a=j();k().forEach(function(b,c){return l(b,c,a)})}e.exports={logStories:a}}),null);
__d("NewsFeedTailLoadController",["Arbiter","Bootloader","CSS","DOM","ErrorUtils","Event","FBLogger","FeedConsumptionController","FeedConsumptionRestorationLogger","FunnelLogger","LitestandMessages","LitestandStream","LitestandStreamTailStoryPlaceholderVisibilityTracker","NavigationMessage","NewsFeedDedupeStoryController","NewsFeedOffscreenController","NewsFeedTailLoadPlaceholderLogger","Run","SubscriptionsHandler","WebFeedRequestWaterfallLogger","WebFeedUnitDetailsLogger","clearTimeout","ge","getOrCreateDOMID","gkx","intersectionObserverEntryIsIntersecting","observeIntersection","setTimeoutAcrossTransitions","uuid"],(function(a,b,c,d,e,f){__p&&__p();var g,h={"streamload.tail.commence":!0,"streamload.tail.success":!0,"streamload_error.tail":!0,"streamload_error.tail.transport":!0,"streamload_error.tail.final":!0,"streamload.tail.retry":!0,"streamload_error.tail.retry":!0,"streamload.tail.retry.success":!0,"streamload_timeout.tail":!0},i="WWW_NEWSFEED_TAILLOAD_FUNNEL",j=2,k=3e4,l=null;function m(){if(!l)return;l.destroy();l=null}a=function(){"use strict";__p&&__p();function a(a){__p&&__p();var c=this,d=a.container,e=a.cursor,f=a.pager,g=a.pagerConfig,h=a.placeholderLoggingIntersectionThreshold,i=a.streamConfig;a=a.clientQueryId;this.$3=!1;this.$5=null;this.$6=null;this.$7=null;this.$8=null;this.$11=null;this.$12={};this.$14=0;this.$15=0;this.$17=new(b("SubscriptionsHandler"))();this.$20=function(){b("clearTimeout")(c.$5)};this.$19=function(){c.$20(),c.$5=b("setTimeoutAcrossTransitions")(function(){c.$23(["streamload_timeout.tail"])},k)};this.$1=d;this.$2=e;this.$9=f;this.$10=g;this.$16=i;b("LitestandStreamTailStoryPlaceholderVisibilityTracker").setPlaceholderElement(this.$9);var j=new(b("FeedConsumptionController"))(function(){return b("DOM").scry(b("LitestandStream").getStreamRoot(),b("LitestandStream").getStoriesSelector()).length});j.setPlaceholder(this.$9);this.$17.addSubscriptions({remove:function(){j.cancel(),j.destroy()}});this.$4=j;if(b("gkx")("678702")){var l=new(b("FeedConsumptionRestorationLogger"))(this.$4);l.reset();this.$17.addSubscriptions({remove:function(){l.destroy()}})}this.$17.addSubscriptions(this.$18(),b("Arbiter").subscribe(b("LitestandMessages").STORIES_REQUESTED,this.$19),b("Arbiter").subscribe([b("LitestandMessages").STORIES_INSERTED,b("LitestandMessages").STREAM_LOAD_ERROR],this.$20),b("Arbiter").subscribe(b("NavigationMessage").NAVIGATION_BEGIN,m),b("Event").listen(this.$9,"click",function(a){c.$21(),a.preventDefault()}),b("Run").onLeave(m));if(h!=null){var n=new(b("NewsFeedTailLoadPlaceholderLogger"))(h,this.$9);this.$17.addSubscriptions({remove:function(){n.tailLoadCancelled(),n.destroy()}});this.$11=n}if(a!=null){d=b("WebFeedRequestWaterfallLogger").create(a,!0);d&&d.success()}b("WebFeedUnitDetailsLogger").logStories()}var c=a.prototype;c.destroy=function(){if(this.$3)return;this.$3=!0;this.$20();this.$17.release()};c.handleError=function(a,c){__p&&__p();var d=this;if(this.$3)return;b("Arbiter").inform(b("LitestandMessages").STREAM_LOAD_ERROR,c);(g||(g=b("ErrorUtils"))).applyWithGuard(function(){d.$13&&(d.$13.failed(c),d.$13=null)});var e=["streamload_error.tail"];this.$14>0&&e.push("streamload_error.tail.retry");a!=null&&this.$14<j?(this.$14++,this.$2=a,this.$22(),e.push("streamload.tail.retry"),b("Arbiter").inform(b("LitestandMessages").STREAM_LOAD_RETRY)):(b("CSS").removeClass(this.$9,"async_saving"),this.$11&&this.$11.tailLoadFailed(),this.$4.error());this.$23(e)};c.handleMoreStoriesInserted=function(){if(this.$3)return;this.$14>0&&(this.$14=0,this.$23(["streamload.tail.retry.success"]))};c.handleTailLoadCompleted=function(a,c){__p&&__p();if(this.$3)return;b("CSS").removeClass(this.$9,"async_saving");this.$2=a;this.$17.addSubscriptions(this.$18());this.$14=0;this.$11&&this.$11.tailLoadSuccessful(c);this.$4.complete();this.$13&&(this.$13.success(),this.$13=null);b("WebFeedUnitDetailsLogger").logStories()};c.hidePager=function(){if(this.$3)return;b("CSS").hide(this.$9)};c.setPagerConfig=function(a){this.$10=a};c.$22=function(){__p&&__p();var a=this,c=this.$2;this.$14<1&&(b("FunnelLogger").startFunnel(i,this.$15),b("FunnelLogger").addFunnelTag(i,this.$15,"NewsFeedTailLoadController"));this.$23(["streamload.tail.commence"]);this.$12[c]&&(this.$23(["duplicate_cursor"]),b("Arbiter").inform(b("LitestandMessages").DUPLICATE_CURSOR_ERROR,c));this.$12[c]=!0;var d=b("uuid")();this.$13=b("WebFeedRequestWaterfallLogger").create(d);var e={client_stories_count:b("DOM").scry(b("LitestandStream").getStreamRoot(),b("LitestandStream").getStoriesSelector()).length,cursor:c,pager_config:this.$10,scroll_count:this.$15+1,story_id:b("LitestandStream").getStoryID(),client_query_id:d},f=function(b){a.handleError(c,b)},g=function(b){b.error?a.$23(["streamload_error.tail.final"]):(a.$23(["streamload.tail.success"]),a.$15++)},h=function(b){a.$23(["streamload_error.tail.transport"]),a.handleError(c,b)},j=this.$7;if(!j){this.$8&&(this.$8.remove(),this.$8=null);d=b("Bootloader").loadModules(["UIPagelet"],function(c){a.$8=null,a.$7=j=c.loadFromEndpoint("LitestandTailLoadPagelet",b("getOrCreateDOMID")(a.$1),e,{append:!0,automatic:!0,constHeight:!0,crossPage:!0,errorHandler:f,finallyHandler:g,transportErrorHandler:h,usePipe:!0}),a.$17.addSubscriptions({remove:function(){j&&j.abandon()}}),b("Arbiter").inform(b("LitestandMessages").STORIES_REQUESTED)},"NewsFeedTailLoadController");this.$17.addSubscriptions(d);this.$8=d}else j.setErrorHandler(f).setTransportErrorHandler(h).go(e),b("Arbiter").inform(b("LitestandMessages").STORIES_REQUESTED);this.$13&&this.$13.start()};c.$23=function(a){var c=this;a.forEach(function(a){h[a]&&b("FunnelLogger").appendActionWithTag(i,c.$15,a,"try_"+c.$14)});b("Bootloader").loadModules(["BanzaiODS"],function(b){a.forEach(function(a){b.bumpEntityKey(2966,"feed_load",a)})},"NewsFeedTailLoadController")};c.$18=function(){var a=this,c;this.$6&&this.$6.remove();this.$6=b("observeIntersection")(this.$9,function(c){c=b("intersectionObserverEntryIsIntersecting")(c);c&&(a.$21(),a.$6&&(a.$6.remove(),a.$6=null))},{rootMargin:"0px 0px "+((c=this.$16.bufferPixels)!=null?c:0)+"px 0px"});return this.$6};c.$21=function(){b("CSS").addClass(this.$9,"async_saving"),this.$22(),this.$4.start(),this.$11&&this.$11.tailLoadStarted()};a.destroy=function(){l&&(l.hidePager(),m())};a.handleErrorV2=function(a){l&&l.handleError(a)};a.initV2=function(c){l&&(b("FBLogger")("news_feed_tail_load").warn("The server called `NewsFeedTailLoadController.init()` but an instance of this singleton already exists. Please call `NewsFeedTailLoadController.destroy()` before creating a new one. To mitigate this, the old one has been destroyed for you."),m()),l=new a(c)};a.moreStoriesInserted=function(a){if(!l){b("FBLogger")("news_feed_tail_load").warn("The server called `NewsFeedTailLoadController.moreStoriesInserted()` without an active instance of that controller. Create one first using `NewsFeedTailLoadController.init()` and make sure not to send this sort of signal to the client after having called `NewsFeedTailLoadController.destroy()`.");return}b("Arbiter").inform(b("LitestandMessages").STORIES_INSERTED,{substream_id:a});a=b("ge")(a);a&&(b("NewsFeedOffscreenController").attachSubstream(a),l&&l.handleMoreStoriesInserted())};a.setPagerConfig=function(a){if(!l){b("FBLogger")("news_feed_tail_load").warn("The server called `NewsFeedTailLoadController.setPagerConfig()` without an active instance of that controller. Create one first using `NewsFeedTailLoadController.init()` and make sure not to send this sort of signal to the client after having called `NewsFeedTailLoadController.destroy()`.");return}l.setPagerConfig(a)};a.tailloadCompletedV2=function(a){var c=a.cursor;a=a.lid;if(!l){b("FBLogger")("news_feed_tail_load").warn("The server called `NewsFeedTailLoadController.tailloadCompleted()` without an active instance of that controller. Create one first using `NewsFeedTailLoadController.init()` and make sure not to send this sort of signal to the client after having called `NewsFeedTailLoadController.destroy()`.");return}var d=b("LitestandStream").getStreamRoot();d&&b("NewsFeedDedupeStoryController").dedupe(d,b("LitestandStream").getStoriesSelector());l.handleTailLoadCompleted(c,a)};return a}();e.exports=a}),null);
__d("getInvisibleIframe",["DOM"],(function(a,b,c,d,e,f){"use strict";__p&&__p();e.exports=function(c,d){__p&&__p();var e=b("DOM").create("iframe",{width:0,height:0,frameborder:0,scrolling:"no","aria-hidden":"true",style:{opacity:0,pointerEvents:"none",position:"absolute"}});c!==void 0&&e.setAttribute("src",c);d&&(e.onload=d);b("DOM").appendContent(document.body,e);function f(b){b.__annotator=function(a){return a},b.__d_stub=[],b.__d=function(){b.__d_stub.push(arguments)}}function g(b){b.onerror=function(a,b,c,d,e){return!0}}f(e.contentWindow);g(e.contentWindow);return e}}),null);
__d("ResourcesRefetcher",["Arbiter","DOM","PageEvents","getEarlyResources","getInvisibleIframe"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g=0,h=0;a={_iframe:null,refetchScripts:function(a){var c=this,d=this._getIframe();g=a.length;for(var e=0;e<a.length;e++){var f=b("DOM").create("script",{src:a[e].getAttribute("src"),crossorigin:"anonymous"});f.onload=function(){c.checkForFinished()};b("DOM").appendContent(d.contentDocument.head,f)}},_getIframe:function(){this._iframe===null&&(this._iframe=b("getInvisibleIframe")());return this._iframe},refetchOnLoad:function(a){var c=this;b("Arbiter").subscribeOnce(b("PageEvents").BIGPIPE_ONLOAD,function(){var d=b("getEarlyResources")();a.allEF||(d=d.slice(0,1));c.refetchScripts(d)})},checkForFinished:function(){if(++h<g)return;this._iframe&&(b("DOM").remove(this._iframe),this._iframe=null,h=0)}};e.exports=a}),null);