if (self.CavalryLogger) { CavalryLogger.start_js(["VPiTk"]); }

__d("listenForParentIntegrityContextDialogExit",["Arbiter","DOMQuery"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g="IntegrityContextDialogFactory/dialogExit";function a(a,c){return b("Arbiter").subscribe(g,function(d,e){d=e||{};e=d.dialogRootElem;if(!e)return;b("DOMQuery").contains(e,a)&&c()})}e.exports=a}),null);
__d("ArticleContextDialogShareMapFactory",["cx","BootloadedComponent.react","JSResource","React","ReactDOM","Run","SubscriptionsHandler","XUISpinner.react","listenForParentIntegrityContextDialogExit"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();function h(a,c){b("ReactDOM").render(b("React").jsx(b("BootloadedComponent.react"),babelHelpers["extends"]({bootloadLoader:b("JSResource")("ArticleContextDialogShareMap.react").__setRef("ArticleContextDialogShareMapFactory"),bootloadPlaceholder:b("React").jsx("div",{className:"_60q8",children:b("React").jsx(b("XUISpinner.react"),{size:"large"})})},a)),c);var d=new(b("SubscriptionsHandler"))();a=function(){b("ReactDOM").unmountComponentAtNode(c),d.release()};d.addSubscriptions(b("Run").onLeave(a),b("Run").onUnload(a),b("listenForParentIntegrityContextDialogExit")(c,a))}e.exports={factory:function(a){var b=a.props;a=a.reactRootElem;if(!a)return;h(b,a)}}}),null);
__d("IntegrityContextDialogBody",["DOM","Event","OnVisible","Parent","SubscriptionsHandler"],(function(a,b,c,d,e,f){"use strict";__p&&__p();function g(a,c){if(!c)return null;var d=null,e=b("Parent").byTag(c,"a");e&&b("DOM").contains(a,e)&&(d=e);a=c.tagName;e=c.getAttribute("class")||null;c=d&&d.getAttribute("href")||null;d=d&&d.getAttribute("ajaxify")||null;return{target_tagname:a,target_classes:e,link_href:c,link_ajaxify:d,targetTagname:a,targetClasses:e,linkHref:c,linkAjaxify:d}}a=function(){__p&&__p();function a(a){a=a.modules;this.$1=a||[];this.$2=null;this.$3=null}var c=a.prototype;c.$4=function(){this.$2&&this.$2.release(),this.$2=null};c.$5=function(){this.$3&&this.$3.release(),this.$3=null};c.destroy=function(){this.$4(),this.$5()};c.initModuleVPVHandler=function(a){this.$4();var c=new(b("SubscriptionsHandler"))();this.$1.forEach(function(d,f){var g=d.moduleRoot,h=d.moduleName;c.addSubscriptions(new(b("OnVisible"))(g,function(){a(h,f)},!1,0))});this.$2=c};c.initModuleClickHandler=function(a){this.$5();var c=new(b("SubscriptionsHandler"))();this.$1.forEach(function(d,f){var h=d.moduleRoot,i=d.moduleName;c.addSubscriptions(b("Event").listen(h,"click",function(b){a(i,f,g(h,b.target))}))});this.$3=c};c.getModuleNames=function(){return this.$1.map(function(a){return a.moduleName})};return a}();e.exports=a}),null);
__d("IntegrityContextClientTypedLogger",["Banzai","GeneratedLoggerUtils","nullthrows"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(){this.$1={}}var c=a.prototype;c.log=function(){b("GeneratedLoggerUtils").log("logger:IntegrityContextClientLoggerConfig",this.$1,b("Banzai").BASIC)};c.logVital=function(){b("GeneratedLoggerUtils").log("logger:IntegrityContextClientLoggerConfig",this.$1,b("Banzai").VITAL)};c.logImmediately=function(){b("GeneratedLoggerUtils").log("logger:IntegrityContextClientLoggerConfig",this.$1,{signal:!0})};c.clear=function(){this.$1={};return this};c.getData=function(){return babelHelpers["extends"]({},this.$1)};c.updateData=function(a){this.$1=babelHelpers["extends"]({},this.$1,a);return this};c.setElapsedMs=function(a){this.$1.elapsed_ms=a;return this};c.setEntryPoint=function(a){this.$1.entry_point=a;return this};c.setEvent=function(a){this.$1.event=a;return this};c.setFeedTracking=function(a){this.$1.feed_tracking=a;return this};c.setInstanceID=function(a){this.$1.instance_id=a;return this};c.setIntegrityContextType=function(a){this.$1.integrity_context_type=a;return this};c.setModuleIndex=function(a){this.$1.module_index=a;return this};c.setModuleName=function(a){this.$1.module_name=a;return this};c.setTime=function(a){this.$1.time=a;return this};c.setTriggerLoggerID=function(a){this.$1.trigger_logger_id=a;return this};c.setVC=function(a){this.$1.vc=a;return this};c.setWeight=function(a){this.$1.weight=a;return this};c.updateExtraData=function(a){a=b("nullthrows")(b("GeneratedLoggerUtils").serializeMap(a));b("GeneratedLoggerUtils").checkExtraDataFieldNames(a,g);this.$1=babelHelpers["extends"]({},this.$1,a);return this};c.addToExtraData=function(a,b){var c={};c[a]=b;return this.updateExtraData(c)};return a}();var g={elapsed_ms:!0,entry_point:!0,event:!0,feed_tracking:!0,instance_id:!0,integrity_context_type:!0,module_index:!0,module_name:!0,time:!0,trigger_logger_id:!0,vc:!0,weight:!0};e.exports=a}),null);
__d("IntegrityContextTriggerClientTypedLogger",["Banzai","GeneratedLoggerUtils","nullthrows"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(){this.$1={}}var c=a.prototype;c.log=function(){b("GeneratedLoggerUtils").log("logger:IntegrityContextTriggerClientLoggerConfig",this.$1,b("Banzai").BASIC)};c.logVital=function(){b("GeneratedLoggerUtils").log("logger:IntegrityContextTriggerClientLoggerConfig",this.$1,b("Banzai").VITAL)};c.logImmediately=function(){b("GeneratedLoggerUtils").log("logger:IntegrityContextTriggerClientLoggerConfig",this.$1,{signal:!0})};c.clear=function(){this.$1={};return this};c.getData=function(){return babelHelpers["extends"]({},this.$1)};c.updateData=function(a){this.$1=babelHelpers["extends"]({},this.$1,a);return this};c.setEntryPoint=function(a){this.$1.entry_point=a;return this};c.setEvent=function(a){this.$1.event=a;return this};c.setFeedTracking=function(a){this.$1.feed_tracking=a;return this};c.setIntegrityContextType=function(a){this.$1.integrity_context_type=a;return this};c.setTime=function(a){this.$1.time=a;return this};c.setTriggerLoggerID=function(a){this.$1.trigger_logger_id=a;return this};c.setVC=function(a){this.$1.vc=a;return this};c.setWeight=function(a){this.$1.weight=a;return this};c.updateExtraData=function(a){a=b("nullthrows")(b("GeneratedLoggerUtils").serializeMap(a));b("GeneratedLoggerUtils").checkExtraDataFieldNames(a,g);this.$1=babelHelpers["extends"]({},this.$1,a);return this};c.addToExtraData=function(a,b){var c={};c[a]=b;return this.updateExtraData(c)};return a}();var g={entry_point:!0,event:!0,feed_tracking:!0,integrity_context_type:!0,time:!0,trigger_logger_id:!0,vc:!0,weight:!0};e.exports=a}),null);
__d("IntegrityContextTriggerLoggerManager",["Event","IntegrityContextTriggerClientTypedLogger","OnVisible","Run","SubscriptionsHandler"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g={},h=Object.freeze({TRIGGER_VPV:"trigger_vpv",TRIGGER_HOVER:"trigger_hover",TRIGGER_CLICK:"trigger_click"}),i=function(){__p&&__p();function a(a,c,d){__p&&__p();var e=a.triggerLinkElem,f=a.loggerID,g=a.serializedFTMsg,h=a.integrityContextType;a=a.entryPoint;this.$1=e;this.$2=f;this.$3=g;this.$4=h;this.$6=null;this.$5=a;this.$7=c;e=this.destroy.bind(this);f=new(b("SubscriptionsHandler"))();f.addSubscriptions(b("Run").onLeave(e),b("Run").onUnload(e));d||this.$9(f);this.$8=f}var c=a.prototype;c.$10=function(){var a=this,c=new(b("SubscriptionsHandler"))();c.addSubscriptions(new(b("OnVisible"))(this.$1,function(){c.release(),a.onFirstVisible()},!0,0));return c};c.$11=function(){var a=this,c=new(b("SubscriptionsHandler"))();c.addSubscriptions(b("Event").listen(this.$1,"mouseenter",function(){c.release(),a.onFirstMouseEnter()}));return c};c.$9=function(a){var c=this,d=this.$10(),e=this.$11();a.addSubscriptions(b("Event").listen(this.$1,"click",function(){c.onClick()}),{remove:function(){d.release(),e.release()}})};c.destroy=function(){this.$8.release(),delete g[this.$2]};c.onClick=function(){this.$6=new Date().valueOf(),this.$12(h.TRIGGER_CLICK)};c.onFirstMouseEnter=function(){this.$12(h.TRIGGER_HOVER)};c.onFirstVisible=function(){this.$12(h.TRIGGER_VPV)};c.getLastTriggerTimeMS=function(){return this.$6};c.$12=function(a){a=new(b("IntegrityContextTriggerClientTypedLogger"))().setEvent(a).setTriggerLoggerID(this.$2).setIntegrityContextType(this.$4).setEntryPoint(this.$5);this.$3&&a.setFeedTracking(this.$3);var c=this.$7&&this.$7();c&&a.updateExtraData(c);a.log()};return a}();function j(a){return g[a]||null}function k(a){a=j(a);a&&a.destroy()}function l(a,b,c){var d=a.loggerID;k(d);a=new i(a,b,c);g[d]=a;return a}function a(a,b){b===void 0&&(b=null);return l(a,b,!1)}function c(a){return l(a,null,!0)}function d(a){a=j(a);return a&&a.getLastTriggerTimeMS()||null}e.exports={initLogger:a,initLoggerWithCustomEventListeners:c,clearLogger:k,getLastTriggerTimeMS:d}}),null);
__d("IntegrityContextDialogFactory",["Arbiter","AsyncRequest","IntegrityContextClientTypedLogger","IntegrityContextDialogBody","IntegrityContextTriggerLoggerManager","Run","SubscriptionsHandler","URI","isFacebookURI"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g,h=Object.freeze({LOAD_END:"load_end",MODULE_VPV:"module_vpv",MODULE_CLICK:"module_click",DIALOG_EXIT:"dialog_exit"}),i="IntegrityContextDialogFactory/dialogExit";function j(a){var c=a.dialog,d=a.body,e=a.loggerConfig;a=babelHelpers.objectWithoutPropertiesLoose(a,["dialog","body","loggerConfig"]);if(!c||!d||!e)return null;return!(d instanceof b("IntegrityContextDialogBody"))?null:babelHelpers["extends"]({dialog:c,body:d,loggerConfig:e},a)}var k=function(){__p&&__p();a.factory=function(b){b=j(b);return!b?null:new a(b)};function a(a){__p&&__p();var c=a.dialog,d=a.body,e=a.loggerConfig;a=a.onHideAsyncURI;this.$3=c;this.$4=d;this.$5=!1;d=a&&new(g||(g=b("URI")))(a);this.$6=d&&b("isFacebookURI")(d)?d:null;this.$7=e;this.$8=this.$10();this.$9=new Date().valueOf();this.$11(c);this.$12(h.LOAD_END)}var c=a.prototype;c.$10=function(){var a=this.$7.triggerLoggerID;return!a?null:b("IntegrityContextTriggerLoggerManager").getLastTriggerTimeMS(a)};c.$11=function(a){var c=this,d=this.$13.bind(this),e=new(b("SubscriptionsHandler"))();e.addSubscriptions(a.subscribe("hide",this.$14.bind(this)),b("Run").onLeave(d),b("Run").onUnload(d));this.$1=e;d=new(b("SubscriptionsHandler"))();d.addSubscriptions(a.subscribe("afterexpand",function(){c.$15(),c.$16()}));this.$2=d};c.$15=function(){var a=this,b=this.$4;b.initModuleVPVHandler(function(b,c){a.$17(h.MODULE_VPV,b,c)});b.initModuleClickHandler(function(b,c,d){a.$17(h.MODULE_CLICK,b,c,d)})};c.$16=function(){this.$2&&this.$2.release(),this.$2=null};c.destroy=function(){this.$16(),this.$1.release(),this.$4.destroy()};c.$18=function(){this.$12(h.DIALOG_EXIT),b("Arbiter").inform(i,{dialogRootElem:this.$3.getContentRoot()}),this.destroy()};c.$14=function(){this.$19(),this.$18()};c.$13=function(){this.$18()};c.$19=function(){if(this.$5||!this.$6)return;this.$5=!0;new(b("AsyncRequest"))(this.$6).send()};c.$20=function(a,c){__p&&__p();var d=this.$7,e=d.instanceID,f=d.triggerLoggerID,g=d.serializedFTMsg,i=d.contextType,j=d.entryPoint;d=d.baseDialogExtraData;e=new(b("IntegrityContextClientTypedLogger"))().setEvent(a).setInstanceID(e).setTriggerLoggerID(f).setIntegrityContextType(i).setEntryPoint(j);g&&e.setFeedTracking(g);d&&e.updateExtraData(d);c&&e.updateExtraData(c);if(a===h.LOAD_END){f=this.$8;!f||f>this.$9?e.setElapsedMs(null):e.setElapsedMs(this.$9-f)}else if(a===h.DIALOG_EXIT){i=new Date().valueOf()-this.$9;e.setElapsedMs(i)}return e};c.$12=function(a,b){this.$20(a,b).log()};c.$17=function(a,b,c,d){this.$20(a,d).setModuleName(b).setModuleIndex(c).log()};return a}();e.exports={factory:function(a){return k.factory(a)}}}),null);