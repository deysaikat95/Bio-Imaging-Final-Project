if (self.CavalryLogger) { CavalryLogger.start_js(["bcB2h"]); }

__d("asyncSleep",["regeneratorRuntime","Promise"],(function(a,b,c,d,e,f){"use strict";__p&&__p();function a(a){return b("regeneratorRuntime").async(function(c){while(1)switch(c.prev=c.next){case 0:c.next=2;return b("regeneratorRuntime").awrap(new(b("Promise"))(function(b){return setTimeout(b,a)}));case 2:case"end":return c.stop()}},null,this)}e.exports=a}),null);
__d("NotificationGenericBucket",["NotificationSeenState"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(a){var b=this;this.$2=[];this.$3={};this.$4={};this.$7=function(a,c){a=b.$4[a];c=b.$4[c];return Number(c)-Number(a)};this.$1=a}var c=a.prototype;c.reset=function(){this.$2=[],this.$3={},this.$4={}};c.getType=function(){return this.$1.bucket_type};c.getTitle=function(){return this.$1.title};c.getSortedIDs=function(){return this.$2.slice()};c.insertIfEligible=function(a){var b=this.$5(a);if(!b.eligible)return b;this.$6(a);return{eligible:!0}};c.remove=function(a){if(!this.$3[a])return!1;delete this.$3[a];delete this.$4[a];a=this.$2.indexOf(a);a>-1&&this.$2.splice(a,1);return!0};c.$6=function(a){__p&&__p();var b=a.alert_id;if(this.$3[b])return!0;this.$2.push(b);this.$3[b]=!0;var c=this.$8(this.$1.sort_key_index);if(c>-1){a=a.sort_keys?a.sort_keys[c]:null;a&&(this.$4[b]=a,this.$9())}return!0};c.$9=function(){this.$2.sort(this.$7)};c.$5=function(a){__p&&__p();if(!this.$10(a))return{eligible:!1,type:"bucket",data:a.eligible_buckets};if(!this.$11(a))return{eligible:!1,type:"exp_time"};if(!this.$12(a))return{eligible:!1,type:"seen_filter"};if(!this.$13(a))return{eligible:!1,type:"max_count"};if(!this.$14(a))return{eligible:!1,type:"seen_evict",data:this.$15(a)};return!this.$16(a)?{eligible:!1,type:"read_evict"}:{eligible:!0}};c.$8=function(a){return Number.isInteger(a)?Number(a):-1};c.$10=function(a){return!a.eligible_buckets?!1:a.eligible_buckets.includes(this.getType())};c.$12=function(a){return b("NotificationSeenState").validateFilter(a.alert_id,this.$1.seen_filter)};c.$11=function(a){var b=this.$8(this.$1.min_to_expire);return b===-1?!0:Date.now()-a.creation_time<b*60*1e3};c.$13=function(a){a=this.$8(this.$1.max_count);return a===-1?!0:this.$2.length<a};c.$15=function(a){var b=this.$8(this.$1.sec_to_evict_seen);return{secToEvict:b,firstTime:a.first_seen_time||0,currentTime:Date.now()/1e3}};c.$17=function(a){var b=this.$8(this.$1.sec_to_evict_read);return{secToEvict:b,firstTime:a.first_read_time||0,currentTime:Date.now()/1e3}};c.$18=function(a){var b=a.secToEvict,c=a.firstTime;a=a.currentTime;return b===-1||c===0?!0:a-c<b};c.$14=function(a){return this.$18(this.$15(a))};c.$16=function(a){return this.$18(this.$17(a))};return a}();e.exports=a}),null);
__d("NotificationServerTransport",["invariant","DOM","UIPagelet","compactArray","ifRequired","objectValues","once","promiseDone","setImmediateAcrossTransitions","uniqueID"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();var h={},i={},j={},k={};function l(a){k[a]==null&&(k[a]=b("DOM").create("div",{"class":"hidden_elem",id:a}),b("DOM").appendContent(document.documentElement,k[a]));return k[a].id}function m(a){return a.reduce(function(a,b){b=b.targetNumToLoad;return Math.max(a,b)},0)}function n(a,b){var c=b.getHash(),d=h[c];if(!d||d.clientRequestID!==a)return;delete h[c];d.calls.forEach(function(a){a=a.onCompleted;a&&a()});i[c]!=null&&p(b)}function o(a,b,c){var d=b.getHash(),e=h[d];if(!e||e.clientRequestID!==a)return;delete h[d];e.calls.forEach(function(a){a=a.onError;a&&a(c)});i[d]!=null&&p(b)}function p(a){__p&&__p();var c=a.getHash();if(j[c]!=null)return;j[c]=b("setImmediateAcrossTransitions")(function(){__p&&__p();delete j[c];var d=b("uniqueID")(),e=i[c];delete i[c];if(e==null)return;var f=m(e);f=a.getRequestParams(f);if(f.length<=0){e.forEach(function(a){a=a.onCompleted;a&&a()});return}var k=function(b){o(d,a,b.toError())},n=a.getStreamingTransportPageletName();n!=null||g(0,5708,a.getHash());n=b("UIPagelet").loadFromEndpoint(n,l(d),babelHelpers["extends"]({},f,{clientRequestID:d}),{allowIrrelevantRequests:!0,automatic:!1,crossPage:!0,errorHandler:k,jsNonblock:!0,transportErrorHandler:k,usePipe:!0,usePostRequest:!1});h[c]={calls:e,clientRequestID:d,endpointState:a,transport:n}})}a={makeRequest:function(a,b,c){__p&&__p();var d=c.onChunkResponse,e=c.onCompleted;c=c.onError;var f=a.getHash(),j={onChunkResponse:d,onCompleted:e,onError:c,targetNumToLoad:b};s(f);d=h[f];e=d?m(d.calls)>=b:!1;e?(d!=null||g(0,5709),d.calls.push(j)):(i[f]==null&&(i[f]=[]),i[f].push(j),d==null&&p(a));return{remove:function(){__p&&__p();var b=h[f];if(b!=null){var c=b.calls.indexOf(j);c!==-1&&b.calls.splice(c,1);b.calls.length===0&&(b.transport&&b.transport.abandon(),delete h[f],i[f]!=null&&p(a))}c=i[f];if(c!=null){b=c.indexOf(j);b!==-1&&c.splice(b,1)}}}},handleResponseChunk:function(a,b){a=q(a);a.forEach(function(a){a(b)})},handleRequestCompleted:function(a){var c=b("objectValues")(h).find(function(b){return b&&b.clientRequestID===a});c||g(0,5710,a);n(a,c.endpointState)},handleRequestFailed:function(a,c){__p&&__p();var d=b("objectValues")(h).find(function(b){return b&&b.clientRequestID===a});d||g(0,5711,a);var e=d.endpointState.getHash(),f=h[e];if(!f||f.clientRequestID!==a)return;delete h[e];f.calls.forEach(function(a){a=a.onError;a&&a(new Error(c))});i[e]!=null&&p(d.endpointState)}};function q(a){__p&&__p();var c=Object.keys(h).find(function(b){b=h[b];return b&&b.clientRequestID===a}),d=c&&h[c];if(!d||!c)return[];d=d.calls;c=i[c];c&&(d=d.concat(c));return b("compactArray")(d.map(function(a){return a.onChunkResponse}))}function r(a,c){__p&&__p();var d=c.targetNumToLoad,e=c.endpointState;c=c.payloadPromise;if(h[a]!=null)return;var f=b("uniqueID")();d={calls:[{targetNumToLoad:d}],clientRequestID:f,endpointState:e};h[a]=d;c||g(0,5712);a=c.then(function(a){var b=q(f);b.forEach(function(b){b(a)})});b("promiseDone")(a["finally"](function(){n(f,e)}))}c=function(a){b("ifRequired")("NotificationEagerLoader",function(b){b=b.eagerlyLoadedData;b.hasData&&a===b.endpointState.getHash()&&r(a,b)})};var s=b("once")(c);e.exports=a}),null);
__d("NotificationStore",["FBLogger","NotificationConstants","NotificationEndpointState","NotificationServerTransport","NotificationUpdates","createObjectBy","distinctArrayBy","flatMapArray","objectValues","sortBy"],(function(a,b,c,d,e,f){__p&&__p();var g={};function h(a,b,c){return a.page_info!=null&&b.classification==c.classification&&b.endpointControllerName===c.endpointControllerName&&b.readness==c.readness}function i(a,c){__p&&__p();var d={};b("NotificationEndpointState").getAllInstances(c.endpointControllerName).forEach(function(e){var f;h(a,e,c)?(e.graphQLPageInfo=a.page_info,f=a):a.payloadsource===b("NotificationConstants").PayloadSourceType.SYNC&&c.endpointControllerName==="WebNotificationsPayloadPagelet"&&e.classification==null&&e.readness==null?f=a:f=babelHelpers["extends"]({},a,{nodes:a.nodes?a.nodes.filter(function(a){return j(a,e)}):void 0});if(f.nodes&&f.nodes.length>0){f=k(e,f);f!=null&&f.forEach(function(a){d[a]=!0})}});b("NotificationUpdates").didUpdateNotifications(Object.keys(d))}function j(a,b){__p&&__p();var c=b.classification,d=b.notifications,e=b.order;b=b.readness;e=e.getAllResources();e.length===0;if(e.length>0){e=e[0];d=d.getResource(e);if(d.creation_time>=a.creation_time)return!1}if(c&&(!a.classifications||!a.classifications.includes(c)))return!1;return b&&b==="SEEN_AND_READ"!==(a.seen_state==="SEEN_AND_READ")?!1:!0}function k(a,b){var c=[],d={};b.nodes&&b.nodes.length>0&&b.nodes.forEach(function(b){var e=b.alert_id,f=a.notifications.getResource(e);(!f||f.creation_time<b.creation_time)&&(c.push(e),d[e]=b)});a.notifications.addResourcesAndExecute(d);a.order.addResources(c);return c}b("NotificationUpdates").subscribe("update-notifications",function(a,c){c.payloadsource!==b("NotificationConstants").PayloadSourceType.ENDPOINT&&i(c,{endpointControllerName:c.endpoint!=null?c.endpoint:"WebNotificationsPayloadPagelet"})});var l={getNotifications:function(a,c,d){__p&&__p();var e=b("NotificationEndpointState").getInstance(c),f=e.notifications,h=e.order,j,k=h.executeOrEnqueue(0,a,function(a){if(d){j=f.executeOrEnqueue(a,d);a=f.getUnavailableResources(j);a.length>0&&b("FBLogger")("notifications").warn("The range for this endpoint contained notification IDs for which we have no payload (Notification IDs: %s, Endpoint config: %s)",JSON.stringify(a),JSON.stringify(e.getConfig()))}});function m(){h.unsubscribe(k),j&&f.unsubscribe(j)}if(h.getUnavailableResources(k).length===0)return{remove:m};if(!l.canFetchMore(c)){h.forceRunCallbacks();return{remove:m}}var n=e.getHash();g[n]==null?g[n]=1:g[n]++;function o(){g[n]--,g[n]===0&&h.forceRunCallbacks()}function p(a){if(!(a&&a.nodes))return;b("NotificationUpdates").handleUpdate(b("NotificationConstants").PayloadSourceType.ENDPOINT,a,c.readness,c.classification);i(babelHelpers["extends"]({},a,{payloadsource:b("NotificationConstants").PayloadSourceType.ENDPOINT}),c)}function q(){g[n]--,g[n]===0&&h.forceRunCallbacks()}a=b("NotificationServerTransport").makeRequest(e,a,{onChunkResponse:p,onCompleted:q,onError:o});var r=a.remove;return{remove:function(){m(),r()}}},getNotification:function(a,c){c=c===void 0?{}:c;var d=c.classification,e=c.endpointControllerName;e=e===void 0?"WebNotificationsPayloadPagelet":e;c=c.readness;d=b("NotificationEndpointState").getInstance({classification:d,endpointControllerName:e,readness:c});e=d.notifications;return e.getResource(a)},getAllForAllEndpoints:function(){__p&&__p();var a=this,c=b("NotificationEndpointState").getAllInstances();c=b("flatMapArray")(c,function(c){c=a.getAll(c.getConfig());return b("objectValues")(c)});c=b("sortBy")(c,function(a){return a.creation_time});c=c.reverse();c=b("distinctArrayBy")(c,function(a){return a.alert_id});return b("createObjectBy")(c,function(a){return a.alert_id})},getAll:function(a){var c=b("NotificationEndpointState").getInstance(a),d=c.notifications;c=c.order;var e={};c.getAllResources().forEach(function(c){var f=d.getResource(c);f==null?b("FBLogger")("notifications").warn("The range for this endpoint contained a notification ID for which we have no payload (Notification ID: %s, Endpoint config: %s)",c,JSON.stringify(a)):e[c]=f});return e},getCount:function(a){a=b("NotificationEndpointState").getInstance(a);return a.order.getAllResources().length},canFetchMore:function(a){a=b("NotificationEndpointState").getInstance(a);a=a.graphQLPageInfo;return!a||!Object.prototype.hasOwnProperty.call(a,"has_next_page")||a.has_next_page},registerEndpoint:function(a){b("NotificationEndpointState").getInstance(a)}};l.registerEndpoint({endpointControllerName:"WebNotificationsPayloadPagelet"});e.exports=l}),null);
__d("NotificationBucketStore",["Arbiter","JSLogger","NotificationBucketStoreManager","NotificationConstants","NotificationGenericBucket","NotificationsBucketList","NotificationStore","NotificationUpdates"],(function(a,b,c,d,e,f){"use strict";__p&&__p();a=function(){__p&&__p();function a(a,b){this.$1=[],this.$2={},this.$3=[],this.$4={},this.$5=b,this.$6=!1,this.$7=null,this.$8={},this.$6=!1,this.$9(a),this.$10()}var c=a.prototype;c.$9=function(a){__p&&__p();var c=this;if(this.$6)return;this.$6=!0;b("NotificationsBucketList").buckets.forEach(function(a){a.bucket_type=a.bucket_type.toUpperCase();a=new(b("NotificationGenericBucket"))(a);c.$3.push(a);c.$4[a.getType()]=a});this.$11(a,this.$5,!1)};c.$12=function(){this.$1=this.$3.map(function(a){return{ids:a.getSortedIDs(),title:a.getTitle()||"",bucketType:a.getType()}})};c.$13=function(a,c,d){return!d&&a===b("NotificationBucketStoreManager").getSkipBucketType()&&!c};c.$14=function(a,b,c){__p&&__p();var d=a.alert_id;this.$2[d]||(this.$2[d]=[]);var e=[];for(var f=0,g=this.$3.length;f<g;f++){var h=this.$3[f];if(this.$13(h.getType(),b,c))continue;var i=h.insertIfEligible(a);e.push({bucket:h.getType(),result:i});if(i.eligible){this.$2[d].push(e);this.$15(a,h);return h.getType()}}this.$2[d].push(e);return null};c.$15=function(a,b){this.$8[a.alert_id]=b.getType()};c.$16=function(a,b){delete this.$8[a.alert_id]};c.$17=function(a,b,c){var d=a.alert_id,e=this.$18(d);if(e){e=this.$19(e);e&&(e.remove(d),this.$16(a,e))}this.$14(a,b,c)};c.$18=function(a){return this.$8[a]};c.$19=function(a){return this.$4[a]};c.$20=function(){this.$8={},this.$3.forEach(function(a){return a.reset()}),this.$2={},this.$12()};c.$11=function(a,c,d){var e=b("NotificationStore").getAll({endpointControllerName:a,classification:c});this.$21(Object.keys(e).map(function(a){return e[a]}),d||!1,!1)};c.$21=function(a,b,c){var d=this;a.forEach(function(a){return d.$17(a,b,c)});this.$12()};c.$10=function(){__p&&__p();var a=this;b("NotificationUpdates").subscribe("update-notifications",function(c,d){__p&&__p();c=d.nodes;if(c==null||c.length===0)return;if(d.payloadsource===b("NotificationConstants").PayloadSourceType.LIVE_SEND){var e=!1;c.forEach(function(b){e=b.classifications&&b.classifications.includes(a.$5)});if(!e&&a.$5!=null)return;a.$20();a.$11(d.endpoint||"WebNotificationsPayloadPagelet",a.$5,!0)}else b("NotificationBucketStoreManager").getActiveClassification()==a.$5&&(a.$7==null&&(a.$7=d.servertime),a.$21(c,!1,d.servertime===a.$7))});b("Arbiter").subscribe(b("JSLogger").DUMP_EVENT,function(b,c){c.notifs_bucket_data={bucketInfo:a.$1,data:a.$1.reduce(function(c,b){b=b.ids.map(function(b){return{id:b,results:a.$2[b]}});return[].concat(c,b)},[])}})};c.getBucketListData=function(){return this.$1};c.isEmptyBucketListData=function(){var a=this.$1.reduce(function(a,b){return a+b.ids.length},0);return a===0};return a}();e.exports=a}),null);
__d("NotificationBucketStoreManager",["NotificationBucketStore"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g={},h,i=null;a={getSkipBucketType:function(){return i},setSkipBucketType:function(a){i=a},setActiveClassification:function(a){h=a},getActiveClassification:function(){return h},getBucketStoreInstance:function(a,c){var d=c==null?"NO_CLASSIFICATION":c;g[d]==null&&(g[d]=new(b("NotificationBucketStore"))(a,c));return g[d]}};e.exports=a}),null);
__d("ReadToggle.react",["cx","Event","Keys","React","emptyFunction","joinClasses"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();a=b("React").PropTypes;c=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){var c,d;for(var e=arguments.length,f=new Array(e),g=0;g<e;g++)f[g]=arguments[g];return(c=d=a.call.apply(a,[this].concat(f))||this,d.$2=function(a){b("Event").getKeyCode(a)===b("Keys").RETURN&&d.props.onClick()},c)||babelHelpers.assertThisInitialized(d)}var d=c.prototype;d.render=function(){if(this.props.isRead)return b("React").jsx("div",{"aria-label":this.props.readLabel,className:this.$1(),"data-hover":"tooltip","data-testid":this.props.testid,"data-tooltip-alignh":"center","data-tooltip-content":this.props.readLabel,onClick:this.props.onClick,onKeyDown:this.$2,role:"button",tabIndex:0});else return b("React").jsx("div",{"aria-label":this.props.unreadLabel,className:this.$1(),"data-hover":"tooltip","data-testid":this.props.testid,"data-tooltip-alignh":"center","data-tooltip-content":this.props.unreadLabel,onClick:this.props.onClick,onKeyDown:this.$2,role:"button",tabIndex:"0"})};d.$1=function(){return b("joinClasses")(this.props.className,(this.props.isRead?"":"_5c9q")+(this.props.isRead?" _5c9_":""))};return c}(b("React").Component);c.propTypes={isRead:a.bool.isRequired,onClick:a.func.isRequired,readLabel:a.node.isRequired,unreadLabel:a.node.isRequired};c.defaultProps={onClick:b("emptyFunction")};e.exports=c}),null);
__d("QPLTimeToLoadHelper",["QuickPerformanceLogger","Run","crc32","performance"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g;function h(a){return a&&b("crc32")(a)||0}function i(a){return(g||(g=b("performance")))&&(g||(g=b("performance"))).timing&&(g||(g=b("performance"))).timing.navigationStart&&a<(g||(g=b("performance"))).timing.navigationStart?a+(g||(g=b("performance"))).timing.navigationStart:a}function j(a){a.time!=null&&(a.time=i(a.time));var c=h(a.instanceKey);b("QuickPerformanceLogger").markerStart(a.qplMarker,c,a.time);b("QuickPerformanceLogger").annotateMarkerString(a.qplMarker,"SOURCE",a.source,c);b("Run").onUnload(function(){return b("QuickPerformanceLogger").markerEnd(a.qplMarker,4,c)})}a={startWithClickEvent:function(a){j(babelHelpers["extends"]({},a,{time:a.event.timeStamp}))},start:function(a){j(a)},annotateString:function(a){var c=h(a.instanceKey);b("QuickPerformanceLogger").annotateMarkerString(a.qplMarker,a.annotationKey,a.annotationValue,c)},annotateInt:function(a){var c=h(a.instanceKey);b("QuickPerformanceLogger").annotateMarkerInt(a.qplMarker,a.annotationKey,a.annotationValue,c)},addPoint:function(a){var c=h(a.instanceKey);b("QuickPerformanceLogger").markerPoint(a.qplMarker,a.pointName,a.data,c,a.time!=null?i(a.time):a.time)},drop:function(a){b("QuickPerformanceLogger").markerDrop(a.qplMarker,h(a.instanceKey))},end:function(a){var c=a.time!=null?i(a.time):a.time,d=h(a.instanceKey);b("QuickPerformanceLogger").annotateMarkerInt(a.qplMarker,"END_TIME",c==null?-1:c,d);b("QuickPerformanceLogger").markerEnd(a.qplMarker,a.action?a.action:2,d,c)}};e.exports=a}),null);
__d("PermalinkTimeToLoadLogger",["Arbiter","BigPipe","PageEvents","QPLTimeToLoadHelper","performance"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g;function h(a){a.subscribe(b("PageEvents").AJAXPIPE_SEND,function(a,c){a=c.ts;b("QPLTimeToLoadHelper").start({source:"quickling",qplMarker:655597,time:a})})}function i(a){var c=(g||(g=b("performance"))).timing&&(g||(g=b("performance"))).timing.navigationStart;b("QPLTimeToLoadHelper").start({source:a,qplMarker:655597,time:c});c||b("QPLTimeToLoadHelper").annotateString({qplMarker:655597,annotationKey:"TIMESTAMP_ERROR",annotationValue:"NO_TTI"})}function j(a){b("Arbiter").subscribeOnce(b("BigPipe").Events.init,function(c,d){c=d.arbiter;a.requestType==="quickling"?h(c):i(a.requestType);c.subscribe(b("BigPipe").Events.tti,function(a,c){a=c.ts;b("QPLTimeToLoadHelper").end({qplMarker:655597,time:a})})})}a={log:function(a){j(a)}};e.exports=a}),null);