if (self.CavalryLogger) { CavalryLogger.start_js(["W+4fz"]); }

__d("getMentionableRect",["Rect"],(function(a,b,c,d,e,f){__p&&__p();function g(a){var b=document.selection.createRange().duplicate();b.moveStart("character",-a);return b.getBoundingClientRect()}function h(b){var c=a.getSelection();if(!c.rangeCount)return null;c=c.getRangeAt(0);c=c.cloneRange();var d=c.endContainer,e=c.endOffset,f=null;e>=b&&(c.setStart(d,e-b),f=c.getBoundingClientRect());return f}function c(a,c){a=document.selection?g(a):h(a);if(!a)return null;c=new(b("Rect"))(a.top,c?a.right:a.left,a.bottom,c?a.right:a.left,"viewport");return c.convertTo("document")}e.exports=c}),null);
__d("getMentionsSearchSource",["AtSignMentionsStrategy","CapitalizedNameMentionsStrategy","DocumentCompositeMentionsSource","DocumentMentionsSource","FilteredSearchSource","SearchSourceWithMetrics","WebAsyncSearchSource","filterCapitalizedNames"],(function(a,b,c,d,e,f){__p&&__p();function g(a){var c={bootstrapRequests:[],queryRequests:[],auxiliaryFields:{authorativePerson:"is_authoritative_person",connectedPage:"connected_page",disableAutosuggest:"disable_autosuggest",workForeignEntity:"is_work_foreign_entity",renderType:"render_type",verified:"is_verified",workUser:"is_work_user",indexRank:"index_rank"}};a._bootstrapEndpoints&&a._bootstrapEndpoints.forEach(function(a){c.bootstrapRequests.push({uri:a.endpoint,data:a.data})});a.bootstrapEndpoint&&c.bootstrapRequests.push({uri:a.bootstrapEndpoint,data:a.bootstrapData});a.queryEndpoint&&c.queryRequests.push({uri:a.queryEndpoint,data:a.queryData});return new(b("WebAsyncSearchSource"))(c)}function a(a,c){a=g(a);a=new(b("SearchSourceWithMetrics"))(a,c);c=new(b("FilteredSearchSource"))(b("filterCapitalizedNames"),a);a=[new(b("DocumentMentionsSource"))(b("AtSignMentionsStrategy"),a),new(b("DocumentMentionsSource"))(b("CapitalizedNameMentionsStrategy"),c)];return new(b("DocumentCompositeMentionsSource"))(a)}e.exports=a}),null);
__d("addDelightedTextInEditorState",["TextDelightInComposerController"],(function(a,b,c,d,e,f){"use strict";var g=b("TextDelightInComposerController").matcher;function a(a,b,c){c===void 0&&(c=!1);if(!g)return a;return c?g.matchLastWord(a,"","comment"):g.applyTextDelightEntity(a,b,"","comment")}e.exports=a}),null);
__d("addMentionableEntityToComposerState",["FBLogger","createMentionEntityForContentState","replaceMentionedTextInEditorState"],(function(a,b,c,d,e,f){"use strict";function a(a,c,d){var e=a.inputState;switch(e.__type){case"editor-state-based":c=b("replaceMentionedTextInEditorState")(c,e.editorState,d,b("createMentionEntityForContentState"));return babelHelpers["extends"]({},a,{inputState:babelHelpers["extends"]({},e,{editorState:c})});case"plain-text":default:b("FBLogger")("ufi2").warn("Unimplemented: tried to commit a searchable entity as a mention to a `UFI2ComposerState` with an `inputState` of type `%s`. Only the `editor-state-based` type is supported at the moment.",e.__type);return a}}e.exports=a}),null);
__d("getUpgradedUFI2DelightsComposerPlugin",["DraftEntity","EditorState","React","TextDelightConfig","TextDelightInComposerGating","TextDelightSpan.react","UnicodeUtils","addDelightedTextInEditorState","clearTimeout","getEntityMatcher","installUFI2ComposerInputDecorators","keyCommandBackspaceDelight","setTimeoutAcrossTransitions"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g,h=b("TextDelightConfig").composerOptions,i=null;function j(a,b,c){a.inputProps.onComposerStateChange(function(d){var e=babelHelpers["extends"]({},d,b(d)),f=e.inputState;switch(f.__type){case"editor-state-based":d=d.inputState;if(d.__type!=="editor-state-based")return e;k(a,f.editorState,d.editorState,c);return l(e,d.editorState,!1);default:return e}})}function k(a,c,d,e){__p&&__p();if(d.getCurrentContent()!==c.getCurrentContent()){i&&b("clearTimeout")(i);if(h.autoHighlightThresholdMs>0){var f=b("setTimeoutAcrossTransitions")(function(){a.inputProps.onComposerStateChange(function(a){return l(a,null,!0)})},h.autoHighlightThresholdMs);i=f;e.addSubscriptions({remove:function(){b("clearTimeout")(f)}})}}}function l(a,c,d){var e=a.inputState;if(e.__type!=="editor-state-based")return a;c=b("addDelightedTextInEditorState")(e.editorState,c||e.editorState,d);return c===e.editorState?a:babelHelpers["extends"]({},a,{inputState:babelHelpers["extends"]({},e,{editorState:c})})}function m(a,c,d){__p&&__p();if(!b("TextDelightInComposerGating").isBackspaceEnabled("comment"))return"not-handled";var e=a.composerState.inputState,f=null;if(e.__type!=="editor-state-based")return"not-handled";c==="backspace"&&(f=b("keyCommandBackspaceDelight")(d));if(!f)return"not-handled";a.onComposerStateChange(function(a){return babelHelpers["extends"]({},a,{inputState:babelHelpers["extends"]({},e,{editorState:f||e.editorState})})});return"handled"}function n(a,b,c){var d=m(a,b,c);return d==="not-handled"?a.handleKeyCommand?a.handleKeyCommand(b,c):"not-handled":d}function o(a){a.onComposerStateChange(function(a){var c=a.inputState;if(c.__type!=="editor-state-based")return a;var d=c.editorState;return babelHelpers["extends"]({},a,{inputState:babelHelpers["extends"]({},c,{editorState:b("EditorState").forceSelection(d,d.getSelection())})})})}function a(a){__p&&__p();return{installPlugin:function(a){__p&&__p();a.onInstallContentBlockToTextWithEntitiesInputMessageMappers(function(a,c,d){__p&&__p();var e=a.getEntityAt(c);if(e==null)return null;e=b("DraftEntity").get(e);if(e.getType()==="DELIGHT"){a=a.getText().slice(c,d);d=e.getData();e=d.campaignID;d=d.disabled;if(d)return null;if(/^\d+$/.test(e))return{message:{ranges:[{entity:{id:e},length:(g||(g=b("UnicodeUtils"))).strlen(a),offset:c}],text:a}}}return null})},render:function(c,d){var e=c.subscriptionsHandler,f=babelHelpers.objectWithoutPropertiesLoose(c,["subscriptionsHandler"]),g={component:b("TextDelightSpan.react"),props:babelHelpers["extends"]({},a,{triggerChange:function(){return o(f.inputProps)}}),strategy:b("getEntityMatcher")(function(a){return a.getType()==="DELIGHT"})};return b("React").jsx(d,babelHelpers["extends"]({},f,{inputProps:babelHelpers["extends"]({},f.inputProps,{composerState:b("installUFI2ComposerInputDecorators")(f.inputProps.composerState,g),handleKeyCommand:function(a,b){return n(c.inputProps,a,b)},onComposerStateChange:function(a){return j(c,a,e)}})}))}}}e.exports=a}),null);
__d("upgradedUFI2MentionsComposerPluginCreatorWithTypeaheadView",["ContextualLayer.react","ContextualLayerAutoFlip","ContextualLayerUpdateOnScroll","DataSource","DraftEntity","FBLogger","Keys","MentionSpan.react","MultiBootstrapDataSource","React","TypeaheadMetricReporter","TypeaheadNavigation","UnicodeUtils","WeakMentionSpan.react","addMentionableEntityToComposerState","getEntityMatcher","getMentionableRect","getMentionsSearchSource","installUFI2ComposerInputDecorators","shallowEqual"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g,h=(c=b("Keys")).DOWN,i=c.ESC,j=c.RETURN,k=c.TAB,l=c.UP,m={component:b("MentionSpan.react"),strategy:b("getEntityMatcher")(function(a){return a.getType()==="MENTION"})},n={component:b("WeakMentionSpan.react"),strategy:b("getEntityMatcher")(function(a){return a.getType()==="MENTION"?!!((a=a.getData())==null?void 0:a.isWeak):!1})};function o(a,c,d,e,f,g,h){__p&&__p();switch(g){case"MentionsAutocomplete/cancel":f&&f.layer&&f.layer.hide();return"handled";case"MentionsAutocomplete/next-mention":d(function(a){var c=null;b("TypeaheadNavigation").moveDown(a.mentionableEntities,a.selectedMentionableEntity,function(a){c=a});return{selectedMentionableEntity:c}});return"handled";case"MentionsAutocomplete/previous-mention":d(function(a){var c=null;b("TypeaheadNavigation").moveUp(a.mentionableEntities,a.selectedMentionableEntity,function(a){c=a});return{selectedMentionableEntity:c}});return"handled";case"MentionsAutocomplete/select-mention":if(c.selectedMentionableEntity!=null){p(a,c,c.selectedMentionableEntity,"return-key",e);return"handled"}break}return a.inputProps.handleKeyCommand?a.inputProps.handleKeyCommand(g,h):"not-handled"}function p(a,c,d,e,f){f.reportSelect(d.getUniqueID(),d.getType(),c.mentionableEntities.indexOf(d),e==="click"),f.sessionEnd(),f.sessionStart(),a.inputProps.onComposerStateChange(function(a){return b("addMentionableEntityToComposerState")(a,d,c.characterOffset)})}function q(a,b,c,d){__p&&__p();if(b.mentionableEntities.length&&c&&c.layer&&c.layer.isShown())switch(d.keyCode){case h:d.preventDefault();return"MentionsAutocomplete/next-mention";case i:d.preventDefault();return"MentionsAutocomplete/cancel";case j:if(b.selectedMentionableEntity){d.preventDefault();return"MentionsAutocomplete/select-mention"}break;case k:if(b.selectedMentionableEntity){d.preventDefault();return"MentionsAutocomplete/select-mention"}break;case l:d.preventDefault();return"MentionsAutocomplete/previous-mention"}return a.inputProps.keyBindingFn?a.inputProps.keyBindingFn(d):null}function r(a,c,d,e,f){__p&&__p();var g=d.inputProps.composerState.inputState;switch(g.__type){case"editor-state-based":var h=g.editorState.getCurrentContent();g=g.editorState.getSelection();if(g.getHasFocus()===!1){(e.characterOffset!==0||e.mentionableEntities.length!==0)&&f(function(a){return{characterOffset:0,mentionableEntities:[]}});return}d.subscriptionsHandler.addSubscriptions(c.search(h,g,function(c,d){__p&&__p();var g=a.typeaheadViewProps,h=d||0,i=c||[];g&&g.sortFn&&i.sort(g.sortFn);i=i.slice(0,a.maxResults);d=!b("shallowEqual")(e.mentionableEntities,i);if(i.length===0&&d===!1)return;var j=i.includes(e.selectedMentionableEntity)?e.selectedMentionableEntity:i[0];if(e.characterOffset===h&&e.selectedMentionableEntity===j&&d===!1)return;f(function(a){return{characterOffset:h,mentionableEntities:i,selectedMentionableEntity:j}})}));break;default:return}}function a(a){__p&&__p();return function(c,d){__p&&__p();var e,f,h=new(b("TypeaheadMetricReporter"))({event_name:"tinder_mentions"}),i=function(){__p&&__p();if(c.mentions_datasource_js_constructor_args_json){var a;try{a=JSON.parse(c.mentions_datasource_js_constructor_args_json),c.maxResults!=null&&(a[0]=babelHelpers["extends"]({},a[0],{maxResults:c.maxResults,queryData:a[0].queryData=babelHelpers["extends"]({},a[0].queryData,{max_result:c.maxResults})}))}catch(a){b("FBLogger")("ufi2").warn("Failed to parse `mentions_datasource_js_constructor_args_json` The mentions autocomplete is disabled for this user. (JSON string was: `%s`. Config was: `%s`.)",c.mentions_datasource_js_constructor_args_json,JSON.stringify(c))}if(a){var d=a[0];d=(d==null?void 0:(d=d.bootstrapEndpoints)==null?void 0:d.length)>0?new(Function.prototype.bind.apply(b("MultiBootstrapDataSource"),[null].concat(a)))():new(Function.prototype.bind.apply(b("DataSource"),[null].concat(a)))();return b("getMentionsSearchSource")(d,h)}}else b("FBLogger")("ufi2").warn("Failed to fetch `mentions_datasource_js_constructor_args_json` for use in the UFI composer mentions plugin. The mentions autocomplete is disabled for this user.");return b("getMentionsSearchSource")(new(b("DataSource"))({}),h)}();d&&i.bootstrap();function j(a){h.sessionEnd()}function k(a){i.bootstrap(),h.sessionStart()}return{componentDidUpdate:function(a,b,d,e,f){b=d.inputProps.composerState.inputState;a=a.inputProps.composerState.inputState;b.__type==="editor-state-based"&&(a.__type!=="editor-state-based"||a.__type==="editor-state-based"&&a.editorState!==b.editorState)&&r(c,i,d,e,f)},componentWillUnmount:function(){h.sessionDeactivate()},installPlugin:function(a){__p&&__p();a.onInstallContentBlockToTextWithEntitiesInputMessageMappers(function(a,c,d){__p&&__p();var e=a.getEntityAt(c);if(e==null)return null;e=b("DraftEntity").get(e);if(e.getType()==="MENTION"){a=a.getText().slice(c,d);d=e.getData();d=d.id;if(/^\d+$/.test(d)){var f;return{entitiesByID:(f={},f[d]=e,f),message:{ranges:[{entity:{id:d},entity_is_weak_reference:e.isWeak,length:(g||(g=b("UnicodeUtils"))).strlen(a),offset:c}],text:a}}}}return null})},render:function(d,g,i,l){__p&&__p();d.subscriptionsHandler;var r=babelHelpers.objectWithoutPropertiesLoose(d,["subscriptionsHandler"]);return b("React").jsxs(b("React").Fragment,{children:[b("React").jsx(l,babelHelpers["extends"]({},r,{inputProps:babelHelpers["extends"]({},r.inputProps,{composerState:b("installUFI2ComposerInputDecorators")(r.inputProps.composerState,n,m),handleKeyCommand:function(a,b){return o(d,g,i,h,f,a,b)},keyBindingFn:function(a){return q(d,g,f,a)},onBlur:function(a){r.inputProps.onBlur&&r.inputProps.onBlur(a);if(a.isDefaultPrevented())return;j(a)},onFocus:function(a){r.inputProps.onFocus&&r.inputProps.onFocus(a);if(a.isDefaultPrevented())return;k(a)},onInputRefUpdated:function(a){e=a,r.inputProps.onInputRefUpdated&&r.inputProps.onInputRefUpdated(a)}})})),g.mentionableEntities.length?b("React").jsx(b("ContextualLayer.react"),{behaviors:{ContextualLayerAutoFlip:b("ContextualLayerAutoFlip"),ContextualLayerUpdateOnScroll:b("ContextualLayerUpdateOnScroll")},contextBounds:b("getMentionableRect")(g.characterOffset,!1),contextRef:function(){return e},"data-testid":"UFI2ComposerMentionsPlugin/autocomplete",offsetX:0,offsetY:5,position:"below",ref:function(a){f=a},shown:!0,children:b("React").jsx(a,babelHelpers["extends"]({},c.typeaheadViewProps,{entries:g.mentionableEntities,highlightedEntry:g.selectedMentionableEntity,onHighlight:function(a){i({selectedMentionableEntity:a})},onSelect:function(a,b){b.preventDefault(),p(d,g,a,"click",h)}}))}):null]})}}}}e.exports=a}),null);
__d("createUpgradedUFI2GroupMentionsComposerPlugin",["GroupMentionsTypeaheadView.react","upgradedUFI2MentionsComposerPluginCreatorWithTypeaheadView"],(function(a,b,c,d,e,f){"use strict";e.exports=b("upgradedUFI2MentionsComposerPluginCreatorWithTypeaheadView")(b("GroupMentionsTypeaheadView.react"))}),null);
__d("createUpgradedUFI2MentionsComposerPlugin",["XUITypeaheadView.react","upgradedUFI2MentionsComposerPluginCreatorWithTypeaheadView"],(function(a,b,c,d,e,f){"use strict";e.exports=b("upgradedUFI2MentionsComposerPluginCreatorWithTypeaheadView")(b("XUITypeaheadView.react"))}),null);