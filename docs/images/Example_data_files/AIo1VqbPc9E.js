if (self.CavalryLogger) { CavalryLogger.start_js(["jA8xt"]); }

__d("StoriesCardOverlayPositioner_bounds.graphql",[],(function(a,b,c,d,e,f){"use strict";a={kind:"Fragment",name:"StoriesCardOverlayPositioner_bounds",type:"StoryOverlayRectangle",metadata:null,argumentDefinitions:[],selections:[{kind:"ScalarField",alias:null,name:"x",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"y",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"width",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"height",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"rotation",args:null,storageKey:null}]};e.exports=a}),null);
__d("StoriesCardOverlayResharedPost_overlay$normalization.graphql",[],(function(a,b,c,d,e,f){"use strict";a={kind:"SplitOperation",name:"StoriesCardOverlayResharedPost_overlay$normalization",metadata:{derivedFrom:"StoriesCardOverlayResharedPost_overlay"},selections:[{kind:"LinkedField",alias:null,name:"attached_story",storageKey:null,args:null,concreteType:"Story",plural:!1,selections:[{kind:"ScalarField",alias:null,name:"id",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"url",args:null,storageKey:null}]},{kind:"ScalarField",alias:null,name:"action_title",args:null,storageKey:null},{kind:"LinkedField",alias:null,name:"bounds",storageKey:null,args:null,concreteType:"StoryOverlayRectangle",plural:!1,selections:[{kind:"ScalarField",alias:null,name:"x",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"y",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"width",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"height",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"rotation",args:null,storageKey:null}]}]};e.exports=a}),null);
__d("StoriesCardOverlayResharedPost_overlay.graphql",[],(function(a,b,c,d,e,f){"use strict";a={kind:"Fragment",name:"StoriesCardOverlayResharedPost_overlay",type:"StoryOverlayResharedPost",metadata:null,argumentDefinitions:[],selections:[{kind:"LinkedField",alias:null,name:"attached_story",storageKey:null,args:null,concreteType:"Story",plural:!1,selections:[{kind:"ScalarField",alias:null,name:"id",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"url",args:null,storageKey:null}]},{kind:"ScalarField",alias:null,name:"action_title",args:null,storageKey:null},{kind:"LinkedField",alias:null,name:"bounds",storageKey:null,args:null,concreteType:"StoryOverlayRectangle",plural:!1,selections:[{kind:"FragmentSpread",name:"StoriesCardOverlayPositioner_bounds",args:null}]}]};e.exports=a}),null);
__d("StoriesTagSticker_overlay$normalization.graphql",[],(function(a,b,c,d,e,f){"use strict";a={kind:"SplitOperation",name:"StoriesTagSticker_overlay$normalization",metadata:{derivedFrom:"StoriesTagSticker_overlay"},selections:[{kind:"ScalarField",alias:null,name:"profile_action_link",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"type",args:null,storageKey:null},{kind:"LinkedField",alias:null,name:"bounds",storageKey:null,args:null,concreteType:"StoryOverlayRectangle",plural:!1,selections:[{kind:"ScalarField",alias:null,name:"x",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"y",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"width",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"height",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"rotation",args:null,storageKey:null}]}]};e.exports=a}),null);
__d("StoriesTagSticker_overlay.graphql",[],(function(a,b,c,d,e,f){"use strict";a={kind:"Fragment",name:"StoriesTagSticker_overlay",type:"StoryOverlayTagSticker",metadata:null,argumentDefinitions:[],selections:[{kind:"ScalarField",alias:null,name:"profile_action_link",args:null,storageKey:null},{kind:"ScalarField",alias:null,name:"type",args:null,storageKey:null},{kind:"LinkedField",alias:null,name:"bounds",storageKey:null,args:null,concreteType:"StoryOverlayRectangle",plural:!1,selections:[{kind:"FragmentSpread",name:"StoriesCardOverlayPositioner_bounds",args:null}]}]};e.exports=a}),null);
__d("StoriesActorContext",["React"],(function(a,b,c,d,e,f){"use strict";a={actorID:null};e.exports=b("React").createContext(a)}),null);
__d("StoriesEnums",[],(function(a,b,c,d,e,f){"use strict";a={BACKGROUND_STYLE:{BLACK:"black",DEFAULT:"default"},EMPTY_BUCKET_TYPES:{FRIEND_BUCKET:"FRIEND_BUCKET",SELF_BUCKET:"SELF_BUCKET",UNSELECTED:"UNSELECTED"},END_CARD:{AVATAR_WIDTH:44,COUNT_DOWN_SECONDS:5,MAX_AVATAR_NUMBER:4,MAX_TILE_NUMBER:6,MS_PER_INTERVAL:200,MS_PER_SECOND:1e3,PADDING_HORIZONTAL:12,PADDING_VERTICAL:16,PADDING_WIDTH:32},GRADIENT_DIRECTION:{BL_TR:"BL_TR",BOTTOM_TOP:"BOTTOM_TOP",BR_TL:"BR_TL",LEFT_RIGHT:"LEFT_RIGHT",RIGHT_LEFT:"RIGHT_LEFT",TL_BR:"TL_BR",TOP_BOTTOM:"TOP_BOTTOM",TR_BL:"TR_BL"},LIGHTWEIGHT_REACTION_UNICODES:{ANGER:"\ud83d\ude21",HAHA:"\ud83d\ude06",LIKE:"\ud83d\udc4d",LOVE:"\u2764\ufe0f",SORRY:"\ud83d\ude22",WOW:"\ud83d\ude2e"},NAV_DIRECTIONS:{NEXT_BUCKET:"next-bucket",NEXT_CARD:"next-card",PREV_BUCKET:"prev-bucket",PREV_CARD:"prev-card",STAY_HERE:"stay_here"},POLL_STICKER:{AVAILABLE_WIDTH_RATIO:(240-2*12)/240,BACKGROUND_COLOR_BLUE:"#79A6FF",BACOGROUND_COLOR_SLATE:"#EAEFF2",CHILD_PADDING_RATIO:.05,DEFAULT_OPTION_WIDTH_RATIO:.5,FIVE_OPTION_STAR_RATING:"FIVE_OPTION_STAR_RATING",IG_LEFT_TEXT_COLOR:"#13bda6",IG_RIGHT_TEXT_COLOR:"#F36B7F",IG_TWO_OPTION_COMBINED:"IG_TWO_OPTION_COMBINED",MINIMUM_LABEL_WIDTH_RATIO:.31,OVERFLOW_CHILD_PADDING_RATIO:20/240,TEXT_BIG_HEIGHT_RATIO:40/72,TEXT_BIG_MAX_FONT_SIZE_RATIO:30/240,TEXT_BIG_MIN_FONT_SIZE_RATIO:20/240,TEXT_BIG_RATIO:40/240,TEXT_COLOR_GREY:"#5F6673",TEXT_ONLY_FONT_SIZE_RATIO:20/240,TEXT_SMALL_HEIGHT_RATIO:25/72,TEXT_SMALL_MAX_FONT_SIZE_RATIO:20/240,TEXT_SMALL_MIN_FONT_SIZE_RATIO:12/240,TEXT_SMALL_RATIO:25/240,TEXT_WITH_PERCENT_FONT_SIZE_RATIO:12/240,TWO_OPTION_COMBINED:"TWO_OPTION_COMBINED",VOTE_COUNT_FONT_SIZE_RATIO:32/240,VOTE_OPTION_MAX_WIDTH_RATIO:.75},PRONOUN:{FEMALE:"FEMALE",MALE:"MALE",NEUTRAL:"NEUTRAL"},RECTANGULAR_TILE_TYPES:{THREE_COLUMN:"THREE_COLUMN",TWO_COLUMN:"TWO_COLUMN"},STORIES_BUCKETS_INDEXER_TYPES:{OTHER:"OTHER",OWNED_SELF:"OWNED_SELF"},STORIES_OPTION_TYPES:{DELETE:"DELETE",DELETE_STORY_AND_BLOCK_MEMBER:"DELETE_STORY_AND_BLOCK_MEMBER",MUTE_CARD_OWNER:"MUTE_CARD_OWNER",REPORT_TO_GROUP_ADMINS:"REPORT_TO_GROUP_ADMINS"},STORY_CARD_TYPES:{ARCHIVED_STORY:"ARCHIVED_STORY",BIRTHDAY_STORY:"BIRTHDAY_STORY",CREW_STORY:"CREW_STORY",EVENT_STORY:"EVENT_STORY",GOODWILL_GENERATED_STORY:"GOODWILL_GENERATED_STORY",GOODWILL_STORY:"GOODWILL_STORY",GROUP_STORY:"GROUP_STORY",HIGHLIGHTED_STORY:"HIGHLIGHTED_STORY",LIVE_STORY:"LIVE_STORY",M_GROUP_STORY:"M_GROUP_STORY",NULL_STATE_STORY:"NULL_STATE_STORY",PAGE_STORY:"PAGE_STORY",PROFILE_PLUS_STORY:"PROFILE_PLUS_STORY",PROMOTION_STORY:"PROMOTION_STORY",SHARED_PAGE_STORY:"SHARED_PAGE_STORY",STORY:"STORY",TOPIC_STORY:"TOPIC_STORY",UNKNOWN:"UNKNOWN",WAS_LIVE:"WAS_LIVE"},STORY_MEDIA_TYPES:{PHOTO:"Photo",VIDEO:"Video"},STORY_OVERLAY_TYPES:{EXTERNAL_SONG:"StoryOverlayExternalSong",INTERACTIVE_STICKER:"StoryOverlayReactionSticker",POLL_STICKER:"StoryOverlayPollSticker",RESHARED_CONTENT:"StoryOverlayResharedContent",RESHARED_POST:"StoryOverlayResharedPost",TAG_STICKER:"StoryOverlayTagSticker"},STORY_REACTION_STICKERS_CONSTANTS:{ANIMATIONS_INTERVAL_THROTTLE:300,HOLD_DELAY:1e3},TAG_STICKER_TYPES:{LOCATION:"LOCATION",PAGE:"PAGE",PEOPLE:"PEOPLE",PRODUCT:"PRODUCT"},VIEWER_THEME:{DARK_MODE:"dark_mode",DARK15:"dark_15",DARK30:"dark_30",DEFAULT:"default"},VIEWERSHEET_STYLE:{DEFAULT:"default",SLIDE_UP:"slide_up"},VOTING_PHASES:{JUST_VOTED:"JUST_VOTED",RESULTS:"RESULTS",VOTING:"VOTING"}};e.exports=a}),null);
__d("StoriesRelayInternal",["requireCond","cr:1013932"],(function(a,b,c,d,e,f){"use strict";e.exports=b("cr:1013932")}),null);
__d("storiesCreateFragmentContainer",["requireCond","cr:987224"],(function(a,b,c,d,e,f){"use strict";e.exports=b("cr:987224").createFragmentContainer}),null);
__d("storiesCreatePaginationContainer",["requireCond","cr:779698"],(function(a,b,c,d,e,f){"use strict";e.exports=b("cr:779698").createPaginationContainer}),null);
__d("StoriesRelay",["requireCond","CometRelay","StoriesRelayInternal","cr:1150430","storiesCreateFragmentContainer","storiesCreatePaginationContainer"],(function(a,b,c,d,e,f){"use strict";c=(a=b("CometRelay")).fetchQuery;d=a.useBlockingPaginationFragment;f=a.useFragment;var g=a.useLazyLoadQuery,h=a.useLegacyPaginationFragment,i=a.usePreloadedQuery,j=a.useRefetchableFragment,k=a.useRelayEnvironment;e.exports={MatchContainer:a.MatchContainer,commitLocalUpdate:(e=b("StoriesRelayInternal")).commitLocalUpdate,commitMutation:e.commitMutation,createFragmentContainer:b("storiesCreateFragmentContainer"),createPaginationContainer:b("storiesCreatePaginationContainer"),environment:b("cr:1150430"),fetchQuery:c,graphql:e.graphql,readInlineData:e.readInlineData,requestSubscription:e.requestSubscription,useBlockingPaginationFragment:d,useFragment:f,useLazyLoadQuery:g,useLegacyPaginationFragment:h,usePaginationFragment:h,usePreloadedQuery:i,useRefetchableFragment:j,useRelayEnvironment:k}}),null);
__d("StoriesCardOverlayPositioner.react",["React","StoriesRelay","StoriesCardOverlayPositioner_bounds.graphql"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g;c=b("StoriesRelay").createFragmentContainer;b("StoriesRelay").graphql;function a(a){__p&&__p();var c=a.bounds,d=a.children,e=a.containerHeight;a=a.containerWidth;if(c!=null){var f=c.height,g=c.rotation,h=c.width,i=c.x;c=c.y;if(typeof f==="number"&&typeof h==="number"&&typeof g==="number"&&typeof i==="number"&&typeof c==="number"){if(c>1||i>1)return null;f=f*e;e=h*a;h=c*100;a=i*100;c={height:f+"px",left:a+"%",position:"absolute",top:h+"%",width:e+"px"};i={height:"100%",transform:"rotate("+g+"deg)",width:"100%"};return b("React").jsx("div",{className:"storiesCardOverlay/root",style:c,children:b("React").jsx("div",{className:"storiesCardOverlay/rotation",style:i,children:d})})}}return null}e.exports=c(a,{bounds:g!==void 0?g:g=b("StoriesCardOverlayPositioner_bounds.graphql")})}),null);
__d("StoriesPauseReasons",[],(function(a,b,c,d,e,f){"use strict";a={BUCKET_TRANSITION:"BUCKET_TRANSITION",BUG_DIALOG:"BUG_DIALOG",CARD_CHANGE:"CARD_CHANGE",CLICK_ADD_STORY:"CLICK_ADD_STORY",CLICK_ARCHIVE_RESHARE_BUTTON:"CLICK_ARCHIVE_RESHARE_BUTTON",CLICK_GIF_SELECTOR:"CLICK_GIF_SELECTOR",CLICK_ON_OVERLAY_STICKER:"CLICK_ON_OVERLAY_STICKER",CLICK_PAUSE_ICON:"CLICK_PAUSE_ICON",CLICK_SEE_MORE_LONG_TEXT:"CLICK_SEE_MORE_LONG_TEXT",CONFIRMATION_DIALOG:"CONFIRMATION_DIALOG",FOCUSE_ON_INPUT:"FOCUSE_ON_INPUT",HOVER_ON_ARCHIVE_RESHARE_BUTTON:"HOVER_ON_ARCHIVE_RESHARE_BUTTON",HOVER_ON_OVERLAY_STICKER:"HOVER_ON_OVERLAY_STICKER",HOVER_ON_PAUSE_OVERLAY:"HOVER_ON_PAUSE_OVERLAY",HOVER_ON_RATING_STICKER:"HOVER_ON_RATING_STICKER",HOVER_ON_REACTION_ICON:"HOVER_ON_REACTION_ICON",HOVER_ON_SHARE_BUTTON:"HOVER_ON_SHARE_BUTTON",HOVER_ON_SONG_STICKER:"HOVER_ON_SONG_STICKER",JEWEL:"JEWEL",KEYBOARD:"KEYBOARD",LWR_PLAYBACK:"LWR_PLAYBACK",MENU:"MENU",MESSAGE_VIEWER:"MESSAGE_VIEWER",MOUSE_CLICK_AND_HOLD:"MOUSE_CLICK_AND_HOLD",MOUSE_ENTER_POLL_STICKER:"MOUSE_ENTER_POLL_STICKER",REPORT_DIALOG:"REPORT_DIALOG",SETTINGS_DIALOG:"SETTINGS_DIALOG",TILE_GRID_BUTTON:"TILE_GRID_BUTTON",VIDEO_PLAYBACK:"VIDEO_PLAYBACK",VISIBILITY_CHANGE:"VISIBILITY_CHANGE"};e.exports=a}),null);
__d("StoriesCardOverlayResharedPost.react",["Banzai","CometLink.react","CometTooltip.react","React","StoriesCardOverlayPositioner.react","StoriesPauseReasons","StoriesRelay","stylex","StoriesCardOverlayResharedPost_overlay.graphql"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g;a=b("StoriesRelay").createFragmentContainer;b("StoriesRelay").graphql;c=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){__p&&__p();var c,d;for(var e=arguments.length,f=new Array(e),g=0;g<e;g++)f[g]=arguments[g];return(c=d=a.call.apply(a,[this].concat(f))||this,d.$1=function(a){__p&&__p();a=d.props;var c=a.cardID;a=a.overlay;a=a.attached_story;if(a){a=a.id;if(typeof a==="string"){a={event_name:"reshared_post_tap",reshared_post_graphql_id:a,thread_graphql_id:c};b("Banzai").post("stories_reshares_consumption",a)}}},d.$2=function(a){d.props.setPause(!0,b("StoriesPauseReasons").HOVER_ON_OVERLAY_STICKER)},d.$3=function(a){d.props.setPause(!1,b("StoriesPauseReasons").HOVER_ON_OVERLAY_STICKER)},c)||babelHelpers.assertThisInitialized(d)}var d=c.prototype;d.render=function(){__p&&__p();var a=this.props,c=a.containerHeight,d=a.containerWidth;a=a.overlay;var e=a.action_title,f=a.attached_story;a=a.bounds;if(f&&e!==""&&a){f=f.url;if(typeof f==="string"){var g;return(g=b("React")).jsx(b("StoriesCardOverlayPositioner.react"),{bounds:a,containerHeight:c,containerWidth:d,children:g.jsx(b("CometTooltip.react"),{align:"middle",position:"above",tooltip:e,children:g.jsx(b("CometLink.react"),{href:f,onClick:this.$1,target:"_blank",children:g.jsx("div",{className:"k4urcfbm pmk7jnqg mrt03zmi datstx6m","data-testid":"overlay_reshared_post",onMouseEnter:this.$2,onMouseLeave:this.$3})})})},"reshare_"+f)}}return null};return c}(b("React").PureComponent);e.exports=a(c,{overlay:g!==void 0?g:g=b("StoriesCardOverlayResharedPost_overlay.graphql")})}),null);
__d("StoriesTagSticker.react",["fbt","CometLink.react","CometTooltip.react","React","StoriesCardOverlayPositioner.react","StoriesEnums","StoriesPauseReasons","StoriesRelay","stylex","StoriesTagSticker_overlay.graphql"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();var h,i=b("StoriesEnums").TAG_STICKER_TYPES;a=b("StoriesRelay").createFragmentContainer;b("StoriesRelay").graphql;c=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){var c,d;for(var e=arguments.length,f=new Array(e),g=0;g<e;g++)f[g]=arguments[g];return(c=d=a.call.apply(a,[this].concat(f))||this,d.$1=function(a){d.props.setPause(!0,b("StoriesPauseReasons").HOVER_ON_OVERLAY_STICKER)},d.$2=function(a){d.props.setPause(!1,b("StoriesPauseReasons").HOVER_ON_OVERLAY_STICKER)},c)||babelHelpers.assertThisInitialized(d)}var d=c.prototype;d.render=function(){__p&&__p();var a=this.props,c=a.containerHeight,d=a.containerWidth;a=a.overlay;var e=a==null?void 0:a.type,f=a==null?void 0:a.profile_action_link,h=a==null?void 0:a.bounds;if(a==null||e==null||f==null||h==null)return null;a=[i.LOCATION,i.PAGE,i.PEOPLE,i.PRODUCT];if(!a.includes(e))return null;a=function(){switch(e){case i.LOCATION:return g._("See Location");case i.PAGE:return g._("See Profile");case i.PEOPLE:return g._("See Profile");case i.PRODUCT:return g._("See Product");default:return}};var j=function(){switch(e){case i.LOCATION:return g._("Link");case i.PAGE:return g._("Link");case i.PEOPLE:return g._("Link");case i.PRODUCT:return g._("Link");default:return}};a=a();j=j();return b("React").jsx(b("StoriesCardOverlayPositioner.react"),{bounds:h,containerHeight:c,containerWidth:d,children:b("React").jsx(b("CometTooltip.react"),{align:"middle",position:"above",tooltip:a,children:b("React").jsx(b("CometLink.react"),{"aria-label":j,href:f,target:"_blank",children:b("React").jsx("div",{className:"k4urcfbm fi2e5rcv pmk7jnqg oqq733wu datstx6m nhd2j8a9",onMouseEnter:this.$1,onMouseLeave:this.$2})})})})};return c}(b("React").PureComponent);e.exports=a(c,{overlay:h!==void 0?h:h=b("StoriesTagSticker_overlay.graphql")})}),null);
__d("StoriesUniqueID",[],(function(a,b,c,d,e,f){"use strict";var g="js_",h=36,i=0;function a(){return g+(i++).toString(h)}e.exports=a}),null);
__d("StoriesRelayBlue",["requireCond","cr:1150431"],(function(a,b,c,d,e,f){"use strict";e.exports=b("cr:1150431")}),null);
__d("storiesBlueCreateFragmentContainer",["requireCond","cr:1150432"],(function(a,b,c,d,e,f){"use strict";e.exports={createFragmentContainer:b("cr:1150432").createFragmentContainer}}),null);
__d("storiesExperimentalRelayBridge",["invariant","CometRelay","React"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();c=b("CometRelay").createSuspenseFragmentContainer_DEPRECATED;var h=b("CometRelay").createSuspensePaginationContainer_DEPRECATED,i=b("React").useMemo;function a(a,c,d){__p&&__p();var e=b("React").forwardRef(function(c,d){var e=c.relay;c=babelHelpers.objectWithoutPropertiesLoose(c,["relay"]);var f=i(function(){return{environment:e.environment,hasMore:function(){return e.hasMore},isLoading:function(){return e.isLoading},loadMore:function(a,b,c){return e.loadMore(a,{force:c&&c.force,onComplete:b})},refetchConnection:function(a,b,c){return e.refetchConnection(a,(a=c)!=null?a:{},{onComplete:b})}}},[e]);return b("React").jsx(a,babelHelpers["extends"]({},c,{ref:d,relay:f}))}),f=h(e,c,{getFragmentRefsFromResponse:function(a){var b=Object.keys(a);b.length===1||g(0,12733,b.length);return a[b[0]]},getVariables:d.getVariables,query:d.query});e=function(a,c){return b("React").jsx(f,babelHelpers["extends"]({},a,{ref:c}))};e.displayName=f.displayName;c=b("React").forwardRef(e);return c}e.exports={createFragmentContainer:c,createPaginationContainer:a}}),null);