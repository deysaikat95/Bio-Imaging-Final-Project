if (self.CavalryLogger) { CavalryLogger.start_js(["HAwk5"]); }

__d("FacepileRoundedProfile.react",["cx","fbt","CurrentUser","HovercardLink","Image.react","Link.react","React","Tooltip.react","URI","joinClasses"],(function(a,b,c,d,e,f,g,h){"use strict";__p&&__p();var i;a=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){return a.apply(this,arguments)||this}var d=c.prototype;d.render=function(){__p&&__p();var a,c=this.props,d=c.borderColor,e=c.borderWidth,f=c.getCustomActivationLink,g=c.getCustomHovercardLink,i=c.hoverBehavior,j=c.imageSize,k=c.isClickable;c=c.profile;var l=this.props.style,m=c.glyph_size,n=c.image_src,o=c.entity_id;l=babelHelpers["extends"]({},l,{borderColor:d,borderWidth:e,height:j+"px",width:j+"px"});d={};m&&m<j&&(d.margin=(j-m)/2+"px");e="_4mnq";c.className&&(e=b("joinClasses")(e,c.className));a=b("CurrentUser").isWorkUser()&&((a=c.work_foreign_entity_info)==null?void 0:a.type)==="FOREIGN"?b("React").jsx("div",{className:"_7cf0"}):null;e=b("React").jsxs(b("React").Fragment,{children:[b("React").jsx("div",{className:e,style:l,children:b("React").jsx(b("Image.react"),{className:"_1h_6",height:m||j,src:n,style:d,width:m||j})}),a]});i==="name"&&c.name&&(e=b("React").jsx(b("Tooltip.react"),{className:"_4mns",tooltip:c.name,children:e}));l=i==="hovercard";if(o&&(k||l)){n={};l&&(n["data-hovercard"]=g(o));k&&(n.href=f(o));e=b("React").jsx(b("Link.react"),babelHelpers["extends"]({"aria-label":h._("Profile of {name}",[h._param("name",c.name)])},n,{children:e}))}return e};return c}(b("React").Component);a.defaultProps={getCustomActivationLink:function(a){return new(i||(i=b("URI")))("/"+a)},getCustomHovercardLink:function(a){return b("HovercardLink").constructEndpoint({id:a})},hoverBehavior:"none",imageSize:32,isClickable:!1};e.exports=a}),null);
__d("FacepileRoundedCount.react",["cx","fbt","ix","FacepileRoundedProfile.react","Image.react","Link.react","React","Tooltip.react"],(function(a,b,c,d,e,f,g,h,i){"use strict";__p&&__p();a=b("React").PropTypes;var j=.3438;c=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){return a.apply(this,arguments)||this}var d=c.prototype;d.render=function(){__p&&__p();var a=this.props,c=a.backgroundColor,d=a.borderColor,e=a.borderWidth,f=a.color,g=a.count,k=a.fontSize,l=a.profiles,m=a.size,n=a.style,o=a.use,p=a.shouldHideCountToolTip,q=a.image;a=a.href;var r=l.length;g=g||r;e={backgroundColor:c,borderColor:d,borderWidth:e,color:f,borderRadius:m,fontSize:(c=k)!=null?c:m*j+"px",height:m,minWidth:m};f=b("React").jsx("span",{className:"_4mnq _34n6",style:e,children:b("React").jsx("span",{className:"_40vg",children:"+"+g})});if(o==="image"){k=null;switch(m){case 16:k=i("409177");break;case 20:k=i("409178");break;case 28:k=i("409179");break;case 32:k=i("409180");break;case 48:k=i("409181");break}k&&(f=b("React").jsx("div",{className:"_4mnq",style:e,children:b("React").jsx(b("Image.react"),{className:"_1h_6",src:q?q:k})}))}else if(o==="face"&&r>0){c={left:"50%",marginLeft:"-"+m/2+"px"};f=b("React").jsxs("div",{className:"_ric",style:{borderColor:d},children:[b("React").jsx(b("FacepileRoundedProfile.react"),{borderColor:d,imageSize:m,profile:l[0],style:c}),f]})}e=g-r;q=l.map(function(a){return a.name}).filter(Boolean);k=r>0?q.join("\n")+"\n":"";if(e>0){o=h._({"*":"and {count} others.","_1":"and {count} other."},[h._param("count",e),h._plural(e)]);k+=o.toString()}d=b("React").jsx("div",{style:{whiteSpace:"pre-wrap"},children:k});a!=null&&(f=b("React").jsxs(b("Link.react"),{href:a,children:[b("React").jsx("span",{className:"accessible_elem",children:k}),f]}));return b("React").jsx(b("Tooltip.react"),{className:"_4mns",style:n,tooltip:p?null:d,children:f})};return c}(b("React").Component);c.defaultProps={size:32,use:"count",image:null};c.propTypes={backgroundColor:a.string,borderColor:a.string,borderWidth:a.number,color:a.string,className:a.string,count:a.number,fontSize:a.number,use:a.oneOf(["count","face","image"]),image:a.any,profiles:a.arrayOf(a.shape({className:a.string,entity_id:a.string,glyph_size:a.number,image_src:a.any.isRequired,name:a.string})).isRequired,size:a.number,style:a.object,shouldHideCountToolTip:a.bool};e.exports=c}),null);
__d("FacepileRounded.react",["cx","FacepileRoundedCount.react","FacepileRoundedProfile.react","React","joinClasses"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();var h=.3125;a=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){return a.apply(this,arguments)||this}var d=c.prototype;d.render=function(){__p&&__p();var a=this.props,c=a.borderColor,d=a.borderWidth,e=a.direction,f=a.getCustomActivationLink,g=a.getCustomHovercardLink,i=a.hoverBehavior,j=a.imageSize,k=a.isClickable,l=a.message,m=a.numProfilesToDisplay,n=a.overflowBackgroundColor,o=a.overflowFontSize,p=a.overflowTextColor,q=a.profiles,r=a.remainingProfilesCount,s=a.remainingProfilesCountStyle,t=a.remainingProfilesCountImage,u=a.remainingProfilesCountHref,v=a.spacing;a=a.shouldHideCountToolTip;var w=e==="descending";e=q.length;var x=null,y=this.props.className;y=b("joinClasses")("_4mnv"+(w?" _4wh8":""),y);l&&(x=b("React").jsx("div",{className:"_4mnt",children:l}));l=q;var z=null;m&&m<e&&(l=q.slice(0,m),z=q.slice(m,e));var A={getCustomActivationLink:f,getCustomHovercardLink:g,hoverBehavior:i,imageSize:j,isClickable:k},B=(v!=null?v:-(j*h))+"px";m=l.map(function(a,e){var f={zIndex:w?q.length-e:e};e>0&&(f.marginLeft=B);return b("React").jsx(b("FacepileRoundedProfile.react"),babelHelpers["extends"]({borderColor:c,profile:a,style:f,borderWidth:d},A),e)});f=null;z&&(f=b("React").jsx(b("FacepileRoundedCount.react"),{backgroundColor:n,borderColor:c,color:p,fontSize:o,count:r,profiles:z,size:j,shouldHideCountToolTip:a,style:{marginLeft:B,zIndex:w?0:e},use:s,image:t,href:u,borderWidth:d}));return b("React").jsxs("div",{className:y,children:[b("React").jsxs("div",{className:"_4mnw",children:[m,f]}),x]})};return c}(b("React").Component);a.defaultProps={direction:"ascending",hoverBehavior:"none",imageSize:32,isClickable:!1,isOverlapDisabled:!1,remainingProfilesCountStyle:"count"};e.exports=a}),null);
__d("BUISwitch.react",["cx","fbt","BUIComponent","Event","Keys","React"],(function(a,b,c,d,e,f,g,h){"use strict";__p&&__p();a=b("React").PropTypes;c=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){__p&&__p();var c,d;for(var e=arguments.length,f=new Array(e),g=0;g<e;g++)f[g]=arguments[g];return(c=d=a.call.apply(a,[this].concat(f))||this,d.$BUISwitch1=function(a){if(d.props.disabled)return;d.props.onToggle&&d.props.onToggle(!d.props.value);d.props.preventEventBubbling&&a.stopPropagation()},d.$BUISwitch2=function(a){if(d.props.disabled)return;var c=b("Event").getKeyCode(a);(c===b("Keys").RETURN||c===b("Keys").SPACE)&&(a.preventDefault(),d.props.onToggle&&d.props.onToggle(!d.props.value),d.props.preventEventBubbling&&a.stopPropagation())},c)||babelHelpers.assertThisInitialized(d)}var d=c.prototype;d.render=function(){return b("React").jsxs("div",babelHelpers["extends"]({},this.props,{className:"_128j"+(this.props.value?" _128k":"")+(this.props.value?"":" _128l")+(this.props.disabled?" _128m":"")+(this.props.animate?" _128n":"")+(this.props.inline?" _3n6a":""),children:[b("React").jsx("div",{"aria-checked":this.props.value?"true":"false",className:"_128o",onClick:this.$BUISwitch1,onKeyDown:this.$BUISwitch2,onMouseDown:this.$BUISwitch3,role:"checkbox",tabIndex:this.props.disabled?"-1":"0",children:b("React").jsx("div",{className:"_128p"})}),this.$BUISwitch4()]}))};d.$BUISwitch4=function(){return!this.props.showLabel?null:b("React").jsx("span",{className:"_128q",children:this.props.value?h._("ON"):h._("OFF")})};d.$BUISwitch3=function(a){a.preventDefault()};return c}(b("BUIComponent"));c.propTypes={animate:a.bool.isRequired,disabled:a.bool,onToggle:a.func,showLabel:a.bool,value:a.bool.isRequired,preventEventBubbling:a.bool,inline:a.bool};c.defaultProps={animate:!0};e.exports=c}),null);
__d("TextInputControl",["DOMControl","Event","Input","debounce"],(function(a,b,c,d,e,f){__p&&__p();a=function(a){"use strict";__p&&__p();babelHelpers.inheritsLoose(c,a);function c(c){c=a.call(this,c)||this;var d=c.getRoot(),e=b("debounce")(c.update.bind(babelHelpers.assertThisInitialized(c)),0);b("Event").listen(d,{input:e,keydown:e,paste:e});return c}var d=c.prototype;d.setMaxLength=function(a){b("Input").setMaxLength(this.getRoot(),a);return this};d.getValue=function(){return b("Input").getValue(this.getRoot())};d.isEmpty=function(){return b("Input").isEmpty(this.getRoot())};d.setValue=function(a){b("Input").setValue(this.getRoot(),a);this.update();return this};d.clear=function(){return this.setValue("")};d.setPlaceholderText=function(a){b("Input").setPlaceholder(this.getRoot(),a);return this};return c}(b("DOMControl"));e.exports=a}),null);
__d("transferTextStyles",["Style"],(function(a,b,c,d,e,f){var g={fontFamily:null,fontSize:null,fontStyle:null,fontWeight:null,lineHeight:null,wordWrap:null};function a(a,c){for(var d in g)Object.prototype.hasOwnProperty.call(g,d)&&(g[d]=b("Style").get(a,d));b("Style").apply(c,g)}e.exports=a}),null);
__d("TextMetrics",["DOM","Style","UserAgent","transferTextStyles"],(function(a,b,c,d,e,f){__p&&__p();function g(a){var c=a.clientWidth,d=b("Style").get(a,"-moz-box-sizing")=="border-box";if(d&&b("UserAgent").isBrowser("Firefox < 29"))return c;d=b("Style").getFloat(a,"paddingLeft")+b("Style").getFloat(a,"paddingRight");return c-d}a=function(){"use strict";__p&&__p();function a(a,c){this.$1=a;this.$2=!!c;c="textarea";var d="textMetrics";this.$2&&(c="div",d+=" textMetricsInline");this.$3=b("DOM").create(c,{className:d});b("transferTextStyles")(a,this.$3);document.body.appendChild(this.$3)}var c=a.prototype;c.measure=function(a){var c=this.$1,d=this.$3;a=(a||c.value)+"...";if(!this.$2){var e=g(c);b("Style").set(d,"width",Math.max(e,0)+"px")}c.nodeName==="TEXTAREA"?d.value=a:b("DOM").setContent(d,a);return{width:d.scrollWidth,height:d.scrollHeight}};c.destroy=function(){b("DOM").remove(this.$3)};return a}();e.exports=a}),null);
__d("TextAreaControl",["Arbiter","ArbiterMixin","CSS","DOMControl","Event","Style","TextInputControl","TextMetrics","classWithMixins","mixin"],(function(a,b,c,d,e,f){__p&&__p();function g(a,c){return b("Style").getFloat(a,c)||0}a=function(a){"use strict";__p&&__p();babelHelpers.inheritsLoose(c,a);function c(c){var d;d=a.call(this,c)||this;d.autogrow=b("CSS").hasClass(c,"uiTextareaAutogrow");d.autogrowWithPlaceholder=b("CSS").hasClass(c,"uiTextareaAutogrowWithPlaceholder");d.width=null;b("Event").listen(c,"focus",d._handleFocus.bind(babelHelpers.assertThisInitialized(d)));return d}var d=c.prototype;d.setAutogrow=function(a){this.autogrow=a;return this};d.onupdate=function(){a.prototype.onupdate.call(this),this.updateHeight()};d.updateHeight=function(){__p&&__p();if(this.autogrow){var a=this.getRoot();this.metrics||(this.metrics=new(b("TextMetrics"))(a));typeof this.initialHeight==="undefined"&&(this.isBorderBox=b("Style").get(a,"box-sizing")==="border-box"||b("Style").get(a,"-moz-box-sizing")==="border-box"||b("Style").get(a,"-webkit-box-sizing")==="border-box",this.borderBoxOffset=g(a,"padding-top")+g(a,"padding-bottom")+g(a,"border-top-width")+g(a,"border-bottom-width"),this.initialHeight=a.offsetHeight-this.borderBoxOffset);var c;(!a.value||a.value.length===0)&&this.autogrowWithPlaceholder?c=this.metrics.measure(a.placeholder):c=this.metrics.measure();c=Math.max(this.initialHeight,c.height);this.isBorderBox&&(c+=this.borderBoxOffset);this.maxHeight&&c>this.maxHeight&&(c=this.maxHeight,b("Arbiter").inform("maxHeightExceeded",{textArea:a}));c!==this.height&&(this.height=c,b("Style").set(a,"height",c+"px"),b("Arbiter").inform("reflow"),this.inform("resize"))}else this.metrics&&(this.metrics.destroy(),this.metrics=null)};d.resetHeight=function(){this.height=-1,this.update()};d.setMaxHeight=function(a){this.maxHeight=a};d.setAutogrowWithPlaceholder=function(a){this.autogrowWithPlacedholder=a};d._handleFocus=function(){this.width=null};c.getInstance=function(a){return b("DOMControl").getInstance(a)||new c(a)};return c}(b("classWithMixins")(b("TextInputControl"),b("mixin")(b("ArbiterMixin"))));e.exports=a}),null);
__d("AbstractTextArea.react",["cx","AbstractTextField.react","React","TextAreaControl"],(function(a,b,c,d,e,f,g){__p&&__p();a=b("React").Component;c=b("React").PropTypes;d=function(a){"use strict";__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){return a.apply(this,arguments)||this}var d=c.prototype;d.componentDidUpdate=function(){this.$2&&this.$2.onupdate()};d.componentWillUnmount=function(){this.$2=null};d.render=function(){var a=this;return b("React").jsx(b("AbstractTextField.react"),babelHelpers["extends"]({},this.props,{children:b("React").jsx("textarea",{className:"_58an",onClick:this.props.onClick,onMouseDown:this.props.onMouseDown,onKeyUp:this.props.onKeyUp,rows:this.props.rows,tabIndex:this.props.tabIndex,ref:function(b){a.$1=b,a.$3()}})}))};d.$3=function(){if(this.$1&&this.props.autoGrow&&!this.$2){var a=new(b("TextAreaControl"))(this.$1);a.setAutogrow(!0);a.onupdate();this.$2=a}};d.focusInput=function(){this.$1&&this.$1.focus()};d.blurInput=function(){this.$1&&this.$1.blur()};d.getTextFieldDOM=function(){return this.$1};d.getValue=function(){return this.$1?this.$1.value:""};return c}(a);d.propTypes=babelHelpers["extends"]({},b("AbstractTextField.react").propTypes,{autoGrow:c.bool});e.exports=d}),null);
__d("XUIMenuSeparator.react",["MenuSeparator.react"],(function(a,b,c,d,e,f){a=b("MenuSeparator.react");e.exports=a}),null);
__d("PopoverMenuOverlappingBorder",["cx","CSS","DOM","Style","shield"],(function(a,b,c,d,e,f,g){__p&&__p();a=function(){"use strict";__p&&__p();function a(a){this._popoverMenu=a,this._popover=a.getPopover(),this._triggerElem=a.getTriggerElem()}var c=a.prototype;c.enable=function(){this._setMenuSubscription=this._popoverMenu.subscribe("setMenu",b("shield")(this._onSetMenu,this))};c.disable=function(){this._popoverMenu.unsubscribe(this._setMenuSubscription),this._setMenuSubscription=null,this._removeBorderSubscriptions(),this._removeShortBorder()};c._onSetMenu=function(){this._removeBorderSubscriptions();this._menu=this._popoverMenu.getMenu();this._renderShortBorder(this._menu.getRoot());this._showSubscription=this._popover.subscribe("show",b("shield")(this._updateBorder,this));var a=this._updateBorder.bind(this);this._menuSubscription=this._menu.subscribe(["change","resize"],function(){setTimeout(a,0)});this._updateBorder()};c._updateBorder=function(){var a=this._menu.getRoot(),c=this._triggerElem.offsetWidth;a=Math.max(a.offsetWidth-c,0);b("Style").set(this._shortBorder,"width",a+"px")};c._renderShortBorder=function(a){this._shortBorder=b("DOM").create("div",{className:"_54hx"}),b("DOM").appendContent(a,this._shortBorder),b("CSS").addClass(a,"_54hy")};c._removeShortBorder=function(){this._shortBorder&&(b("DOM").remove(this._shortBorder),this._shortBorder=null,b("CSS").removeClass(this._popoverMenu.getMenu().getRoot(),"_54hy"))};c._removeBorderSubscriptions=function(){this._showSubscription&&(this._popover.unsubscribe(this._showSubscription),this._showSubscription=null),this._menuSubscription&&(this._menu.unsubscribe(this._menuSubscription),this._menuSubscription=null)};return a}();Object.assign(a.prototype,{_shortBorder:null,_setMenuSubscription:null,_showSubscription:null,_menuSubscription:null});e.exports=a}),null);
__d("PageContentTabSuccessDialog.react",["cx","ix","Image.react","React","XUIDialog.react","XUIDialogBody.react"],(function(a,b,c,d,e,f,g,h){__p&&__p();a=b("React").PropTypes;c=function(a){"use strict";babelHelpers.inheritsLoose(c,a);function c(){return a.apply(this,arguments)||this}var d=c.prototype;d.render=function(){var a;return(a=b("React")).jsx(b("XUIDialog.react"),{width:344,shown:!0,layerHideOnBlur:!1,layerFadeOnShow:!0,children:a.jsxs(b("XUIDialogBody.react"),{className:"_--l",children:[a.jsx(b("Image.react"),{src:h("101769"),className:"_--n"}),a.jsx("div",{className:"_--o",children:this.props.successLabel})]})})};return c}(b("React").Component);c.propTypes={successLabel:a.node};e.exports=c}),null);
__d("PageContentTabLoadingDialog",["cx","PageContentTabSuccessDialog.react","PageContentTabSuccessDialogTimer","React","ReactDOM","WaitTimeArea.react","XUIDialog.react","XUIDialogBody.react","XUISpinner.react"],(function(a,b,c,d,e,f,g){__p&&__p();var h=b("PageContentTabSuccessDialogTimer").TIME_IN_MS,i=function(c){"use strict";babelHelpers.inheritsLoose(a,c);function a(){return c.apply(this,arguments)||this}var d=a.prototype;d.render=function(){var a;return(a=b("React")).jsx(b("WaitTimeArea.react"),{name:"PageContentTabLoadingDialog",owner:"pages_publishing",children:a.jsx(b("XUIDialog.react"),{width:300,shown:!0,layerHideOnBlur:!1,children:a.jsx(b("XUIDialogBody.react"),{className:"_5xp9",children:a.jsx(b("XUISpinner.react"),{background:"light",className:"_5xpe",size:"large"})})})})};return a}(b("React").Component);a={show:function(){this._container||(this._container=document.createElement("div")),b("ReactDOM").render(b("React").jsx(i,{}),this._container)},hide:function(){if(!this._container)return;this.destroy()},hideWithSuccessMessage:function(a,c){if(!this._container)return;b("ReactDOM").render(b("React").jsx(b("PageContentTabSuccessDialog.react"),{successLabel:a}),this._container);setTimeout(this.destroy.bind(this),c?c:h)},destroy:function(){b("ReactDOM").unmountComponentAtNode(this._container),this._container=null}};e.exports=a}),null);
__d("FBStorySavedState",[],(function(a,b,c,d,e,f){"use strict";var g={ARCHIVED:"ARCHIVED",NOT_SAVABLE:"NOT_SAVABLE",NOT_SAVED:"NOT_SAVED",SAVED:"SAVED",isMutableState:function(a){return a===g.ARCHIVED||a===g.SAVED||a===g.NOT_SAVED}};e.exports=g}),null);
__d("divisorSignedModulo",[],(function(a,b,c,d,e,f){function a(a,b){return(a%b+b)%b}e.exports=a}),null);
__d("wrapIndex",["divisorSignedModulo"],(function(a,b,c,d,e,f){"use strict";function a(a,c){return b("divisorSignedModulo")(a,c)}e.exports=a}),null);
__d("SUIActionMenu.react",["cx","Link.react","Locale","React","RTLKeys","SUIComponent","SUIFocusUtil","SUISubActionMenuLayer.react","SUITheme","cxMargin","wrapIndex"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();var h=1,i=4;a=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){__p&&__p();var c,d;for(var e=arguments.length,f=new Array(e),g=0;g<e;g++)f[g]=arguments[g];return(c=d=a.call.apply(a,[this].concat(f))||this,d.state={items:d.props.items,highlightedIndex:null,activeSubmenuIndex:null,flattenedEntries:q(d.props.items)},d.$SUIActionMenu1=null,d.$SUIActionMenu2=new Map(),d.$SUIActionMenu5=function(a){var b=d.state.flattenedEntries;b=b[a];b&&p(b.item)&&(b.item.onClick&&b.item.onClick(),d.props.onItemClick&&d.props.onItemClick())},d.$SUIActionMenu8=function(a){var b=d.state.flattenedEntries;b=b[a];b&&p(b.item)?d.$SUIActionMenu4(a):d.$SUIActionMenu4(null)},d.$SUIActionMenu9=function(a,b,c){d.$SUIActionMenu4(a,b),d.props.onSubmenuHide&&d.props.onSubmenuHide(c)},d.$SUIActionMenu10=function(a){__p&&__p();var c=d.state,e=c.activeSubmenuIndex;c=c.highlightedIndex;var f=d.state.flattenedEntries,g=0,h=b("RTLKeys").getRight(),i=a.target instanceof HTMLAnchorElement&&a.target.href!=null;switch(a.keyCode){case b("RTLKeys").RETURN:case b("RTLKeys").SPACE:i||a.preventDefault();if(c!==null){i=f[c].item;i.type==="submenu"?d.setState({activeSubmenuIndex:c}):d.$SUIActionMenu5(c)}break;case b("RTLKeys").UP:a.preventDefault();g=-1;break;case b("RTLKeys").DOWN:a.preventDefault();g=1;break;case h:a.preventDefault();if(c!==null&&e!==c){i=f[c].item;i.type==="submenu"&&d.setState({activeSubmenuIndex:c})}break}if(g===0)return;if(!f.some(function(a){a=a.item;return p(a)}))return;h=d.state.highlightedIndex!==null?b("wrapIndex")(d.state.highlightedIndex+g,f.length):g===1?0:f.length-1;while(!p(f[h].item))h=b("wrapIndex")(h+g,f.length);d.$SUIActionMenu4(h)},c)||babelHelpers.assertThisInitialized(d)}c.getDerivedStateFromProps=function(a,b){return a.items!==b.items?{flattenedEntries:q(a.items),items:a.items}:null};var d=c.prototype;d.componentDidMount=function(){var a=this;this.props.focusFirstElementOnMount&&(this.$SUIActionMenu3(),this.$SUIActionMenu1=window.setTimeout(function(){a.$SUIActionMenu4(0,!1)},0))};d.componentWillUnmount=function(){this.$SUIActionMenu3()};d.$SUIActionMenu4=function(a,b){b===void 0&&(b=!1),this.$SUIActionMenu6(a),this.$SUIActionMenu7(a,b)};d.$SUIActionMenu6=function(a){a=a!==null?this.$SUIActionMenu2.get(a):null;a!=null&&a.focus()};d.$SUIActionMenu7=function(a,b){b=b?a:null;this.setState({highlightedIndex:a,activeSubmenuIndex:b})};d.$SUIActionMenu3=function(){this.$SUIActionMenu1&&(window.clearTimeout(this.$SUIActionMenu1),this.$SUIActionMenu1=null)};d.render=function(){__p&&__p();var a=this,c=b("SUITheme").get(this).SUIActionMenu,d=this.state.flattenedEntries,e=d.some(function(a){return a.item.type==="submenu"}),f=d.some(function(a){return(a.item.type==="item"||a.item.type==="link"||a.item.type==="submenu")&&a.item.icon!==void 0});return b("React").jsx("ul",{className:"_2pi2 _6ff6",id:this.props.id,onKeyDown:this.$SUIActionMenu10,role:"menu",style:{backgroundColor:c.backgroundColor,borderColor:c.borderColor,borderRadius:c.borderRadius,borderWidth:c.borderWidth,paddingLeft:c.horizontalPadding,paddingRight:c.horizontalPadding},tabIndex:"0",children:d.map(function(d,g){__p&&__p();var h=d.item;d=d.isIndented;switch(h.type){case"item":case"link":var i={"data-testid":h["data-testid"],description:h.description,hasIconSibling:f,hasSubmenuSibling:e,icon:h.icon,index:g,isDisabled:!!h.isDisabled,isHighlighted:g===a.state.highlightedIndex,isIndented:d,key:g,label:h.label,onClick:a.$SUIActionMenu5,onMouseEnter:a.$SUIActionMenu8,ref:function(b){return b&&a.$SUIActionMenu2.set(g,b)},rightContent:h.rightContent,uniform:c};return h.type==="item"?b("React").jsx(j,babelHelpers["extends"]({},i)):b("React").jsx(j,babelHelpers["extends"]({},i,{href:h.href,openLinkInNewTab:h.openLinkInNewTab,rel:h.rel}));case"submenu":return b("React").jsx(k,{behaviors:h.behaviors?h.behaviors:{},hasIconSibling:f,icon:h.icon,index:g,isDisabled:!!h.isDisabled,isHighlighted:g===a.state.highlightedIndex,isIndented:d,isSubmenuOpen:g===a.state.activeSubmenuIndex,items:h.items,label:h.label,onItemClick:a.$SUIActionMenu5,onSubmenuVisibilityChange:a.$SUIActionMenu9,onToggle:h.onToggle,position:h.position?h.position:"right",ref:function(b){return b&&a.$SUIActionMenu2.set(g,b)},uniform:c},g);case"separator":return b("React").jsx(n,{isIndented:d,uniform:c},g);case"group":return b("React").jsx(o,{hasSubmenuSibling:e,isFirstItem:g===0,label:h.label,rightContent:h.rightContent,uniform:c},g);default:break}return null})})};return c}(b("SUIComponent"));a.defaultProps={focusFirstElementOnMount:!1};var j=function(c){__p&&__p();babelHelpers.inheritsLoose(a,c);function a(){var a,d;for(var e=arguments.length,f=new Array(e),g=0;g<e;g++)f[g]=arguments[g];return(a=d=c.call.apply(c,[this].concat(f))||this,d.$1=b("React").createRef(),d.$2=function(){d.props.onClick(d.props.index)},d.$3=function(){d.props.onMouseEnter(d.props.index)},a)||babelHelpers.assertThisInitialized(d)}var d=a.prototype;d.focus=function(){var a=this.$1.current;a instanceof HTMLElement&&b("SUIFocusUtil").setFocus(a)};d.render=function(){__p&&__p();var a=this.props,c=a.description,d=a.hasIconSibling,e=a.hasSubmenuSibling,f=a.href,g=a.icon,h=a.isDisabled,i=a.isHighlighted,j=a.isIndented,k=a.label,n=a.openLinkInNewTab,o=a.rel,p=a.rightContent;a=a.uniform;i=l({isDisabled:h,isHighlighted:i,uniform:a});var q=p!=null;q=!q&&e;e=f!=null&&f!=="";var r="_8l9y";j="_6ff7"+(j?" _6ff8":"")+(q?" _6ff9":"");q=b("React").jsx(m,{description:c,hasIconSibling:d,icon:g,isDisabled:h,label:k,rightContent:p,uniform:a});c={"data-testid":this.props["data-testid"],onClick:this.$2,onMouseEnter:this.$3,role:"menuitem",tabIndex:0};return e&&!h?b("React").jsx("li",{className:r,role:"presentation",children:b("React").jsx(b("Link.react"),babelHelpers["extends"]({},c,{className:j,href:f,linkRef:this.$1,rel:o,style:i,target:n===!0?"_blank":"_self",children:q}))}):b("React").jsx("li",babelHelpers["extends"]({},c,{className:r,ref:this.$1,children:b("React").jsx("div",{className:j,style:i,children:q})}))};return a}((c=b("React")).PureComponent),k=function(c){__p&&__p();babelHelpers.inheritsLoose(a,c);function a(){var a,b;for(var d=arguments.length,e=new Array(d),f=0;f<d;f++)e[f]=arguments[f];return(a=b=c.call.apply(c,[this].concat(e))||this,b.state={itemRef:null},b.$1=function(a){b.setState({itemRef:a})},b.$2=function(){b.props.onItemClick(b.props.index)},b.$3=function(){b.props.onSubmenuVisibilityChange(b.props.index,!0)},b.$4=function(a){b.props.onSubmenuVisibilityChange(b.props.index,!1,a)},b.$5=function(){return b.state.itemRef},a)||babelHelpers.assertThisInitialized(b)}var d=a.prototype;d.componentDidUpdate=function(a){a.isSubmenuOpen!==this.props.isSubmenuOpen&&this.props.onToggle&&this.props.onToggle(this.props.isSubmenuOpen)};d.componentWillUnmount=function(){this.props.isSubmenuOpen&&this.props.onToggle&&this.props.onToggle(!1)};d.focus=function(){this.state.itemRef!==null&&b("SUIFocusUtil").setFocus(this.state.itemRef)};d.render=function(){__p&&__p();var a=this.props,c=a.behaviors;c=c===void 0?{}:c;var d=a.isDisabled,e=a.isSubmenuOpen,f=a.items,g=a.position;g=g===void 0?"right":g;var j=a.uniform,k=a.icon;a=a.hasIconSibling;var m=b("Locale").isRTL(),n=l({isDisabled:this.props.isDisabled,isHighlighted:this.props.isHighlighted,uniform:j}),o={width:j.itemIconWidth};m=m?j.submenuIndicatorRTL:j.submenuIndicatorLTR;var p={};d&&(p.opacity=.5);return b("React").jsx("li",{className:"_8l9y",onClick:this.$3,onMouseEnter:this.$3,ref:this.$1,role:"menuitem",tabIndex:0,children:b("React").jsxs("div",{className:"_6ff7 _6ffc"+(this.props.isIndented?" _6ff8":""),style:n,children:[a?b("React").jsxs("div",{className:"_2pi3 _6vpg"+(j.itemIconAlignment==="center"?" _85sa":""),children:[k&&b("React").jsx("div",{className:"_6vph",style:o,children:k}),this.props.label]}):this.props.label,b("React").jsx("div",{className:"_6ffd",style:p,children:m}),d||this.$5()==null?null:b("React").jsx(b("SUISubActionMenuLayer.react"),{behaviors:c,getContextNode:this.$5,isVisible:e,items:f,offsetY:-1*i-h,onClose:this.$4,onItemClick:this.$2,position:g,uniform:j})]})})};return a}(c.PureComponent);function l(a){var b=a.isDisabled,c=a.isHighlighted;a=a.uniform;var d=babelHelpers["extends"]({},a.itemTypeStyle,{borderRadius:a.itemBorderRadius,color:a.itemColor,minHeight:a.itemHeight});b?(d.cursor="default",d.color=a.disabledItemColor,d.userSelect="none"):c&&(d.color=a.highlightedItemColor,d.backgroundColor=a.highlightedItemBackgroundColor);return d}var m=function(c){__p&&__p();babelHelpers.inheritsLoose(a,c);function a(){return c.apply(this,arguments)||this}var d=a.prototype;d.render=function(){var a=this.props,c=a.description,d=a.hasIconSibling,e=a.icon,f=a.isDisabled,g=a.label,h=a.rightContent;a=a.uniform;var i=h!=null,j=c!=null,k=e!=null,l=typeof a.itemIconWidth==="number",m={width:a.itemIconWidth};return b("React").jsxs(b("React").Fragment,{children:[j||d?b("React").jsxs("div",{className:"_2pi3 _6vpg"+(a.itemIconAlignment==="center"?" _85sa":""),children:[k||d&&l?b("React").jsx("div",{className:"_6vph",style:m,children:e}):null,b("React").jsxs("div",{children:[g,j?b("React").jsx("div",{style:babelHelpers["extends"]({},a.descriptionTypeStyle,f?{color:a.disabledItemColor}:{}),children:c}):null]})]}):g,i?b("React").jsx("span",{className:"_3-9a",children:h}):null]})};return a}(c.PureComponent),n=function(c){babelHelpers.inheritsLoose(a,c);function a(){return c.apply(this,arguments)||this}var d=a.prototype;d.render=function(){return b("React").jsx("li",{className:"_6ffg"+(this.props.isIndented?" _6ffh":""),style:{borderColor:this.props.uniform.borderColor}})};return a}(c.PureComponent),o=function(c){__p&&__p();babelHelpers.inheritsLoose(a,c);function a(){return c.apply(this,arguments)||this}var d=a.prototype;d.render=function(){var a=this.props,c=a.rightContent;a=a.uniform;c=c!=null;c=!c&&this.props.hasSubmenuSibling;return b("React").jsx("li",{className:"_8lau",children:b("React").jsxs("div",{className:"_6ffi"+(c?" _6ff9":""),style:babelHelpers["extends"]({},a.headerTypeStyle,{color:a.headerColor,minHeight:a.itemHeight}),children:[this.props.label,this.props.rightContent]})})};return a}(c.PureComponent);function p(a){return(a.type==="item"||a.type==="link"||a.type==="submenu")&&!a.isDisabled}function q(a){__p&&__p();var b=[];a.forEach(function(c,d){__p&&__p();switch(c.type){case"item":b.push({isIndented:!1,item:c});break;case"link":b.push({isIndented:!1,item:c});break;case"separator":b.push({isIndented:!1,item:c});break;case"submenu":b.push({isIndented:!1,item:c});break;case"group":var e=d===0;e=!e;e&&b.push({isIndented:!1,item:{type:"separator"}});b.push({isIndented:!1,item:c});c.items.forEach(function(a){b.push({isIndented:!0,item:a})});e=d===a.length-1;d=a[d+1];d=d&&(d.type==="group"||d.type==="separator");e=!e&&!d;e&&b.push({isIndented:!1,item:{type:"separator"}});break;default:c.type}});return b}e.exports=a}),null);
__d("SUIContextMenuLayerBehaviors",["ContextualLayerAutoFlip","ContextualLayerHideOnScroll","LayerAutoFocus","LayerFitHeightToScreen","LayerHideOnBlur","LayerHideOnEscape"],(function(a,b,c,d,e,f){"use strict";a={ContextualLayerAutoFlip:b("ContextualLayerAutoFlip"),ContextualLayerHideOnScroll:b("ContextualLayerHideOnScroll"),LayerAutoFocus:b("LayerAutoFocus"),LayerFitHeightToScreen:b("LayerFitHeightToScreen"),LayerHideOnBlur:b("LayerHideOnBlur"),LayerHideOnEscape:b("LayerHideOnEscape")};e.exports=a}),null);
__d("SUISubActionMenuLayer.react",["cssVar","ContextualLayer.react","LayerHideSources","React","RTLKeys","SUIActionMenu.react","SUIContextMenuLayerBehaviors"],(function(a,b,c,d,e,f,g){"use strict";__p&&__p();a=function(a){__p&&__p();babelHelpers.inheritsLoose(c,a);function c(){var c,d;for(var e=arguments.length,f=new Array(e),g=0;g<e;g++)f[g]=arguments[g];return(c=d=a.call.apply(a,[this].concat(f))||this,d.$1=function(a){var c=b("RTLKeys").getLeft();a.keyCode===c&&(a.preventDefault(),d.props.onClose())},d.$2=function(a){(a===b("LayerHideSources").BLUR||a===b("LayerHideSources").ESCAPE)&&d.props.onClose(a)},c)||babelHelpers.assertThisInitialized(d)}var d=c.prototype;d.render=function(){__p&&__p();var a=this.props,c=a.behaviors;c=c===void 0?{}:c;var d=a.getContextNode,e=a.isVisible,f=a.onItemClick,g=a.offsetY;g=g===void 0?0:g;var h=a.position;h=h===void 0?"right":h;a=a.uniform;if(!e)return null;a={borderRadius:a==null?void 0:a.borderRadius,boxShadow:(e=a==null?void 0:a.boxShadow)!=null?e:"0 1px 10px rgba(0, 0, 0, 0.2)",minWidth:160};return b("React").jsx(b("ContextualLayer.react"),{alignment:"left",behaviors:babelHelpers["extends"]({},b("SUIContextMenuLayerBehaviors"),c),contextRef:d,includeHideSource:!0,isStrictlyControlled:!0,offsetY:g,onHide:this.props.onClose,position:h,shouldSetARIAProperties:!0,shown:!0,children:b("React").jsx("div",{onKeyDown:this.$1,style:a,children:b("React").jsx(b("SUIActionMenu.react"),{focusFirstElementOnMount:!0,items:this.props.items,onItemClick:f,onSubmenuHide:this.$2})})})};return c}(b("React").PureComponent);e.exports=a}),null);
__d("PECurrency",["PECurrencyConfig","intlNumUtils"],(function(a,b,c,d,e,f){"use strict";__p&&__p();var g=b("PECurrencyConfig").currency_map_for_cc,h=b("PECurrencyConfig").currency_map_for_render,i=100;function j(a){var b=0;a=a;while(a>1)b++,a/=10;return b}function k(a,b,c){var d=h[a].symbol,e=h[a].format||"{symbol}{amount}";c===!0&&d!=a&&(e.indexOf("{symbol}")>=e.indexOf("{amount}")?e+=" ("+a+") ":e+=" "+a);return e.replace("{symbol}",d).replace("{amount}",String(b))}function a(a,c,d){d=babelHelpers["extends"]({showCurrencyCode:!1,showDecimals:!0,showSymbol:!0,stripZeros:!1,thousandSeparator:!1},d);var e=o(a)||0;c=c/i;e=d.showDecimals?j(e):0;d.stripZeros||(c=b("intlNumUtils").formatNumber(c,e));d.thousandSeparator&&(typeof c==="string"&&(c=b("intlNumUtils").parseNumber(c)),c=b("intlNumUtils").formatNumberWithThousandDelimiters(Number(c),e));!d.showSymbol?e=d.showCurrencyCode?c+" "+a:String(c):(typeof c==="number"&&(c=""+c),e=k(a,c,d.showCurrencyCode));return e}function c(a,b,c,d,e){b=l(a,b,!0,c,d,e);switch(a){case"AUD":return"A"+b;case"CAD":return"C"+b;case"HKD":return"HK"+b;case"SGD":return"S"+b;case"COP":return"COP"+b;default:return b}}function l(a,c,d,e,f,g){__p&&__p();d=d!=null?d:!0;e=e!=null?e:!1;f=f!=null?f:!1;g=g!=null?g:!1;var h=o(a)||0,l=Math.abs(c)/i;h=j(h);f||(l=b("intlNumUtils").formatNumber(l,h));g&&(typeof l==="string"&&(l=b("intlNumUtils").parseNumber(l)),l=b("intlNumUtils").formatNumberWithThousandDelimiters(Number(l),f?0:h));!d?g=e?l+" "+a:String(l):(typeof l==="number"&&(l=""+l),g=k(a,l,e));c<0&&(g="-"+g);return g}function d(a,b,c,d,e){return l(a.currency,a.amount,b,c,d,e)}function f(a){a=p(a);return a!=null?Object.keys(a):[]}function m(a){return!h[a]?null:h[a].screen_name}function n(a){return!h[a]?null:h[a].symbol}function o(a){return!h[a]?null:h[a].offset}function p(a){switch(a){case 2:return g;case 1:return h;default:return null}}e.exports={DEFAULT_AMOUNT_OFFSET:i,formatAmount:l,formatAmountWithExtendedSymbol:c,formatAmountX:a,formatCurrencyAmount:d,formatRawAmount:k,getAllCurrencies:f,getCurrencyScreenName:m,getCurrencySymbol:n,getCurrencyOffset:o}}),null);
__d("react-dom",["react-dom-0.0.0"],(function(a,b,c,d,e,f){e.exports=b("react-dom-0.0.0")()}),null);