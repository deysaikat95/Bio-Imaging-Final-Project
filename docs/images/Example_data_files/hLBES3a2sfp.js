if (self.CavalryLogger) { CavalryLogger.start_js(["ML\/Hw"]); }

__d("DocumentTitle",["Arbiter"],(function(a,b,c,d,e,f){__p&&__p();var g=1500,h=null,i=!1,j=0,k=[],l=null,m=document.title;function n(){k.length>0?!i?(o(k[j].title),j=++j%k.length):p():(clearInterval(h),h=null,p())}function o(a){document.title=a,i=!0}function p(){q.set(l||m,!0),i=!1}var q=function(){"use strict";__p&&__p();function a(a){this.$1=a}a.get=function(){return m};a.set=function(a,c){var d=a.toString();document.title=d;!c?(m=d,l=null,b("Arbiter").inform("update_title",a)):l=d};a.blink=function(b){b={title:b.toString()};k.push(b);h===null&&(h=setInterval(n,g));return new a(b)};var c=a.prototype;c.stop=function(){var a=k.indexOf(this.$1);a>=0&&(k.splice(a,1),j>a?j--:j==a&&j==k.length&&(j=0))};a.badge=function(c){var d=a.get();d=c?"("+c+") "+d:d;a.set(d,!0);b("Arbiter").inform("update_title_badge",c,"state")};return a}();e.exports=q}),null);