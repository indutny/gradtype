!function(e){var t={};function n(o){if(t[o])return t[o].exports;var r=t[o]={i:o,l:!1,exports:{}};return e[o].call(r.exports,r,r.exports,n),r.l=!0,r.exports}n.m=e,n.c=t,n.d=function(e,t,o){n.o(e,t)||Object.defineProperty(e,t,{configurable:!1,enumerable:!0,get:o})},n.r=function(e){Object.defineProperty(e,"__esModule",{value:!0})},n.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return n.d(t,"a",t),t},n.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},n.p="",n(n.s=2)}([function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});t.default=["He turned round and, walking to the window, drew up the blind","The true mystery of the world is the visible, not the invisible","The common people who acted with me seemed to me to be godlike","Having locked the door behind him, he crept quietly downstairs","Success was given to the strong, failure thrust upon the weak","I should not be sorry to see you disgraced, publicly disgraced","England is bad enough I know, and English society is all wrong","And I don't think it really matters about your not being there","It was the imagination that set remorse to dog the feet of sin","I was away with my love in a forest that no man had ever seen","As for being poisoned by a book, there is no such thing as that","The folk don't like to have that sort of thing in their houses","Modern morality consists in accepting the standard of one's age","The flower seemed to quiver, and then swayed gently to and fro","Every month as it wanes brings you nearer to something dreadful","She will represent something to you that you have never known","Our grandmothers painted in order to try and talk brilliantly","It would have made me in love with love for the rest of my life","It is the confession, not the priest, that gives us absolution","I want you to get rid of the dreadful people you associate with"].map(function(e){return e.replace(/\s/g,"␣").toLowerCase()})},function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var o=/firefox/i.test(window.navigator.userAgent),r=o||void 0===window.performance?function(){return Date.now()}:function(){return window.performance.now()};function s(){return r()}t.now=s,t.detect=function(){if(!o)return!1;var e={};function t(t){var n=s();return function(t){for(var n=0;n<t;n++)void 0===e.x?e.x=0:e.x++}(t),s()-n}function n(e){for(var n=0,o=0;o<100;o++)n+=t(e);return n/100}for(var r=[],i=1;i<33554432;i*=1.1){var a=n(i);if(r.push(a),a>1)break}var u=[];for(i=r.length-1;i>=1;i--){var h=r[i],d=r[i-1],l=h/1.1,c=Math.abs(l-d)/(l+1e-24);if(c>1)break;u.push(c)}return u.length/(r.length-1)>.5}},function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var o=n(1),r=n(0),s="gradtype-survey-v1",i=["You're doing great!","Just few more!","Almost there!"],a=Math.floor(r.default.length)/(i.length+1);new(function(){function e(){var e=this;this.log=[],this.start=o.now(),this.elems={display:document.getElementById("display"),counter:document.getElementById("counter"),wrap:document.getElementById("wrap"),reassure:document.getElementById("reassure")},this.sentenceIndex=0,this.charIndex=0,this.lastReassure=0,window.localStorage&&window.localStorage.getItem(s)?this.complete():(this.displaySentence(),window.addEventListener("keydown",function(t){return t.preventDefault(),e.onKeyDown(t.key),!1},!0),window.addEventListener("keyup",function(t){return t.preventDefault(),e.onKeyUp(t.key),!1},!0))}return e.prototype.displaySentence=function(){var e=r.default[this.sentenceIndex];this.elems.counter.textContent=(r.default.length-this.sentenceIndex).toString(),this.elems.display.innerHTML="<span class=sentence-completed>"+e.slice(0,this.charIndex)+"</span><span class=sentence-pending>"+e.slice(this.charIndex)},e.prototype.nextSentence=function(){var e=this;if(this.charIndex=0,this.sentenceIndex++,this.log.push("r"),this.sentenceIndex-this.lastReassure>=a&&(this.lastReassure=this.sentenceIndex,this.elems.reassure.textContent=i.shift()||""),this.sentenceIndex===r.default.length)return this.elems.counter.textContent="0",void this.save(function(t,n){if(t)return e.error();e.complete(n)});this.displaySentence()},e.prototype.onKeyDown=function(e){var t=this,n=o.now();if(this.log.push({e:"d",ts:(n-this.start)/1e3,k:e}),this.sentenceIndex!==r.default.length){var s=r.default[this.sentenceIndex],i=s[this.charIndex];(e===i||" "===e&&"␣"===i)&&(this.charIndex++,this.displaySentence(),this.charIndex===s.length&&setTimeout(function(){t.nextSentence()},50))}},e.prototype.onKeyUp=function(e){var t=o.now();this.log.push({e:"u",ts:(t-this.start)/1e3,k:e})},e.prototype.save=function(e){var t=JSON.stringify(this.log);this.elems.wrap.innerHTML="<h1>Uploading, please do not close this window...</h1>";var n=new XMLHttpRequest;n.onload=function(){var t;try{t=JSON.parse(n.responseText)}catch(t){return e(t)}return t.code?e(void 0,t.code):e(new Error("No response code"))},n.onerror=function(t){return e(new Error("XHR error"))},n.open("PUT","https://gradtype-survey.darksi.de/",!0),n.setRequestHeader("Content-Type","application/json"),n.send(t)},e.prototype.complete=function(e){window.localStorage&&window.localStorage.setItem(s,"submitted");var t=this.elems.wrap;t.innerHTML=void 0===e?"<h1>Thank you for submitting survey!</h1>":'<h1>Thank you for submitting survey!</h1><h1 style="color:red">Your code is '+e+"</h1>"},e.prototype.error=function(){this.elems.wrap.innerHTML="<h1>Server error, please retry later!</h1>"},e}())}]);
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8vd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8vLy4vc3JjL3NlbnRlbmNlcy50cyIsIndlYnBhY2s6Ly8vLi9zcmMvcGVyZm9ybWFuY2UudHMiLCJ3ZWJwYWNrOi8vLy4vc3JjL2FwcC50cyJdLCJuYW1lcyI6WyJpbnN0YWxsZWRNb2R1bGVzIiwiX193ZWJwYWNrX3JlcXVpcmVfXyIsIm1vZHVsZUlkIiwiZXhwb3J0cyIsIm1vZHVsZSIsImkiLCJsIiwibW9kdWxlcyIsImNhbGwiLCJtIiwiYyIsImQiLCJuYW1lIiwiZ2V0dGVyIiwibyIsIk9iamVjdCIsImRlZmluZVByb3BlcnR5IiwiY29uZmlndXJhYmxlIiwiZW51bWVyYWJsZSIsImdldCIsInIiLCJ2YWx1ZSIsIm4iLCJfX2VzTW9kdWxlIiwib2JqZWN0IiwicHJvcGVydHkiLCJwcm90b3R5cGUiLCJoYXNPd25Qcm9wZXJ0eSIsInAiLCJzIiwiZGVmYXVsdCIsIm1hcCIsInNlbnRlbmNlIiwicmVwbGFjZSIsInRvTG93ZXJDYXNlIiwiaXNGaXJlZm94IiwidGVzdCIsIndpbmRvdyIsIm5hdmlnYXRvciIsInVzZXJBZ2VudCIsInRzIiwidW5kZWZpbmVkIiwicGVyZm9ybWFuY2UiLCJEYXRlIiwibm93IiwiZGV0ZWN0Iiwic2NvcGUiLCJtZWFzdXJlIiwidGltZXMiLCJzdGFydCIsIngiLCJidXN5IiwibWVhbiIsInJlc3VsdCIsInZhbHVlcyIsInB1c2giLCJkZWx0YXMiLCJsZW5ndGgiLCJjdXIiLCJwcmV2IiwiZXhwZWN0ZWQiLCJkZWx0YSIsIk1hdGgiLCJhYnMiLCJzZW50ZW5jZXNfMSIsIkxTX0tFWSIsIlJFQVNTVVJFIiwiUkVBU1NVUkVfRVZFUlkiLCJmbG9vciIsIkFwcGxpY2F0aW9uIiwiX3RoaXMiLCJ0aGlzIiwibG9nIiwiZWxlbXMiLCJkaXNwbGF5IiwiZG9jdW1lbnQiLCJnZXRFbGVtZW50QnlJZCIsImNvdW50ZXIiLCJ3cmFwIiwicmVhc3N1cmUiLCJzZW50ZW5jZUluZGV4IiwiY2hhckluZGV4IiwibGFzdFJlYXNzdXJlIiwibG9jYWxTdG9yYWdlIiwiZ2V0SXRlbSIsImNvbXBsZXRlIiwiZGlzcGxheVNlbnRlbmNlIiwiYWRkRXZlbnRMaXN0ZW5lciIsImUiLCJwcmV2ZW50RGVmYXVsdCIsIm9uS2V5RG93biIsImtleSIsIm9uS2V5VXAiLCJ0ZXh0Q29udGVudCIsInRvU3RyaW5nIiwiaW5uZXJIVE1MIiwic2xpY2UiLCJuZXh0U2VudGVuY2UiLCJzaGlmdCIsInNhdmUiLCJlcnIiLCJjb2RlIiwiZXJyb3IiLCJrIiwic2V0VGltZW91dCIsImNhbGxiYWNrIiwianNvbiIsIkpTT04iLCJzdHJpbmdpZnkiLCJ4aHIiLCJYTUxIdHRwUmVxdWVzdCIsIm9ubG9hZCIsInJlc3BvbnNlIiwicGFyc2UiLCJyZXNwb25zZVRleHQiLCJFcnJvciIsIm9uZXJyb3IiLCJvcGVuIiwic2V0UmVxdWVzdEhlYWRlciIsInNlbmQiLCJzZXRJdGVtIl0sIm1hcHBpbmdzIjoiYUFDQSxJQUFBQSxLQUdBLFNBQUFDLEVBQUFDLEdBR0EsR0FBQUYsRUFBQUUsR0FDQSxPQUFBRixFQUFBRSxHQUFBQyxRQUdBLElBQUFDLEVBQUFKLEVBQUFFLElBQ0FHLEVBQUFILEVBQ0FJLEdBQUEsRUFDQUgsWUFVQSxPQU5BSSxFQUFBTCxHQUFBTSxLQUFBSixFQUFBRCxRQUFBQyxJQUFBRCxRQUFBRixHQUdBRyxFQUFBRSxHQUFBLEVBR0FGLEVBQUFELFFBS0FGLEVBQUFRLEVBQUFGLEVBR0FOLEVBQUFTLEVBQUFWLEVBR0FDLEVBQUFVLEVBQUEsU0FBQVIsRUFBQVMsRUFBQUMsR0FDQVosRUFBQWEsRUFBQVgsRUFBQVMsSUFDQUcsT0FBQUMsZUFBQWIsRUFBQVMsR0FDQUssY0FBQSxFQUNBQyxZQUFBLEVBQ0FDLElBQUFOLEtBTUFaLEVBQUFtQixFQUFBLFNBQUFqQixHQUNBWSxPQUFBQyxlQUFBYixFQUFBLGNBQWlEa0IsT0FBQSxLQUlqRHBCLEVBQUFxQixFQUFBLFNBQUFsQixHQUNBLElB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QkEsRUFBSSxTQUloRGhDLEVBQUEzQyxVQUFBNEUsTUFBQSxXQUNFL0IsS0FBS0UsTUFBTUssS0FBS2lCLFVBQVksOENBRWhDMUIsRUExSkEiLCJmaWxlIjoiYnVuZGxlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiIFx0Ly8gVGhlIG1vZHVsZSBjYWNoZVxuIFx0dmFyIGluc3RhbGxlZE1vZHVsZXMgPSB7fTtcblxuIFx0Ly8gVGhlIHJlcXVpcmUgZnVuY3Rpb25cbiBcdGZ1bmN0aW9uIF9fd2VicGFja19yZXF1aXJlX18obW9kdWxlSWQpIHtcblxuIFx0XHQvLyBDaGVjayBpZiBtb2R1bGUgaXMgaW4gY2FjaGVcbiBcdFx0aWYoaW5zdGFsbGVkTW9kdWxlc1ttb2R1bGVJZF0pIHtcbiBcdFx0XHRyZXR1cm4gaW5zdGFsbGVkTW9kdWxlc1ttb2R1bGVJZF0uZXhwb3J0cztcbiBcdFx0fVxuIFx0XHQvLyBDcmVhdGUgYSBuZXcgbW9kdWxlIChhbmQgcHV0IGl0IGludG8gdGhlIGNhY2hlKVxuIFx0XHR2YXIgbW9kdWxlID0gaW5zdGFsbGVkTW9kdWxlc1ttb2R1bGVJZF0gPSB7XG4gXHRcdFx0aTogbW9kdWxlSWQsXG4gXHRcdFx0bDogZmFsc2UsXG4gXHRcdFx0ZXhwb3J0czoge31cbiBcdFx0fTtcblxuIFx0XHQvLyBFeGVjdXRlIHRoZSBtb2R1bGUgZnVuY3Rpb25cbiBcdFx0bW9kdWxlc1ttb2R1bGVJZF0uY2FsbChtb2R1bGUuZXhwb3J0cywgbW9kdWxlLCBtb2R1bGUuZXhwb3J0cywgX193ZWJwYWNrX3JlcXVpcmVfXyk7XG5cbiBcdFx0Ly8gRmxhZyB0aGUgbW9kdWxlIGFzIGxvYWRlZFxuIFx0XHRtb2R1bGUubCA9IHRydWU7XG5cbiBcdFx0Ly8gUmV0dXJuIHRoZSBleHBvcnRzIG9mIHRoZSBtb2R1bGVcbiBcdFx0cmV0dXJuIG1vZHVsZS5leHBvcnRzO1xuIFx0fVxuXG5cbiBcdC8vIGV4cG9zZSB0aGUgbW9kdWxlcyBvYmplY3QgKF9fd2VicGFja19tb2R1bGVzX18pXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm0gPSBtb2R1bGVzO1xuXG4gXHQvLyBleHBvc2UgdGhlIG1vZHVsZSBjYWNoZVxuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5jID0gaW5zdGFsbGVkTW9kdWxlcztcblxuIFx0Ly8gZGVmaW5lIGdldHRlciBmdW5jdGlvbiBmb3IgaGFybW9ueSBleHBvcnRzXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLmQgPSBmdW5jdGlvbihleHBvcnRzLCBuYW1lLCBnZXR0ZXIpIHtcbiBcdFx0aWYoIV9fd2VicGFja19yZXF1aXJlX18ubyhleHBvcnRzLCBuYW1lKSkge1xuIFx0XHRcdE9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBuYW1lLCB7XG4gXHRcdFx0XHRjb25maWd1cmFibGU6IGZhbHNlLFxuIFx0XHRcdFx0ZW51bWVyYWJsZTogdHJ1ZSxcbiBcdFx0XHRcdGdldDogZ2V0dGVyXG4gXHRcdFx0fSk7XG4gXHRcdH1cbiBcdH07XG5cbiBcdC8vIGRlZmluZSBfX2VzTW9kdWxlIG9uIGV4cG9ydHNcbiBcdF9fd2VicGFja19yZXF1aXJlX18uciA9IGZ1bmN0aW9uKGV4cG9ydHMpIHtcbiBcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsICdfX2VzTW9kdWxlJywgeyB2YWx1ZTogdHJ1ZSB9KTtcbiBcdH07XG5cbiBcdC8vIGdldERlZmF1bHRFeHBvcnQgZnVuY3Rpb24gZm9yIGNvbXBhdGliaWxpdHkgd2l0aCBub24taGFybW9ueSBtb2R1bGVzXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm4gPSBmdW5jdGlvbihtb2R1bGUpIHtcbiBcdFx0dmFyIGdldHRlciA9IG1vZHVsZSAmJiBtb2R1bGUuX19lc01vZHVsZSA/XG4gXHRcdFx0ZnVuY3Rpb24gZ2V0RGVmYXVsdCgpIHsgcmV0dXJuIG1vZHVsZVsnZGVmYXVsdCddOyB9IDpcbiBcdFx0XHRmdW5jdGlvbiBnZXRNb2R1bGVFeHBvcnRzKCkgeyByZXR1cm4gbW9kdWxlOyB9O1xuIFx0XHRfX3dlYnBhY2tfcmVxdWlyZV9fLmQoZ2V0dGVyLCAnYScsIGdldHRlcik7XG4gXHRcdHJldHVybiBnZXR0ZXI7XG4gXHR9O1xuXG4gXHQvLyBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGxcbiBcdF9fd2VicGFja19yZXF1aXJlX18ubyA9IGZ1bmN0aW9uKG9iamVjdCwgcHJvcGVydHkpIHsgcmV0dXJuIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvYmplY3QsIHByb3BlcnR5KTsgfTtcblxuIFx0Ly8gX193ZWJwYWNrX3B1YmxpY19wYXRoX19cbiBcdF9fd2VicGFja19yZXF1aXJlX18ucCA9IFwiXCI7XG5cblxuIFx0Ly8gTG9hZCBlbnRyeSBtb2R1bGUgYW5kIHJldHVybiBleHBvcnRzXG4gXHRyZXR1cm4gX193ZWJwYWNrX3JlcXVpcmVfXyhfX3dlYnBhY2tfcmVxdWlyZV9fLnMgPSAyKTtcbiIsImNvbnN0IHNlbnRlbmNlcyA9IFtcbiAgXCJIZSB0dXJuZWQgcm91bmQgYW5kLCB3YWxraW5nIHRvIHRoZSB3aW5kb3csIGRyZXcgdXAgdGhlIGJsaW5kXCIsXG4gIFwiVGhlIHRydWUgbXlzdGVyeSBvZiB0aGUgd29ybGQgaXMgdGhlIHZpc2libGUsIG5vdCB0aGUgaW52aXNpYmxlXCIsXG4gIFwiVGhlIGNvbW1vbiBwZW9wbGUgd2hvIGFjdGVkIHdpdGggbWUgc2VlbWVkIHRvIG1lIHRvIGJlIGdvZGxpa2VcIixcbiAgXCJIYXZpbmcgbG9ja2VkIHRoZSBkb29yIGJlaGluZCBoaW0sIGhlIGNyZXB0IHF1aWV0bHkgZG93bnN0YWlyc1wiLFxuICBcIlN1Y2Nlc3Mgd2FzIGdpdmVuIHRvIHRoZSBzdHJvbmcsIGZhaWx1cmUgdGhydXN0IHVwb24gdGhlIHdlYWtcIixcbiAgXCJJIHNob3VsZCBub3QgYmUgc29ycnkgdG8gc2VlIHlvdSBkaXNncmFjZWQsIHB1YmxpY2x5IGRpc2dyYWNlZFwiLFxuICBcIkVuZ2xhbmQgaXMgYmFkIGVub3VnaCBJIGtub3csIGFuZCBFbmdsaXNoIHNvY2lldHkgaXMgYWxsIHdyb25nXCIsXG4gIFwiQW5kIEkgZG9uJ3QgdGhpbmsgaXQgcmVhbGx5IG1hdHRlcnMgYWJvdXQgeW91ciBub3QgYmVpbmcgdGhlcmVcIixcbiAgXCJJdCB3YXMgdGhlIGltYWdpbmF0aW9uIHRoYXQgc2V0IHJlbW9yc2UgdG8gZG9nIHRoZSBmZWV0IG9mIHNpblwiLFxuICBcIkkgd2FzIGF3YXkgd2l0aCBteSBsb3ZlIGluIGEgZm9yZXN0IHRoYXQgbm8gbWFuIGhhZCBldmVyIHNlZW5cIixcbiAgXCJBcyBmb3IgYmVpbmcgcG9pc29uZWQgYnkgYSBib29rLCB0aGVyZSBpcyBubyBzdWNoIHRoaW5nIGFzIHRoYXRcIixcbiAgXCJUaGUgZm9sayBkb24ndCBsaWtlIHRvIGhhdmUgdGhhdCBzb3J0IG9mIHRoaW5nIGluIHRoZWlyIGhvdXNlc1wiLFxuICBcIk1vZGVybiBtb3JhbGl0eSBjb25zaXN0cyBpbiBhY2NlcHRpbmcgdGhlIHN0YW5kYXJkIG9mIG9uZSdzIGFnZVwiLFxuICBcIlRoZSBmbG93ZXIgc2VlbWVkIHRvIHF1aXZlciwgYW5kIHRoZW4gc3dheWVkIGdlbnRseSB0byBhbmQgZnJvXCIsXG4gIFwiRXZlcnkgbW9udGggYXMgaXQgd2FuZXMgYnJpbmdzIHlvdSBuZWFyZXIgdG8gc29tZXRoaW5nIGRyZWFkZnVsXCIsXG4gIFwiU2hlIHdpbGwgcmVwcmVzZW50IHNvbWV0aGluZyB0byB5b3UgdGhhdCB5b3UgaGF2ZSBuZXZlciBrbm93blwiLFxuICBcIk91ciBncmFuZG1vdGhlcnMgcGFpbnRlZCBpbiBvcmRlciB0byB0cnkgYW5kIHRhbGsgYnJpbGxpYW50bHlcIixcbiAgXCJJdCB3b3VsZCBoYXZlIG1hZGUgbWUgaW4gbG92ZSB3aXRoIGxvdmUgZm9yIHRoZSByZXN0IG9mIG15IGxpZmVcIixcbiAgXCJJdCBpcyB0aGUgY29uZmVzc2lvbiwgbm90IHRoZSBwcmllc3QsIHRoYXQgZ2l2ZXMgdXMgYWJzb2x1dGlvblwiLFxuICBcIkkgd2FudCB5b3UgdG8gZ2V0IHJpZCBvZiB0aGUgZHJlYWRmdWwgcGVvcGxlIHlvdSBhc3NvY2lhdGUgd2l0aFwiXG5dO1xuXG5leHBvcnQgZGVmYXVsdCBzZW50ZW5jZXMubWFwKChzZW50ZW5jZSkgPT4ge1xuICByZXR1cm4gc2VudGVuY2UucmVwbGFjZSgvXFxzL2csICfikKMnKS50b0xvd2VyQ2FzZSgpO1xufSk7XG4iLCJjb25zdCBpc0ZpcmVmb3ggPSAvZmlyZWZveC9pLnRlc3Qod2luZG93Lm5hdmlnYXRvci51c2VyQWdlbnQpO1xuXG5jb25zdCB0cyA9IChpc0ZpcmVmb3ggfHwgd2luZG93LnBlcmZvcm1hbmNlID09PSB1bmRlZmluZWQpID8gKCkgPT4gRGF0ZS5ub3coKSA6XG4gICgpID0+IHdpbmRvdy5wZXJmb3JtYW5jZS5ub3coKTtcblxuZXhwb3J0IGZ1bmN0aW9uIG5vdygpOiBudW1iZXIge1xuICByZXR1cm4gdHMoKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRldGVjdCgpOiBib29sZWFuIHtcbiAgaWYgKCFpc0ZpcmVmb3gpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cblxuICBjb25zdCBzY29wZTogYW55ID0ge307XG5cbiAgZnVuY3Rpb24gYnVzeSh0aW1lczogbnVtYmVyKTogdm9pZCB7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aW1lczsgaSsrKSB7XG4gICAgICBpZiAoc2NvcGUueCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICAgIHNjb3BlLnggPSAwO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgc2NvcGUueCsrO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIG1lYXN1cmUodGltZXM6IG51bWJlcik6IG51bWJlciB7XG4gICAgY29uc3Qgc3RhcnQgPSBub3coKTtcbiAgICBidXN5KHRpbWVzKTtcbiAgICByZXR1cm4gbm93KCkgLSBzdGFydDtcbiAgfVxuXG4gIGZ1bmN0aW9uIG1lYW4odGltZXM6IG51bWJlcik6IG51bWJlciB7XG4gICAgbGV0IHJlc3VsdDogbnVtYmVyID0gMDtcbiAgICBjb25zdCBjb3VudCA9IDEwMDtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGNvdW50OyBpKyspIHtcbiAgICAgIHJlc3VsdCArPSBtZWFzdXJlKHRpbWVzKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdCAvPSBjb3VudDtcbiAgfVxuXG4gIGNvbnN0IG11bCA9IDEuMTtcblxuICBjb25zdCB2YWx1ZXMgPSBbXTtcbiAgZm9yIChsZXQgaSA9IDE7IGkgPCAzMzU1NDQzMjsgaSAqPSBtdWwpIHtcbiAgICBjb25zdCBtID0gbWVhbihpKTtcbiAgICB2YWx1ZXMucHVzaChtKTtcbiAgICBpZiAobSA+IDEpIHtcbiAgICAgIGJyZWFrO1xuICAgIH1cbiAgfVxuXG4gIGNvbnN0IGRlbHRhczogbnVtYmVyW10gPSBbXTtcbiAgZm9yIChsZXQgaSA9IHZhbHVlcy5sZW5ndGggLSAxOyBpID49IDE7IGktLSkge1xuICAgIGNvbnN0IGN1ciA9IHZhbHVlc1tpXTtcbiAgICBjb25zdCBwcmV2ID0gdmFsdWVzW2kgLSAxXTtcblxuICAgIGNvbnN0IGV4cGVjdGVkID0gY3VyIC8gbXVsO1xuICAgIGNvbnN0IGRlbHRhID0gTWF0aC5hYnMoZXhwZWN0ZWQgLSBwcmV2KSAvIChleHBlY3RlZCArIDFlLTI0KTtcblxuICAgIGlmIChkZWx0YSA+IDEpIHtcbiAgICAgIGJyZWFrO1xuICAgIH1cblxuICAgIGRlbHRhcy5wdXNoKGRlbHRhKTtcbiAgfVxuXG4gIGNvbnN0IHBlcmNlbnQgPSBkZWx0YXMubGVuZ3RoIC8gKHZhbHVlcy5sZW5ndGggLSAxKTtcblxuICAvLyBDb21wbGV0ZWx5IGFyYml0cmFyeVxuICByZXR1cm4gcGVyY2VudCA+IDAuNTtcbn1cbiIsImltcG9ydCAqIGFzIHBlcmZvcm1hbmNlIGZyb20gJy4vcGVyZm9ybWFuY2UnO1xuaW1wb3J0IHsgZGVmYXVsdCBhcyBzZW50ZW5jZXMgfSBmcm9tICcuL3NlbnRlbmNlcyc7XG5cbmNvbnN0IEFQSV9FTkRQT0lOVCA9ICdodHRwczovL2dyYWR0eXBlLXN1cnZleS5kYXJrc2kuZGUvJztcbmNvbnN0IExTX0tFWSA9ICdncmFkdHlwZS1zdXJ2ZXktdjEnO1xuXG5jb25zdCBSRUFTU1VSRTogc3RyaW5nW10gPSBbXG4gICdZb3VcXCdyZSBkb2luZyBncmVhdCEnLFxuICAnSnVzdCBmZXcgbW9yZSEnLFxuICAnQWxtb3N0IHRoZXJlISdcbl07XG5cbmNvbnN0IFJFQVNTVVJFX0VWRVJZID0gTWF0aC5mbG9vcihzZW50ZW5jZXMubGVuZ3RoKSAvIChSRUFTU1VSRS5sZW5ndGggKyAxKTtcblxudHlwZSBMb2dLaW5kID0gJ2QnIHwgJ3UnO1xuXG50eXBlIExvZ0V2ZW50ID0ge1xuICByZWFkb25seSBlOiBMb2dLaW5kO1xuICByZWFkb25seSB0czogbnVtYmVyO1xuICByZWFkb25seSBrOiBzdHJpbmc7XG59IHwgJ3InO1xuXG5jbGFzcyBBcHBsaWNhdGlvbiB7XG4gIHByaXZhdGUgcmVhZG9ubHkgbG9nOiBMb2dFdmVudFtdID0gW107XG4gIHByaXZhdGUgcmVhZG9ubHkgc3RhcnQ6IG51bWJlciA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBwcml2YXRlIHJlYWRvbmx5IGVsZW1zID0ge1xuICAgIGRpc3BsYXk6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkaXNwbGF5JykhLFxuICAgIGNvdW50ZXI6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjb3VudGVyJykhLFxuICAgIHdyYXA6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd3cmFwJykhLFxuICAgIHJlYXNzdXJlOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVhc3N1cmUnKSEsXG4gIH07XG5cbiAgcHJpdmF0ZSBzZW50ZW5jZUluZGV4OiBudW1iZXIgPSAwO1xuICBwcml2YXRlIGNoYXJJbmRleDogbnVtYmVyID0gMDtcbiAgcHJpdmF0ZSBsYXN0UmVhc3N1cmU6IG51bWJlciA9IDA7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgaWYgKHdpbmRvdy5sb2NhbFN0b3JhZ2UgJiYgd2luZG93LmxvY2FsU3RvcmFnZS5nZXRJdGVtKExTX0tFWSkpIHtcbiAgICAgIHRoaXMuY29tcGxldGUoKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICB0aGlzLmRpc3BsYXlTZW50ZW5jZSgpO1xuXG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoJ2tleWRvd24nLCAoZSkgPT4ge1xuICAgICAgZS5wcmV2ZW50RGVmYXVsdCgpO1xuICAgICAgdGhpcy5vbktleURvd24oZS5rZXkpO1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH0sIHRydWUpO1xuXG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoJ2tleXVwJywgKGUpID0+IHtcbiAgICAgIGUucHJldmVudERlZmF1bHQoKTtcbiAgICAgIHRoaXMub25LZXlVcChlLmtleSk7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfSwgdHJ1ZSk7XG4gIH1cblxuICBkaXNwbGF5U2VudGVuY2UoKSB7XG4gICAgY29uc3Qgc2VudGVuY2UgPSBzZW50ZW5jZXNbdGhpcy5zZW50ZW5jZUluZGV4XTtcblxuICAgIHRoaXMuZWxlbXMuY291bnRlci50ZXh0Q29udGVudCA9XG4gICAgICAoc2VudGVuY2VzLmxlbmd0aCAtIHRoaXMuc2VudGVuY2VJbmRleCkudG9TdHJpbmcoKTtcbiAgICB0aGlzLmVsZW1zLmRpc3BsYXkuaW5uZXJIVE1MID1cbiAgICAgICc8c3BhbiBjbGFzcz1zZW50ZW5jZS1jb21wbGV0ZWQ+JyArXG4gICAgICBzZW50ZW5jZS5zbGljZSgwLCB0aGlzLmNoYXJJbmRleCkgK1xuICAgICAgJzwvc3Bhbj4nICtcbiAgICAgICc8c3BhbiBjbGFzcz1zZW50ZW5jZS1wZW5kaW5nPicgK1xuICAgICAgc2VudGVuY2Uuc2xpY2UodGhpcy5jaGFySW5kZXgpXG4gICAgICAnPC9zcGFuPic7XG4gIH1cblxuICBuZXh0U2VudGVuY2UoKSB7XG4gICAgdGhpcy5jaGFySW5kZXggPSAwO1xuICAgIHRoaXMuc2VudGVuY2VJbmRleCsrO1xuICAgIHRoaXMubG9nLnB1c2goJ3InKTtcblxuICAgIGlmICh0aGlzLnNlbnRlbmNlSW5kZXggLSB0aGlzLmxhc3RSZWFzc3VyZSA+PSBSRUFTU1VSRV9FVkVSWSkge1xuICAgICAgdGhpcy5sYXN0UmVhc3N1cmUgPSB0aGlzLnNlbnRlbmNlSW5kZXg7XG4gICAgICB0aGlzLmVsZW1zLnJlYXNzdXJlLnRleHRDb250ZW50ID0gUkVBU1NVUkUuc2hpZnQoKSB8fCAnJztcbiAgICB9XG5cbiAgICBpZiAodGhpcy5zZW50ZW5jZUluZGV4ID09PSBzZW50ZW5jZXMubGVuZ3RoKSB7XG4gICAgICB0aGlzLmVsZW1zLmNvdW50ZXIudGV4dENvbnRlbnQgPSAnMCc7XG5cbiAgICAgIHRoaXMuc2F2ZSgoZXJyLCBjb2RlKSA9PiB7XG4gICAgICAgIGlmIChlcnIpIHtcbiAgICAgICAgICByZXR1cm4gdGhpcy5lcnJvcigpO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMuY29tcGxldGUoY29kZSEpO1xuICAgICAgfSk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdGhpcy5kaXNwbGF5U2VudGVuY2UoKTtcbiAgfVxuXG4gIG9uS2V5RG93bihrZXk6IHN0cmluZykge1xuICAgIGNvbnN0IG5vdyA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgIHRoaXMubG9nLnB1c2goeyBlOiAnZCcsIHRzOiAobm93IC0gdGhpcy5zdGFydCkgLyAxMDAwLCBrOiBrZXkgfSk7XG5cbiAgICBpZiAodGhpcy5zZW50ZW5jZUluZGV4ID09PSBzZW50ZW5jZXMubGVuZ3RoKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3Qgc2VudGVuY2UgPSBzZW50ZW5jZXNbdGhpcy5zZW50ZW5jZUluZGV4XTtcbiAgICBjb25zdCBleHBlY3RlZCA9IHNlbnRlbmNlW3RoaXMuY2hhckluZGV4XTtcbiAgICBpZiAoa2V5ICE9PSBleHBlY3RlZCAmJiAhKGtleSA9PT0gJyAnICYmIGV4cGVjdGVkID09PSAn4pCjJykpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICB0aGlzLmNoYXJJbmRleCsrO1xuICAgIHRoaXMuZGlzcGxheVNlbnRlbmNlKCk7XG5cbiAgICBpZiAodGhpcy5jaGFySW5kZXggIT09IHNlbnRlbmNlLmxlbmd0aCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIC8vIEdpdmUgZW5vdWdoIHRpbWUgdG8gcmVjb3JkIHRoZSBsYXN0IGtleXN0cm9rZVxuICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgdGhpcy5uZXh0U2VudGVuY2UoKTtcbiAgICB9LCA1MCk7XG4gIH1cblxuICBvbktleVVwKGtleTogc3RyaW5nKSB7XG4gICAgY29uc3Qgbm93ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gICAgdGhpcy5sb2cucHVzaCh7IGU6ICd1JywgdHM6IChub3cgLSB0aGlzLnN0YXJ0KSAvIDEwMDAsIGs6IGtleSB9KTtcbiAgfVxuXG4gIHNhdmUoY2FsbGJhY2s6IChlcnI/OiBFcnJvciwgY29kZT86IHN0cmluZykgPT4gdm9pZCkge1xuICAgIGNvbnN0IGpzb24gPSBKU09OLnN0cmluZ2lmeSh0aGlzLmxvZyk7XG5cbiAgICB0aGlzLmVsZW1zLndyYXAuaW5uZXJIVE1MID1cbiAgICAgICc8aDE+VXBsb2FkaW5nLCBwbGVhc2UgZG8gbm90IGNsb3NlIHRoaXMgd2luZG93Li4uPC9oMT4nO1xuXG4gICAgY29uc3QgeGhyID0gbmV3IFhNTEh0dHBSZXF1ZXN0KCk7XG5cbiAgICB4aHIub25sb2FkID0gKCkgPT4ge1xuICAgICAgbGV0IHJlc3BvbnNlOiBhbnk7XG4gICAgICB0cnkge1xuICAgICAgICByZXNwb25zZSA9IEpTT04ucGFyc2UoeGhyLnJlc3BvbnNlVGV4dCk7XG4gICAgICB9IGNhdGNoIChlKSB7XG4gICAgICAgIHJldHVybiBjYWxsYmFjayhlKTtcbiAgICAgIH1cblxuICAgICAgaWYgKCFyZXNwb25zZS5jb2RlKSB7XG4gICAgICAgIHJldHVybiBjYWxsYmFjayhuZXcgRXJyb3IoJ05vIHJlc3BvbnNlIGNvZGUnKSk7XG4gICAgICB9XG5cbiAgICAgIHJldHVybiBjYWxsYmFjayh1bmRlZmluZWQsIHJlc3BvbnNlLmNvZGUpO1xuICAgIH07XG5cbiAgICB4aHIub25lcnJvciA9IChlcnIpID0+IHtcbiAgICAgIHJldHVybiBjYWxsYmFjayhuZXcgRXJyb3IoJ1hIUiBlcnJvcicpKTtcbiAgICB9O1xuXG4gICAgeGhyLm9wZW4oJ1BVVCcsIEFQSV9FTkRQT0lOVCwgdHJ1ZSk7XG4gICAgeGhyLnNldFJlcXVlc3RIZWFkZXIoJ0NvbnRlbnQtVHlwZScsICdhcHBsaWNhdGlvbi9qc29uJyk7XG4gICAgeGhyLnNlbmQoanNvbik7XG4gIH1cblxuICBjb21wbGV0ZShjb2RlPzogc3RyaW5nKSB7XG4gICAgaWYgKHdpbmRvdy5sb2NhbFN0b3JhZ2UpIHtcbiAgICAgIHdpbmRvdy5sb2NhbFN0b3JhZ2Uuc2V0SXRlbShMU19LRVksICdzdWJtaXR0ZWQnKTtcbiAgICB9XG4gICAgY29uc3Qgd3JhcCA9IHRoaXMuZWxlbXMud3JhcDtcbiAgICBpZiAoY29kZSA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICB3cmFwLmlubmVySFRNTCA9ICc8aDE+VGhhbmsgeW91IGZvciBzdWJtaXR0aW5nIHN1cnZleSE8L2gxPic7XG4gICAgfSBlbHNlIHtcbiAgICAgIHdyYXAuaW5uZXJIVE1MID0gJzxoMT5UaGFuayB5b3UgZm9yIHN1Ym1pdHRpbmcgc3VydmV5ITwvaDE+JyArXG4gICAgICAgIGA8aDEgc3R5bGU9XCJjb2xvcjpyZWRcIj5Zb3VyIGNvZGUgaXMgJHtjb2RlfTwvaDE+YDtcbiAgICB9XG4gIH1cblxuICBlcnJvcigpIHtcbiAgICB0aGlzLmVsZW1zLndyYXAuaW5uZXJIVE1MID0gJzxoMT5TZXJ2ZXIgZXJyb3IsIHBsZWFzZSByZXRyeSBsYXRlciE8L2gxPic7XG4gIH1cbn1cblxuY29uc3QgYXBwID0gbmV3IEFwcGxpY2F0aW9uKCk7XG4iXSwic291cmNlUm9vdCI6IiJ9