!function(e){var t={};function n(o){if(t[o])return t[o].exports;var r=t[o]={i:o,l:!1,exports:{}};return e[o].call(r.exports,r,r.exports,n),r.l=!0,r.exports}n.m=e,n.c=t,n.d=function(e,t,o){n.o(e,t)||Object.defineProperty(e,t,{configurable:!1,enumerable:!0,get:o})},n.r=function(e){Object.defineProperty(e,"__esModule",{value:!0})},n.n=function(e){var t=e&&e.__esModule?function(){return e.default}:function(){return e};return n.d(t,"a",t),t},n.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},n.p="",n(n.s=2)}([function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});t.default=["It seemed to him that misfortune followed wherever he went","Then he threw his head back and laughed","He hailed it and in a low voice gave the driver an address","I see things differently, I think of them differently","I am forced to bring you into the matter","A bitter blast swept across the square","I thought that I was going to be wonderful","You laugh, but I tell you she has genius","I know now that I was quite right in what I fancied about him","Those who go beneath the surface do so at their peril","To be in love is to surpass one's self","She never gets confused over her dates, and I always do","And yet the thing would still live on","She tells me she is going down to Selby","It is my masterpiece as it stands","But I am much obliged for the compliment, all the same","It might be a most brilliant marriage for Sibyl","Most of the servants were at Selby Royal","The ugly and the stupid have the best of it in this world","Success was given to the strong, failure thrust upon the weak","He came close to him and put his hand upon his shoulder","Her love was trembling in laughter on her lips","It will be a great pity, for it will alter you","It is too ugly, too horrible, too distressing","I came here at once and was miserable at not finding you","The knocking still continued and grew louder","Her companion watched her enviously","Now, wherever you go, you charm the world","He lit a cigarette and then threw it away","The lad listened sulkily to her and made no answer","Gray to wait, Parker: I shall be in in a few moments","Yes: it was certainly a tedious party","There were tears in his eyes as he went downstairs","He would never bring misery upon any one","It was of himself, and of his own future, that he had to think","The man looked at her in terror and began to whimper","And they passed into the dining-room","You gave her good advice and broke her heart","They are horrible, and they don't mean anything","They have had my own divorce-case and Alan Campbell's suicide","Make my excuses to Lady Narborough","Harry, to whom I talked about it, laughed at me","A decent-looking man, sir, but rough-like","Dorian looked at him for a moment","In one of the top-windows stood a lamp","There had been a madness of murder in the air","Of course, I am very fond of Harry","The artist is the creator of beautiful things","That is a great advantage, don't you think so, Mr","Erskine, an absolutely reasonable people","I am analysing women at present, so I ought to know","He felt as if the load had been lifted from him already","But here was a visible symbol of the degradation of sin","The little duchess is quite devoted to you","They are the elect to whom beautiful things mean only beauty","But I wish you had left word where you had really gone to","I know you are surprised at my talking to you like this","The sunlight slipped over the polished leaves","A curious sensation of terror came over me","He was heart-sick at leaving home","People say sometimes that beauty is only superficial","It was merely the name men gave to their mistakes","Lord Henry shrugged his shoulders","It was the most premature definition ever given","It will be a great pity, for it will alter you","It had been given to him by Adrian Singleton","At last he heard a step outside, and the door opened","They have a right to demand it back","There is no mystery in any of them","When Lord Henry had sat down again, Mr","Besides, individualism has really the higher aim","You find me consoled, and you are furious","This one is little more than a boy","Come to the club with Basil and myself","The man who had been shot in the thicket was James Vane","One could hear her singing as she ran upstairs","You were the most unspoiled creature in the whole world","It is so much more real than life","Come and see me some afternoon in Curzon Street","When the verities become acrobats, we can judge them","Isaacs has been very good to us, and we owe him money","Indeed, in some measure it was a disappointment to her","It is the only way I get to know of them","Years ago he was christened Prince Charming","The man who had been shot in the thicket was James Vane","He was not to go to the gold-fields at all","Anything would be better than this dreadful state of doubt","It has a perfect host, and a perfect library","It is the feet of clay that make the gold of the image precious"].map(function(e){return e.replace(/\s/g,"␣").toLowerCase()})},function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var o=/firefox/i.test(window.navigator.userAgent),r=o||void 0===window.performance?function(){return Date.now()}:function(){return window.performance.now()};function a(){return r()}t.now=a,t.detect=function(){if(!o)return!1;var e={};function t(t){var n=a();return function(t){for(var n=0;n<t;n++)void 0===e.x?e.x=0:e.x++}(t),a()-n}function n(e){for(var n=0,o=0;o<100;o++)n+=t(e);return n/100}for(var r=[],s=1;s<33554432;s*=1.1){var i=n(s);if(r.push(i),i>1)break}var h=[];for(s=r.length-1;s>=1;s--){var l=r[s],d=r[s-1],u=l/1.1,c=Math.abs(u-d)/(u+1e-24);if(c>1)break;h.push(c)}return h.length/(r.length-1)>.5}},function(e,t,n){"use strict";Object.defineProperty(t,"__esModule",{value:!0});var o=n(1),r=n(0),a="gradtype-survey-v1",s=["You're doing great!","Just few more!","Almost there!"],i=Math.floor(r.default.length)/(s.length+1);new(function(){function e(){var e=this;this.log=[],this.start=o.now(),this.elems={display:document.getElementById("display"),counter:document.getElementById("counter"),wrap:document.getElementById("wrap"),reassure:document.getElementById("reassure")},this.sentenceIndex=0,this.charIndex=0,this.lastReassure=0,window.localStorage&&window.localStorage.getItem(a)?this.complete():(this.displaySentence(),window.addEventListener("keydown",function(t){e.onKeyDown(t.key)},!0),window.addEventListener("keyup",function(t){e.onKeyUp(t.key)},!0))}return e.prototype.displaySentence=function(){var e=r.default[this.sentenceIndex];this.elems.counter.textContent=(r.default.length-this.sentenceIndex).toString(),this.elems.display.innerHTML="<span class=sentence-completed>"+e.slice(0,this.charIndex)+"</span><span class=sentence-pending>"+e.slice(this.charIndex)},e.prototype.nextSentence=function(){var e=this;if(this.charIndex=0,this.sentenceIndex++,this.log.push("r"),this.sentenceIndex-this.lastReassure>=i&&(this.lastReassure=this.sentenceIndex,this.elems.reassure.textContent=s.shift()||""),this.sentenceIndex===r.default.length)return this.elems.counter.textContent="0",void this.save(function(t,n){if(t)return e.error();e.complete(n)});this.displaySentence()},e.prototype.onKeyDown=function(e){var t=this,n=o.now();if(this.log.push({e:"d",ts:(n-this.start)/1e3,k:e}),this.sentenceIndex!==r.default.length){var a=r.default[this.sentenceIndex],s=a[this.charIndex];(e===s||" "===e&&"␣"===s)&&(this.charIndex++,this.displaySentence(),this.charIndex===a.length&&setTimeout(function(){t.nextSentence()},50))}},e.prototype.onKeyUp=function(e){var t=o.now();this.log.push({e:"u",ts:(t-this.start)/1e3,k:e})},e.prototype.save=function(e){var t=JSON.stringify(this.log);this.elems.wrap.innerHTML="<h1>Uploading, please do not close this window...</h1>";var n=new XMLHttpRequest;n.onload=function(){var t;try{t=JSON.parse(n.responseText)}catch(t){return e(t)}return t.code?e(void 0,t.code):e(new Error("No response code"))},n.onerror=function(t){return e(new Error("XHR error"))},n.open("PUT","https://gradtype-survey.darksi.de/",!0),n.setRequestHeader("Content-Type","application/json"),n.send(t)},e.prototype.complete=function(e){window.localStorage&&window.localStorage.setItem(a,"submitted");var t=this.elems.wrap;t.innerHTML=void 0===e?"<h1>Thank you for submitting survey!</h1>":'<h1>Thank you for submitting survey!</h1><h1 style="color:red">Your code is '+e+"</h1>"},e.prototype.error=function(){this.elems.wrap.innerHTML="<h1>Server error, please retry later!</h1>"},e}())}]);
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly8vd2VicGFjay9ib290c3RyYXAiLCJ3ZWJwYWNrOi8vLy4vc3JjL3NlbnRlbmNlcy50cyIsIndlYnBhY2s6Ly8vLi9zcmMvcGVyZm9ybWFuY2UudHMiLCJ3ZWJwYWNrOi8vLy4vc3JjL2FwcC50cyJdLCJuYW1lcyI6WyJpbnN0YWxsZWRNb2R1bGVzIiwiX193ZWJwYWNrX3JlcXVpcmVfXyIsIm1vZHVsZUlkIiwiZXhwb3J0cyIsIm1vZHVsZSIsImkiLCJsIiwibW9kdWxlcyIsImNhbGwiLCJtIiwiYyIsImQiLCJuYW1lIiwiZ2V0dGVyIiwibyIsIk9iamVjdCIsImRlZmluZVByb3BlcnR5IiwiY29uZmlndXJhYmxlIiwiZW51bWVyYWJsZSIsImdldCIsInIiLCJ2YWx1ZSIsIm4iLCJfX2VzTW9kdWxlIiwib2JqZWN0IiwicHJvcGVydHkiLCJwcm90b3R5cGUiLCJoYXNPd25Qcm9wZXJ0eSIsInAiLCJzIiwiZGVmYXVsdCIsIm1hcCIsInNlbnRlbmNlIiwicmVwbGFjZSIsInRvTG93ZXJDYXNlIiwiaXNGaXJlZm94IiwidGVzdCIsIndpbmRvdyIsIm5hdmlnYXRvciIsInVzZXJBZ2VudCIsInRzIiwidW5kZWZpbmVkIiwicGVyZm9ybWFuY2UiLCJEYXRlIiwibm93IiwiZGV0ZWN0Iiwic2NvcGUiLCJtZWFzdXJlIiwidGltZXMiLCJzdGFydCIsIngiLCJidXN5IiwibWVhbiIsInJlc3VsdCIsInZhbHVlcyIsInB1c2giLCJkZWx0YXMiLCJsZW5ndGgiLCJjdXIiLCJwcmV2IiwiZXhwZWN0ZWQiLCJkZWx0YSIsIk1hdGgiLCJhYnMiLCJzZW50ZW5jZXNfMSIsIkxTX0tFWSIsIlJFQVNTVVJFIiwiUkVBU1NVUkVfRVZFUlkiLCJmbG9vciIsIkFwcGxpY2F0aW9uIiwiX3RoaXMiLCJ0aGlzIiwibG9nIiwiZWxlbXMiLCJkaXNwbGF5IiwiZG9jdW1lbnQiLCJnZXRFbGVtZW50QnlJZCIsImNvdW50ZXIiLCJ3cmFwIiwicmVhc3N1cmUiLCJzZW50ZW5jZUluZGV4IiwiY2hhckluZGV4IiwibGFzdFJlYXNzdXJlIiwibG9jYWxTdG9yYWdlIiwiZ2V0SXRlbSIsImNvbXBsZXRlIiwiZGlzcGxheVNlbnRlbmNlIiwiYWRkRXZlbnRMaXN0ZW5lciIsImUiLCJvbktleURvd24iLCJrZXkiLCJvbktleVVwIiwidGV4dENvbnRlbnQiLCJ0b1N0cmluZyIsImlubmVySFRNTCIsInNsaWNlIiwibmV4dFNlbnRlbmNlIiwic2hpZnQiLCJzYXZlIiwiZXJyIiwiY29kZSIsImVycm9yIiwiayIsInNldFRpbWVvdXQiLCJjYWxsYmFjayIsImpzb24iLCJKU09OIiwic3RyaW5naWZ5IiwieGhyIiwiWE1MSHR0cFJlcXVlc3QiLCJvbmxvYWQiLCJyZXNwb25zZSIsInBhcnNlIiwicmVzcG9uc2VUZXh0IiwiRXJyb3IiLCJvbmVycm9yIiwib3BlbiIsInNldFJlcXVlc3RIZWFkZXIiLCJzZW5kIiwic2V0SXRlbSJdLCJtYXBwaW5ncyI6ImFBQ0EsSUFBQUEsS0FHQSxTQUFBQyxFQUFBQyxHQUdBLEdBQUFGLEVBQUFFLEdBQ0EsT0FBQUYsRUFBQUUsR0FBQUMsUUFHQSxJQUFBQyxFQUFBSixFQUFBRSxJQUNBRyxFQUFBSCxFQUNBSSxHQUFBLEVBQ0FILFlBVUEsT0FOQUksRUFBQUwsR0FBQU0sS0FBQUosRUFBQUQsUUFBQUMsSUFBQUQsUUFBQUYsR0FHQUcsRUFBQUUsR0FBQSxFQUdBRixFQUFBRCxRQUtBRixFQUFBUSxFQUFBRixFQUdBTixFQUFBUyxFQUFBVixFQUdBQyxFQUFBVSxFQUFBLFNBQUFSLEVBQUFTLEVBQUFDLEdBQ0FaLEVBQUFhLEVBQUFYLEVBQUFTLElBQ0FHLE9BQUFDLGVBQUFiLEVBQUFTLEdBQ0FLLGNBQUEsRUFDQUMsWUFBQSxFQUNBQyxJQUFBTixLQU1BWixFQUFBbUIsRUFBQSxTQUFBakIsR0FDQVksT0FBQUMsZUFBQWIsRUFBQSxjQUFpRGtCLE9BQUEsS0FJakRwQixFQUFBcUIsRUFBQSxTQUFBbEIsR0FDQSxJQUFBUyxFQUFBVCxLQUFBbUIs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bEIsT0FBS3VCLEVBQVNYLEtBSVBJLE9BQVMvRCxFQUFXc0UsRUFBU1gsTUFIM0JJLEVBQVMsSUFBSVUsTUFBTSxzQkFNOUJOLEVBQUlPLFFBQVUsU0FBQ2hCLEdBQ2IsT0FBT0ssRUFBUyxJQUFJVSxNQUFNLGVBRzVCTixFQUFJUSxLQUFLLE1BcEpRLHNDQW9KYSxHQUM5QlIsRUFBSVMsaUJBQWlCLGVBQWdCLG9CQUNyQ1QsRUFBSVUsS0FBS2IsSUFHWHBDLEVBQUEzQyxVQUFBMkQsU0FBQSxTQUFTZSxHQUNIL0QsT0FBTzhDLGNBQ1Q5QyxPQUFPOEMsYUFBYW9DLFFBQVF0RCxFQUFRLGFBRXRDLElBQU1hLEVBQU9QLEtBQUtFLE1BQU1LLEtBRXRCQSxFQUFLZ0IsZUFETXJELElBQVQyRCxFQUNlLDRDQUVBLCtFQUN1QkEsRUFBSSxTQUloRC9CLEVBQUEzQyxVQUFBMkUsTUFBQSxXQUNFOUIsS0FBS0UsTUFBTUssS0FBS2dCLFVBQVksOENBRWhDekIsRUF0SkEiLCJmaWxlIjoiYnVuZGxlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiIFx0Ly8gVGhlIG1vZHVsZSBjYWNoZVxuIFx0dmFyIGluc3RhbGxlZE1vZHVsZXMgPSB7fTtcblxuIFx0Ly8gVGhlIHJlcXVpcmUgZnVuY3Rpb25cbiBcdGZ1bmN0aW9uIF9fd2VicGFja19yZXF1aXJlX18obW9kdWxlSWQpIHtcblxuIFx0XHQvLyBDaGVjayBpZiBtb2R1bGUgaXMgaW4gY2FjaGVcbiBcdFx0aWYoaW5zdGFsbGVkTW9kdWxlc1ttb2R1bGVJZF0pIHtcbiBcdFx0XHRyZXR1cm4gaW5zdGFsbGVkTW9kdWxlc1ttb2R1bGVJZF0uZXhwb3J0cztcbiBcdFx0fVxuIFx0XHQvLyBDcmVhdGUgYSBuZXcgbW9kdWxlIChhbmQgcHV0IGl0IGludG8gdGhlIGNhY2hlKVxuIFx0XHR2YXIgbW9kdWxlID0gaW5zdGFsbGVkTW9kdWxlc1ttb2R1bGVJZF0gPSB7XG4gXHRcdFx0aTogbW9kdWxlSWQsXG4gXHRcdFx0bDogZmFsc2UsXG4gXHRcdFx0ZXhwb3J0czoge31cbiBcdFx0fTtcblxuIFx0XHQvLyBFeGVjdXRlIHRoZSBtb2R1bGUgZnVuY3Rpb25cbiBcdFx0bW9kdWxlc1ttb2R1bGVJZF0uY2FsbChtb2R1bGUuZXhwb3J0cywgbW9kdWxlLCBtb2R1bGUuZXhwb3J0cywgX193ZWJwYWNrX3JlcXVpcmVfXyk7XG5cbiBcdFx0Ly8gRmxhZyB0aGUgbW9kdWxlIGFzIGxvYWRlZFxuIFx0XHRtb2R1bGUubCA9IHRydWU7XG5cbiBcdFx0Ly8gUmV0dXJuIHRoZSBleHBvcnRzIG9mIHRoZSBtb2R1bGVcbiBcdFx0cmV0dXJuIG1vZHVsZS5leHBvcnRzO1xuIFx0fVxuXG5cbiBcdC8vIGV4cG9zZSB0aGUgbW9kdWxlcyBvYmplY3QgKF9fd2VicGFja19tb2R1bGVzX18pXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm0gPSBtb2R1bGVzO1xuXG4gXHQvLyBleHBvc2UgdGhlIG1vZHVsZSBjYWNoZVxuIFx0X193ZWJwYWNrX3JlcXVpcmVfXy5jID0gaW5zdGFsbGVkTW9kdWxlcztcblxuIFx0Ly8gZGVmaW5lIGdldHRlciBmdW5jdGlvbiBmb3IgaGFybW9ueSBleHBvcnRzXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLmQgPSBmdW5jdGlvbihleHBvcnRzLCBuYW1lLCBnZXR0ZXIpIHtcbiBcdFx0aWYoIV9fd2VicGFja19yZXF1aXJlX18ubyhleHBvcnRzLCBuYW1lKSkge1xuIFx0XHRcdE9iamVjdC5kZWZpbmVQcm9wZXJ0eShleHBvcnRzLCBuYW1lLCB7XG4gXHRcdFx0XHRjb25maWd1cmFibGU6IGZhbHNlLFxuIFx0XHRcdFx0ZW51bWVyYWJsZTogdHJ1ZSxcbiBcdFx0XHRcdGdldDogZ2V0dGVyXG4gXHRcdFx0fSk7XG4gXHRcdH1cbiBcdH07XG5cbiBcdC8vIGRlZmluZSBfX2VzTW9kdWxlIG9uIGV4cG9ydHNcbiBcdF9fd2VicGFja19yZXF1aXJlX18uciA9IGZ1bmN0aW9uKGV4cG9ydHMpIHtcbiBcdFx0T2JqZWN0LmRlZmluZVByb3BlcnR5KGV4cG9ydHMsICdfX2VzTW9kdWxlJywgeyB2YWx1ZTogdHJ1ZSB9KTtcbiBcdH07XG5cbiBcdC8vIGdldERlZmF1bHRFeHBvcnQgZnVuY3Rpb24gZm9yIGNvbXBhdGliaWxpdHkgd2l0aCBub24taGFybW9ueSBtb2R1bGVzXG4gXHRfX3dlYnBhY2tfcmVxdWlyZV9fLm4gPSBmdW5jdGlvbihtb2R1bGUpIHtcbiBcdFx0dmFyIGdldHRlciA9IG1vZHVsZSAmJiBtb2R1bGUuX19lc01vZHVsZSA/XG4gXHRcdFx0ZnVuY3Rpb24gZ2V0RGVmYXVsdCgpIHsgcmV0dXJuIG1vZHVsZVsnZGVmYXVsdCddOyB9IDpcbiBcdFx0XHRmdW5jdGlvbiBnZXRNb2R1bGVFeHBvcnRzKCkgeyByZXR1cm4gbW9kdWxlOyB9O1xuIFx0XHRfX3dlYnBhY2tfcmVxdWlyZV9fLmQoZ2V0dGVyLCAnYScsIGdldHRlcik7XG4gXHRcdHJldHVybiBnZXR0ZXI7XG4gXHR9O1xuXG4gXHQvLyBPYmplY3QucHJvdG90eXBlLmhhc093blByb3BlcnR5LmNhbGxcbiBcdF9fd2VicGFja19yZXF1aXJlX18ubyA9IGZ1bmN0aW9uKG9iamVjdCwgcHJvcGVydHkpIHsgcmV0dXJuIE9iamVjdC5wcm90b3R5cGUuaGFzT3duUHJvcGVydHkuY2FsbChvYmplY3QsIHByb3BlcnR5KTsgfTtcblxuIFx0Ly8gX193ZWJwYWNrX3B1YmxpY19wYXRoX19cbiBcdF9fd2VicGFja19yZXF1aXJlX18ucCA9IFwiXCI7XG5cblxuIFx0Ly8gTG9hZCBlbnRyeSBtb2R1bGUgYW5kIHJldHVybiBleHBvcnRzXG4gXHRyZXR1cm4gX193ZWJwYWNrX3JlcXVpcmVfXyhfX3dlYnBhY2tfcmVxdWlyZV9fLnMgPSAyKTtcbiIsImNvbnN0IHNlbnRlbmNlcyA9IFtcbiAgXCJJdCBzZWVtZWQgdG8gaGltIHRoYXQgbWlzZm9ydHVuZSBmb2xsb3dlZCB3aGVyZXZlciBoZSB3ZW50XCIsXG4gIFwiVGhlbiBoZSB0aHJldyBoaXMgaGVhZCBiYWNrIGFuZCBsYXVnaGVkXCIsXG4gIFwiSGUgaGFpbGVkIGl0IGFuZCBpbiBhIGxvdyB2b2ljZSBnYXZlIHRoZSBkcml2ZXIgYW4gYWRkcmVzc1wiLFxuICBcIkkgc2VlIHRoaW5ncyBkaWZmZXJlbnRseSwgSSB0aGluayBvZiB0aGVtIGRpZmZlcmVudGx5XCIsXG4gIFwiSSBhbSBmb3JjZWQgdG8gYnJpbmcgeW91IGludG8gdGhlIG1hdHRlclwiLFxuICBcIkEgYml0dGVyIGJsYXN0IHN3ZXB0IGFjcm9zcyB0aGUgc3F1YXJlXCIsXG4gIFwiSSB0aG91Z2h0IHRoYXQgSSB3YXMgZ29pbmcgdG8gYmUgd29uZGVyZnVsXCIsXG4gIFwiWW91IGxhdWdoLCBidXQgSSB0ZWxsIHlvdSBzaGUgaGFzIGdlbml1c1wiLFxuICBcIkkga25vdyBub3cgdGhhdCBJIHdhcyBxdWl0ZSByaWdodCBpbiB3aGF0IEkgZmFuY2llZCBhYm91dCBoaW1cIixcbiAgXCJUaG9zZSB3aG8gZ28gYmVuZWF0aCB0aGUgc3VyZmFjZSBkbyBzbyBhdCB0aGVpciBwZXJpbFwiLFxuICBcIlRvIGJlIGluIGxvdmUgaXMgdG8gc3VycGFzcyBvbmUncyBzZWxmXCIsXG4gIFwiU2hlIG5ldmVyIGdldHMgY29uZnVzZWQgb3ZlciBoZXIgZGF0ZXMsIGFuZCBJIGFsd2F5cyBkb1wiLFxuICBcIkFuZCB5ZXQgdGhlIHRoaW5nIHdvdWxkIHN0aWxsIGxpdmUgb25cIixcbiAgXCJTaGUgdGVsbHMgbWUgc2hlIGlzIGdvaW5nIGRvd24gdG8gU2VsYnlcIixcbiAgXCJJdCBpcyBteSBtYXN0ZXJwaWVjZSBhcyBpdCBzdGFuZHNcIixcbiAgXCJCdXQgSSBhbSBtdWNoIG9ibGlnZWQgZm9yIHRoZSBjb21wbGltZW50LCBhbGwgdGhlIHNhbWVcIixcbiAgXCJJdCBtaWdodCBiZSBhIG1vc3QgYnJpbGxpYW50IG1hcnJpYWdlIGZvciBTaWJ5bFwiLFxuICBcIk1vc3Qgb2YgdGhlIHNlcnZhbnRzIHdlcmUgYXQgU2VsYnkgUm95YWxcIixcbiAgXCJUaGUgdWdseSBhbmQgdGhlIHN0dXBpZCBoYXZlIHRoZSBiZXN0IG9mIGl0IGluIHRoaXMgd29ybGRcIixcbiAgXCJTdWNjZXNzIHdhcyBnaXZlbiB0byB0aGUgc3Ryb25nLCBmYWlsdXJlIHRocnVzdCB1cG9uIHRoZSB3ZWFrXCIsXG4gIFwiSGUgY2FtZSBjbG9zZSB0byBoaW0gYW5kIHB1dCBoaXMgaGFuZCB1cG9uIGhpcyBzaG91bGRlclwiLFxuICBcIkhlciBsb3ZlIHdhcyB0cmVtYmxpbmcgaW4gbGF1Z2h0ZXIgb24gaGVyIGxpcHNcIixcbiAgXCJJdCB3aWxsIGJlIGEgZ3JlYXQgcGl0eSwgZm9yIGl0IHdpbGwgYWx0ZXIgeW91XCIsXG4gIFwiSXQgaXMgdG9vIHVnbHksIHRvbyBob3JyaWJsZSwgdG9vIGRpc3RyZXNzaW5nXCIsXG4gIFwiSSBjYW1lIGhlcmUgYXQgb25jZSBhbmQgd2FzIG1pc2VyYWJsZSBhdCBub3QgZmluZGluZyB5b3VcIixcbiAgXCJUaGUga25vY2tpbmcgc3RpbGwgY29udGludWVkIGFuZCBncmV3IGxvdWRlclwiLFxuICBcIkhlciBjb21wYW5pb24gd2F0Y2hlZCBoZXIgZW52aW91c2x5XCIsXG4gIFwiTm93LCB3aGVyZXZlciB5b3UgZ28sIHlvdSBjaGFybSB0aGUgd29ybGRcIixcbiAgXCJIZSBsaXQgYSBjaWdhcmV0dGUgYW5kIHRoZW4gdGhyZXcgaXQgYXdheVwiLFxuICBcIlRoZSBsYWQgbGlzdGVuZWQgc3Vsa2lseSB0byBoZXIgYW5kIG1hZGUgbm8gYW5zd2VyXCIsXG4gIFwiR3JheSB0byB3YWl0LCBQYXJrZXI6IEkgc2hhbGwgYmUgaW4gaW4gYSBmZXcgbW9tZW50c1wiLFxuICBcIlllczogaXQgd2FzIGNlcnRhaW5seSBhIHRlZGlvdXMgcGFydHlcIixcbiAgXCJUaGVyZSB3ZXJlIHRlYXJzIGluIGhpcyBleWVzIGFzIGhlIHdlbnQgZG93bnN0YWlyc1wiLFxuICBcIkhlIHdvdWxkIG5ldmVyIGJyaW5nIG1pc2VyeSB1cG9uIGFueSBvbmVcIixcbiAgXCJJdCB3YXMgb2YgaGltc2VsZiwgYW5kIG9mIGhpcyBvd24gZnV0dXJlLCB0aGF0IGhlIGhhZCB0byB0aGlua1wiLFxuICBcIlRoZSBtYW4gbG9va2VkIGF0IGhlciBpbiB0ZXJyb3IgYW5kIGJlZ2FuIHRvIHdoaW1wZXJcIixcbiAgXCJBbmQgdGhleSBwYXNzZWQgaW50byB0aGUgZGluaW5nLXJvb21cIixcbiAgXCJZb3UgZ2F2ZSBoZXIgZ29vZCBhZHZpY2UgYW5kIGJyb2tlIGhlciBoZWFydFwiLFxuICBcIlRoZXkgYXJlIGhvcnJpYmxlLCBhbmQgdGhleSBkb24ndCBtZWFuIGFueXRoaW5nXCIsXG4gIFwiVGhleSBoYXZlIGhhZCBteSBvd24gZGl2b3JjZS1jYXNlIGFuZCBBbGFuIENhbXBiZWxsJ3Mgc3VpY2lkZVwiLFxuICBcIk1ha2UgbXkgZXhjdXNlcyB0byBMYWR5IE5hcmJvcm91Z2hcIixcbiAgXCJIYXJyeSwgdG8gd2hvbSBJIHRhbGtlZCBhYm91dCBpdCwgbGF1Z2hlZCBhdCBtZVwiLFxuICBcIkEgZGVjZW50LWxvb2tpbmcgbWFuLCBzaXIsIGJ1dCByb3VnaC1saWtlXCIsXG4gIFwiRG9yaWFuIGxvb2tlZCBhdCBoaW0gZm9yIGEgbW9tZW50XCIsXG4gIFwiSW4gb25lIG9mIHRoZSB0b3Atd2luZG93cyBzdG9vZCBhIGxhbXBcIixcbiAgXCJUaGVyZSBoYWQgYmVlbiBhIG1hZG5lc3Mgb2YgbXVyZGVyIGluIHRoZSBhaXJcIixcbiAgXCJPZiBjb3Vyc2UsIEkgYW0gdmVyeSBmb25kIG9mIEhhcnJ5XCIsXG4gIFwiVGhlIGFydGlzdCBpcyB0aGUgY3JlYXRvciBvZiBiZWF1dGlmdWwgdGhpbmdzXCIsXG4gIFwiVGhhdCBpcyBhIGdyZWF0IGFkdmFudGFnZSwgZG9uJ3QgeW91IHRoaW5rIHNvLCBNclwiLFxuICBcIkVyc2tpbmUsIGFuIGFic29sdXRlbHkgcmVhc29uYWJsZSBwZW9wbGVcIixcbiAgXCJJIGFtIGFuYWx5c2luZyB3b21lbiBhdCBwcmVzZW50LCBzbyBJIG91Z2h0IHRvIGtub3dcIixcbiAgXCJIZSBmZWx0IGFzIGlmIHRoZSBsb2FkIGhhZCBiZWVuIGxpZnRlZCBmcm9tIGhpbSBhbHJlYWR5XCIsXG4gIFwiQnV0IGhlcmUgd2FzIGEgdmlzaWJsZSBzeW1ib2wgb2YgdGhlIGRlZ3JhZGF0aW9uIG9mIHNpblwiLFxuICBcIlRoZSBsaXR0bGUgZHVjaGVzcyBpcyBxdWl0ZSBkZXZvdGVkIHRvIHlvdVwiLFxuICBcIlRoZXkgYXJlIHRoZSBlbGVjdCB0byB3aG9tIGJlYXV0aWZ1bCB0aGluZ3MgbWVhbiBvbmx5IGJlYXV0eVwiLFxuICBcIkJ1dCBJIHdpc2ggeW91IGhhZCBsZWZ0IHdvcmQgd2hlcmUgeW91IGhhZCByZWFsbHkgZ29uZSB0b1wiLFxuICBcIkkga25vdyB5b3UgYXJlIHN1cnByaXNlZCBhdCBteSB0YWxraW5nIHRvIHlvdSBsaWtlIHRoaXNcIixcbiAgXCJUaGUgc3VubGlnaHQgc2xpcHBlZCBvdmVyIHRoZSBwb2xpc2hlZCBsZWF2ZXNcIixcbiAgXCJBIGN1cmlvdXMgc2Vuc2F0aW9uIG9mIHRlcnJvciBjYW1lIG92ZXIgbWVcIixcbiAgXCJIZSB3YXMgaGVhcnQtc2ljayBhdCBsZWF2aW5nIGhvbWVcIixcbiAgXCJQZW9wbGUgc2F5IHNvbWV0aW1lcyB0aGF0IGJlYXV0eSBpcyBvbmx5IHN1cGVyZmljaWFsXCIsXG4gIFwiSXQgd2FzIG1lcmVseSB0aGUgbmFtZSBtZW4gZ2F2ZSB0byB0aGVpciBtaXN0YWtlc1wiLFxuICBcIkxvcmQgSGVucnkgc2hydWdnZWQgaGlzIHNob3VsZGVyc1wiLFxuICBcIkl0IHdhcyB0aGUgbW9zdCBwcmVtYXR1cmUgZGVmaW5pdGlvbiBldmVyIGdpdmVuXCIsXG4gIFwiSXQgd2lsbCBiZSBhIGdyZWF0IHBpdHksIGZvciBpdCB3aWxsIGFsdGVyIHlvdVwiLFxuICBcIkl0IGhhZCBiZWVuIGdpdmVuIHRvIGhpbSBieSBBZHJpYW4gU2luZ2xldG9uXCIsXG4gIFwiQXQgbGFzdCBoZSBoZWFyZCBhIHN0ZXAgb3V0c2lkZSwgYW5kIHRoZSBkb29yIG9wZW5lZFwiLFxuICBcIlRoZXkgaGF2ZSBhIHJpZ2h0IHRvIGRlbWFuZCBpdCBiYWNrXCIsXG4gIFwiVGhlcmUgaXMgbm8gbXlzdGVyeSBpbiBhbnkgb2YgdGhlbVwiLFxuICBcIldoZW4gTG9yZCBIZW5yeSBoYWQgc2F0IGRvd24gYWdhaW4sIE1yXCIsXG4gIFwiQmVzaWRlcywgaW5kaXZpZHVhbGlzbSBoYXMgcmVhbGx5IHRoZSBoaWdoZXIgYWltXCIsXG4gIFwiWW91IGZpbmQgbWUgY29uc29sZWQsIGFuZCB5b3UgYXJlIGZ1cmlvdXNcIixcbiAgXCJUaGlzIG9uZSBpcyBsaXR0bGUgbW9yZSB0aGFuIGEgYm95XCIsXG4gIFwiQ29tZSB0byB0aGUgY2x1YiB3aXRoIEJhc2lsIGFuZCBteXNlbGZcIixcbiAgXCJUaGUgbWFuIHdobyBoYWQgYmVlbiBzaG90IGluIHRoZSB0aGlja2V0IHdhcyBKYW1lcyBWYW5lXCIsXG4gIFwiT25lIGNvdWxkIGhlYXIgaGVyIHNpbmdpbmcgYXMgc2hlIHJhbiB1cHN0YWlyc1wiLFxuICBcIllvdSB3ZXJlIHRoZSBtb3N0IHVuc3BvaWxlZCBjcmVhdHVyZSBpbiB0aGUgd2hvbGUgd29ybGRcIixcbiAgXCJJdCBpcyBzbyBtdWNoIG1vcmUgcmVhbCB0aGFuIGxpZmVcIixcbiAgXCJDb21lIGFuZCBzZWUgbWUgc29tZSBhZnRlcm5vb24gaW4gQ3Vyem9uIFN0cmVldFwiLFxuICBcIldoZW4gdGhlIHZlcml0aWVzIGJlY29tZSBhY3JvYmF0cywgd2UgY2FuIGp1ZGdlIHRoZW1cIixcbiAgXCJJc2FhY3MgaGFzIGJlZW4gdmVyeSBnb29kIHRvIHVzLCBhbmQgd2Ugb3dlIGhpbSBtb25leVwiLFxuICBcIkluZGVlZCwgaW4gc29tZSBtZWFzdXJlIGl0IHdhcyBhIGRpc2FwcG9pbnRtZW50IHRvIGhlclwiLFxuICBcIkl0IGlzIHRoZSBvbmx5IHdheSBJIGdldCB0byBrbm93IG9mIHRoZW1cIixcbiAgXCJZZWFycyBhZ28gaGUgd2FzIGNocmlzdGVuZWQgUHJpbmNlIENoYXJtaW5nXCIsXG4gIFwiVGhlIG1hbiB3aG8gaGFkIGJlZW4gc2hvdCBpbiB0aGUgdGhpY2tldCB3YXMgSmFtZXMgVmFuZVwiLFxuICBcIkhlIHdhcyBub3QgdG8gZ28gdG8gdGhlIGdvbGQtZmllbGRzIGF0IGFsbFwiLFxuICBcIkFueXRoaW5nIHdvdWxkIGJlIGJldHRlciB0aGFuIHRoaXMgZHJlYWRmdWwgc3RhdGUgb2YgZG91YnRcIixcbiAgXCJJdCBoYXMgYSBwZXJmZWN0IGhvc3QsIGFuZCBhIHBlcmZlY3QgbGlicmFyeVwiLFxuICBcIkl0IGlzIHRoZSBmZWV0IG9mIGNsYXkgdGhhdCBtYWtlIHRoZSBnb2xkIG9mIHRoZSBpbWFnZSBwcmVjaW91c1wiXG5dO1xuXG5leHBvcnQgZGVmYXVsdCBzZW50ZW5jZXMubWFwKChzZW50ZW5jZSkgPT4ge1xuICByZXR1cm4gc2VudGVuY2UucmVwbGFjZSgvXFxzL2csICfikKMnKS50b0xvd2VyQ2FzZSgpO1xufSk7XG4iLCJjb25zdCBpc0ZpcmVmb3ggPSAvZmlyZWZveC9pLnRlc3Qod2luZG93Lm5hdmlnYXRvci51c2VyQWdlbnQpO1xuXG5jb25zdCB0cyA9IChpc0ZpcmVmb3ggfHwgd2luZG93LnBlcmZvcm1hbmNlID09PSB1bmRlZmluZWQpID8gKCkgPT4gRGF0ZS5ub3coKSA6XG4gICgpID0+IHdpbmRvdy5wZXJmb3JtYW5jZS5ub3coKTtcblxuZXhwb3J0IGZ1bmN0aW9uIG5vdygpOiBudW1iZXIge1xuICByZXR1cm4gdHMoKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRldGVjdCgpOiBib29sZWFuIHtcbiAgaWYgKCFpc0ZpcmVmb3gpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cblxuICBjb25zdCBzY29wZTogYW55ID0ge307XG5cbiAgZnVuY3Rpb24gYnVzeSh0aW1lczogbnVtYmVyKTogdm9pZCB7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aW1lczsgaSsrKSB7XG4gICAgICBpZiAoc2NvcGUueCA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICAgIHNjb3BlLnggPSAwO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgc2NvcGUueCsrO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIGZ1bmN0aW9uIG1lYXN1cmUodGltZXM6IG51bWJlcik6IG51bWJlciB7XG4gICAgY29uc3Qgc3RhcnQgPSBub3coKTtcbiAgICBidXN5KHRpbWVzKTtcbiAgICByZXR1cm4gbm93KCkgLSBzdGFydDtcbiAgfVxuXG4gIGZ1bmN0aW9uIG1lYW4odGltZXM6IG51bWJlcik6IG51bWJlciB7XG4gICAgbGV0IHJlc3VsdDogbnVtYmVyID0gMDtcbiAgICBjb25zdCBjb3VudCA9IDEwMDtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGNvdW50OyBpKyspIHtcbiAgICAgIHJlc3VsdCArPSBtZWFzdXJlKHRpbWVzKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdCAvPSBjb3VudDtcbiAgfVxuXG4gIGNvbnN0IG11bCA9IDEuMTtcblxuICBjb25zdCB2YWx1ZXMgPSBbXTtcbiAgZm9yIChsZXQgaSA9IDE7IGkgPCAzMzU1NDQzMjsgaSAqPSBtdWwpIHtcbiAgICBjb25zdCBtID0gbWVhbihpKTtcbiAgICB2YWx1ZXMucHVzaChtKTtcbiAgICBpZiAobSA+IDEpIHtcbiAgICAgIGJyZWFrO1xuICAgIH1cbiAgfVxuXG4gIGNvbnN0IGRlbHRhczogbnVtYmVyW10gPSBbXTtcbiAgZm9yIChsZXQgaSA9IHZhbHVlcy5sZW5ndGggLSAxOyBpID49IDE7IGktLSkge1xuICAgIGNvbnN0IGN1ciA9IHZhbHVlc1tpXTtcbiAgICBjb25zdCBwcmV2ID0gdmFsdWVzW2kgLSAxXTtcblxuICAgIGNvbnN0IGV4cGVjdGVkID0gY3VyIC8gbXVsO1xuICAgIGNvbnN0IGRlbHRhID0gTWF0aC5hYnMoZXhwZWN0ZWQgLSBwcmV2KSAvIChleHBlY3RlZCArIDFlLTI0KTtcblxuICAgIGlmIChkZWx0YSA+IDEpIHtcbiAgICAgIGJyZWFrO1xuICAgIH1cblxuICAgIGRlbHRhcy5wdXNoKGRlbHRhKTtcbiAgfVxuXG4gIGNvbnN0IHBlcmNlbnQgPSBkZWx0YXMubGVuZ3RoIC8gKHZhbHVlcy5sZW5ndGggLSAxKTtcblxuICAvLyBDb21wbGV0ZWx5IGFyYml0cmFyeVxuICByZXR1cm4gcGVyY2VudCA+IDAuNTtcbn1cbiIsImltcG9ydCAqIGFzIHBlcmZvcm1hbmNlIGZyb20gJy4vcGVyZm9ybWFuY2UnO1xuaW1wb3J0IHsgZGVmYXVsdCBhcyBzZW50ZW5jZXMgfSBmcm9tICcuL3NlbnRlbmNlcyc7XG5cbmNvbnN0IEFQSV9FTkRQT0lOVCA9ICdodHRwczovL2dyYWR0eXBlLXN1cnZleS5kYXJrc2kuZGUvJztcbmNvbnN0IExTX0tFWSA9ICdncmFkdHlwZS1zdXJ2ZXktdjEnO1xuXG5jb25zdCBSRUFTU1VSRTogc3RyaW5nW10gPSBbXG4gICdZb3VcXCdyZSBkb2luZyBncmVhdCEnLFxuICAnSnVzdCBmZXcgbW9yZSEnLFxuICAnQWxtb3N0IHRoZXJlISdcbl07XG5cbmNvbnN0IFJFQVNTVVJFX0VWRVJZID0gTWF0aC5mbG9vcihzZW50ZW5jZXMubGVuZ3RoKSAvIChSRUFTU1VSRS5sZW5ndGggKyAxKTtcblxudHlwZSBMb2dLaW5kID0gJ2QnIHwgJ3UnO1xuXG50eXBlIExvZ0V2ZW50ID0ge1xuICByZWFkb25seSBlOiBMb2dLaW5kO1xuICByZWFkb25seSB0czogbnVtYmVyO1xuICByZWFkb25seSBrOiBzdHJpbmc7XG59IHwgJ3InO1xuXG5jbGFzcyBBcHBsaWNhdGlvbiB7XG4gIHByaXZhdGUgcmVhZG9ubHkgbG9nOiBMb2dFdmVudFtdID0gW107XG4gIHByaXZhdGUgcmVhZG9ubHkgc3RhcnQ6IG51bWJlciA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBwcml2YXRlIHJlYWRvbmx5IGVsZW1zID0ge1xuICAgIGRpc3BsYXk6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdkaXNwbGF5JykhLFxuICAgIGNvdW50ZXI6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCdjb3VudGVyJykhLFxuICAgIHdyYXA6IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKCd3cmFwJykhLFxuICAgIHJlYXNzdXJlOiBkb2N1bWVudC5nZXRFbGVtZW50QnlJZCgncmVhc3N1cmUnKSEsXG4gIH07XG5cbiAgcHJpdmF0ZSBzZW50ZW5jZUluZGV4OiBudW1iZXIgPSAwO1xuICBwcml2YXRlIGNoYXJJbmRleDogbnVtYmVyID0gMDtcbiAgcHJpdmF0ZSBsYXN0UmVhc3N1cmU6IG51bWJlciA9IDA7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgaWYgKHdpbmRvdy5sb2NhbFN0b3JhZ2UgJiYgd2luZG93LmxvY2FsU3RvcmFnZS5nZXRJdGVtKExTX0tFWSkpIHtcbiAgICAgIHRoaXMuY29tcGxldGUoKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICB0aGlzLmRpc3BsYXlTZW50ZW5jZSgpO1xuXG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoJ2tleWRvd24nLCAoZSkgPT4ge1xuICAgICAgdGhpcy5vbktleURvd24oZS5rZXkpO1xuICAgIH0sIHRydWUpO1xuXG4gICAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoJ2tleXVwJywgKGUpID0+IHtcbiAgICAgIHRoaXMub25LZXlVcChlLmtleSk7XG4gICAgfSwgdHJ1ZSk7XG4gIH1cblxuICBkaXNwbGF5U2VudGVuY2UoKSB7XG4gICAgY29uc3Qgc2VudGVuY2UgPSBzZW50ZW5jZXNbdGhpcy5zZW50ZW5jZUluZGV4XTtcblxuICAgIHRoaXMuZWxlbXMuY291bnRlci50ZXh0Q29udGVudCA9XG4gICAgICAoc2VudGVuY2VzLmxlbmd0aCAtIHRoaXMuc2VudGVuY2VJbmRleCkudG9TdHJpbmcoKTtcbiAgICB0aGlzLmVsZW1zLmRpc3BsYXkuaW5uZXJIVE1MID1cbiAgICAgICc8c3BhbiBjbGFzcz1zZW50ZW5jZS1jb21wbGV0ZWQ+JyArXG4gICAgICBzZW50ZW5jZS5zbGljZSgwLCB0aGlzLmNoYXJJbmRleCkgK1xuICAgICAgJzwvc3Bhbj4nICtcbiAgICAgICc8c3BhbiBjbGFzcz1zZW50ZW5jZS1wZW5kaW5nPicgK1xuICAgICAgc2VudGVuY2Uuc2xpY2UodGhpcy5jaGFySW5kZXgpXG4gICAgICAnPC9zcGFuPic7XG4gIH1cblxuICBuZXh0U2VudGVuY2UoKSB7XG4gICAgdGhpcy5jaGFySW5kZXggPSAwO1xuICAgIHRoaXMuc2VudGVuY2VJbmRleCsrO1xuICAgIHRoaXMubG9nLnB1c2goJ3InKTtcblxuICAgIGlmICh0aGlzLnNlbnRlbmNlSW5kZXggLSB0aGlzLmxhc3RSZWFzc3VyZSA+PSBSRUFTU1VSRV9FVkVSWSkge1xuICAgICAgdGhpcy5sYXN0UmVhc3N1cmUgPSB0aGlzLnNlbnRlbmNlSW5kZXg7XG4gICAgICB0aGlzLmVsZW1zLnJlYXNzdXJlLnRleHRDb250ZW50ID0gUkVBU1NVUkUuc2hpZnQoKSB8fCAnJztcbiAgICB9XG5cbiAgICBpZiAodGhpcy5zZW50ZW5jZUluZGV4ID09PSBzZW50ZW5jZXMubGVuZ3RoKSB7XG4gICAgICB0aGlzLmVsZW1zLmNvdW50ZXIudGV4dENvbnRlbnQgPSAnMCc7XG5cbiAgICAgIHRoaXMuc2F2ZSgoZXJyLCBjb2RlKSA9PiB7XG4gICAgICAgIGlmIChlcnIpIHtcbiAgICAgICAgICByZXR1cm4gdGhpcy5lcnJvcigpO1xuICAgICAgICB9XG4gICAgICAgIHRoaXMuY29tcGxldGUoY29kZSEpO1xuICAgICAgfSk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdGhpcy5kaXNwbGF5U2VudGVuY2UoKTtcbiAgfVxuXG4gIG9uS2V5RG93bihrZXk6IHN0cmluZykge1xuICAgIGNvbnN0IG5vdyA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICAgIHRoaXMubG9nLnB1c2goeyBlOiAnZCcsIHRzOiAobm93IC0gdGhpcy5zdGFydCkgLyAxMDAwLCBrOiBrZXkgfSk7XG5cbiAgICBpZiAodGhpcy5zZW50ZW5jZUluZGV4ID09PSBzZW50ZW5jZXMubGVuZ3RoKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3Qgc2VudGVuY2UgPSBzZW50ZW5jZXNbdGhpcy5zZW50ZW5jZUluZGV4XTtcbiAgICBjb25zdCBleHBlY3RlZCA9IHNlbnRlbmNlW3RoaXMuY2hhckluZGV4XTtcbiAgICBpZiAoa2V5ICE9PSBleHBlY3RlZCAmJiAhKGtleSA9PT0gJyAnICYmIGV4cGVjdGVkID09PSAn4pCjJykpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICB0aGlzLmNoYXJJbmRleCsrO1xuICAgIHRoaXMuZGlzcGxheVNlbnRlbmNlKCk7XG5cbiAgICBpZiAodGhpcy5jaGFySW5kZXggIT09IHNlbnRlbmNlLmxlbmd0aCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIC8vIEdpdmUgZW5vdWdoIHRpbWUgdG8gcmVjb3JkIHRoZSBsYXN0IGtleXN0cm9rZVxuICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgdGhpcy5uZXh0U2VudGVuY2UoKTtcbiAgICB9LCA1MCk7XG4gIH1cblxuICBvbktleVVwKGtleTogc3RyaW5nKSB7XG4gICAgY29uc3Qgbm93ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gICAgdGhpcy5sb2cucHVzaCh7IGU6ICd1JywgdHM6IChub3cgLSB0aGlzLnN0YXJ0KSAvIDEwMDAsIGs6IGtleSB9KTtcbiAgfVxuXG4gIHNhdmUoY2FsbGJhY2s6IChlcnI/OiBFcnJvciwgY29kZT86IHN0cmluZykgPT4gdm9pZCkge1xuICAgIGNvbnN0IGpzb24gPSBKU09OLnN0cmluZ2lmeSh0aGlzLmxvZyk7XG5cbiAgICB0aGlzLmVsZW1zLndyYXAuaW5uZXJIVE1MID1cbiAgICAgICc8aDE+VXBsb2FkaW5nLCBwbGVhc2UgZG8gbm90IGNsb3NlIHRoaXMgd2luZG93Li4uPC9oMT4nO1xuXG4gICAgY29uc3QgeGhyID0gbmV3IFhNTEh0dHBSZXF1ZXN0KCk7XG5cbiAgICB4aHIub25sb2FkID0gKCkgPT4ge1xuICAgICAgbGV0IHJlc3BvbnNlOiBhbnk7XG4gICAgICB0cnkge1xuICAgICAgICByZXNwb25zZSA9IEpTT04ucGFyc2UoeGhyLnJlc3BvbnNlVGV4dCk7XG4gICAgICB9IGNhdGNoIChlKSB7XG4gICAgICAgIHJldHVybiBjYWxsYmFjayhlKTtcbiAgICAgIH1cblxuICAgICAgaWYgKCFyZXNwb25zZS5jb2RlKSB7XG4gICAgICAgIHJldHVybiBjYWxsYmFjayhuZXcgRXJyb3IoJ05vIHJlc3BvbnNlIGNvZGUnKSk7XG4gICAgICB9XG5cbiAgICAgIHJldHVybiBjYWxsYmFjayh1bmRlZmluZWQsIHJlc3BvbnNlLmNvZGUpO1xuICAgIH07XG5cbiAgICB4aHIub25lcnJvciA9IChlcnIpID0+IHtcbiAgICAgIHJldHVybiBjYWxsYmFjayhuZXcgRXJyb3IoJ1hIUiBlcnJvcicpKTtcbiAgICB9O1xuXG4gICAgeGhyLm9wZW4oJ1BVVCcsIEFQSV9FTkRQT0lOVCwgdHJ1ZSk7XG4gICAgeGhyLnNldFJlcXVlc3RIZWFkZXIoJ0NvbnRlbnQtVHlwZScsICdhcHBsaWNhdGlvbi9qc29uJyk7XG4gICAgeGhyLnNlbmQoanNvbik7XG4gIH1cblxuICBjb21wbGV0ZShjb2RlPzogc3RyaW5nKSB7XG4gICAgaWYgKHdpbmRvdy5sb2NhbFN0b3JhZ2UpIHtcbiAgICAgIHdpbmRvdy5sb2NhbFN0b3JhZ2Uuc2V0SXRlbShMU19LRVksICdzdWJtaXR0ZWQnKTtcbiAgICB9XG4gICAgY29uc3Qgd3JhcCA9IHRoaXMuZWxlbXMud3JhcDtcbiAgICBpZiAoY29kZSA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICB3cmFwLmlubmVySFRNTCA9ICc8aDE+VGhhbmsgeW91IGZvciBzdWJtaXR0aW5nIHN1cnZleSE8L2gxPic7XG4gICAgfSBlbHNlIHtcbiAgICAgIHdyYXAuaW5uZXJIVE1MID0gJzxoMT5UaGFuayB5b3UgZm9yIHN1Ym1pdHRpbmcgc3VydmV5ITwvaDE+JyArXG4gICAgICAgIGA8aDEgc3R5bGU9XCJjb2xvcjpyZWRcIj5Zb3VyIGNvZGUgaXMgJHtjb2RlfTwvaDE+YDtcbiAgICB9XG4gIH1cblxuICBlcnJvcigpIHtcbiAgICB0aGlzLmVsZW1zLndyYXAuaW5uZXJIVE1MID0gJzxoMT5TZXJ2ZXIgZXJyb3IsIHBsZWFzZSByZXRyeSBsYXRlciE8L2gxPic7XG4gIH1cbn1cblxuY29uc3QgYXBwID0gbmV3IEFwcGxpY2F0aW9uKCk7XG4iXSwic291cmNlUm9vdCI6IiJ9