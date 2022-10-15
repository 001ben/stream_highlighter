<script>
import { onMount, onDestroy, afterUpdate, beforeUpdate, tick } from 'svelte';
import { animateScroll } from 'svelte-scrollto-element';

import { formatAsTs } from '$lib/util'
import TimeEntry from '$lib/TimeEntry.svelte'
import Chart from 'chart.js/auto';
import annotationPlugin from 'chartjs-plugin-annotation';
Chart.register(annotationPlugin)

let player;
let skipTimeArray = [1,2,5,10,30,60,120];
let skipTimeIdx = 3;
export let data;
const items = data.timestamps
const predictions = data.prediction_array;

// let items = [];
let isPaused = false
    
let chart;
let scrollContainer;
onMount(async() => {
    var options = {
        width: "100%",
        height: "100%",
        // channel: "<channel ID>",
        video: data.video_id,
        // collection: "<collection ID>",
        // only needed if your site is also embedded on embed.example.com and othersite.example.com
        parent: ["localhost"]
    };
    player = new Twitch.Player("player", options);
    isPaused = false
    
    // fetch('/api/data').then(e => e.json()).then(e => {
    //     console.log(e.labels)
    //     console.log(e.labels.map(convertTimestamp))
    //     items = e.labels.map(convertTimestamp)
    // })
    
    chart = new Chart('graph', {
        type: 'line',
        data: {
            labels: Array(predictions.length).fill().map((element, index) => index),
            datasets: [{
                data: predictions,
                backgroundColor: predictions.map(e => e <=0.5 ? '#1e90ff' : '#ef3038'),
                borderColor: '#ede9fe',
            }]
        },
        options: {
            animation: false,
            events: [],
            elements: {
                point: {
                    borderWidth: 0
                },
            },
            plugins: {
                legend: {
                    display: false
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            xMin: 60,
                            xMax: 60,
                            borderColor: 'rgb(255, 99, 132)',
                            borderWidth: 2,
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    display: false,
                    min:-0.1,
                    max:1
                },
                x: {
                    type: 'linear',
                    display: false, 
                    min: 0,
                    max: 1000,
                }
            }
        }
    });
})

let nextItemIdx = 0;
let needUpdate = false;
beforeUpdate(() => {
    if (scrollContainer && items) {
        let foundNextItemIdx = items.findIndex(x => x.start > currentTime)
        foundNextItemIdx = foundNextItemIdx > -1 ? foundNextItemIdx : items.length-1;
        if (foundNextItemIdx != nextItemIdx) {
            // console.log("marking update", foundNextItemIdx, "from", nextItemIdx)
            // console.log(nextItemIdx, foundNextItemIdx)
            if (nextItemIdx && nextItemIdx < items.length) {
                items[nextItemIdx].nextTime = false;
            }
            if (foundNextItemIdx && foundNextItemIdx < items.length) {
                items[foundNextItemIdx].nextTime = true;
            }
            nextItemIdx = foundNextItemIdx;
            needUpdate = true;
        }
    }
})
afterUpdate(() => {
    // scrollContainer.scrollTop+=1;
    // currScroll = scrollContainer.scrollTop;
    if (scrollContainer && items) {
        if (needUpdate) {
            let itemHeight = scrollContainer.getElementsByTagName("li")[0].scrollHeight
            // console.log("update idx", nextItemIdx)
            // console.log('needs update', scrollContainer.scrollTop, ",", (nextItemIdx - 1 * itemHeight) - 0.5*itemHeight)
            // console.log('', scrollContainer.scrollTop, ",", ((nextItemIdx - 1) * itemHeight) - 0.5*itemHeight)
            animateScroll.scrollTo({
                element: "#time-" + items[nextItemIdx].id,
                 container: 'ol',
                 offset: -2*itemHeight
                })
            // animateScroll.scrollTo(((nextItemIdx - 1) * itemHeight) - 0.5*itemHeight)
            needUpdate = false
        }
    }
})

function startStop() {
    if (player) {
        if (isPaused) {
            player.play()
            isPaused = false
        } else {
            player.pause()
            isPaused = true
        }
    }
}
let currentTime = 0;
let delayUpdate = 0;
let currScroll = 0;
function updateCurrentTime(new_time) {
    if (new_time) {
        currentTime = new_time
        delayUpdate = 2
    } else {
        if (delayUpdate>0) {
            delayUpdate-=1;
        } else {
            if(!isPaused) {
                currentTime = player? player.getCurrentTime() : 0;
            }
        }
    }
    chart.options.scales.x.max=Math.min(Math.floor(currentTime) + 120, predictions.length);
    chart.options.scales.x.min=Math.max(Math.floor(currentTime) - 120, 0);
    chart.options.plugins.annotation.annotations.line1.xMax = Math.floor(currentTime);
    chart.options.plugins.annotation.annotations.line1.xMin = Math.floor(currentTime);
    chart.update()
}
function seek(num) {
    if (player) player.seek(num)
    updateCurrentTime(num)
}  
function seekForward() {
    seek(currentTime + skipTimeArray[skipTimeIdx])
}
function seekBackward() {
    seek(currentTime - skipTimeArray[skipTimeIdx])
}
let recordStart = 0;
let isRecording = false;
let recordedTimes = [];
function toggleRecord() {
    if (isRecording) {
        if (currentTime > recordStart){
            recordedTimes = [...recordedTimes, `${formatAsTs(Math.floor(recordStart))}-${formatAsTs(Math.floor(currentTime))}`];
        } 
        recordStart=0;
        isRecording=false;
    } else {
        recordStart = currentTime;
        isRecording = true;
    }
}

function skipNext() {
    if (items && nextItemIdx < items.length)
    seek(items[nextItemIdx].start)
}

function keyboardShortcuts(event) {
    let char = (typeof event !== 'undefined') ? event.keyCode : event.which
    // console.log(char);
    switch(char) {
        case 32:
            startStop();
            event.preventDefault();
            break;
        case 39:
            seekForward();
            event.preventDefault();
            break;
        case 37:
            seekBackward();
            event.preventDefault();
            break;
        case 38:
            console.log(skipTimeIdx)
            skipTimeIdx = Math.min(skipTimeIdx+1, skipTimeArray.length-1);
            event.preventDefault();
            break;
        case 40:
            skipTimeIdx = Math.max(skipTimeIdx-1, 0);
            event.preventDefault();
            break;
        case 90:
            toggleRecord();
            event.preventDefault();
            break;
        case 67:
            copyRecordings();
            event.preventDefault();
            break;
        case 191:
            skipNext();
            event.preventDefault();
            break;
    }
}

function showRecordings() {
    window.alert(recordedTimes.map(e => '  - ' + e).join('\n'))
}

function clearRecording() {
    recordedTimes.pop()
    recordedTimes = recordedTimes
}

function clearAllRecordings() {
    recordedTimes = []
}

function copyRecordings() {
    navigator.clipboard.writeText(recordedTimes.map(e => '  - ' + e).join('\n'))
}

let interval = setInterval(updateCurrentTime, 100);
onDestroy(() => clearInterval(interval));
</script>

<svelte:window on:keydown={keyboardShortcuts} />

<div class="flex flex-row h-screen w-screen p-3">
    <div class="flex flex-col flex-1 w-5/6 h-full">
        <!-- <h1 class="text-4xl font-medium italic underline pb-3">Ben's badass stream labeller</h1> -->
        <div id="player" class="w-full h-full flex-1"></div>
        <div class="shrink">
            <canvas id="graph" class="hidden" height="6px" width="100%"></canvas>
            <div id="control-row" class="text-sm font-medium flex flex-row divide-x mt-2">
                <div class="flex flex-row gap-3 px-3 pt-5">
                    <button on:click={startStop}>
                        {#if isPaused}
                        <svg class="h-8 w-8 text-violet-500 hover:text-purple-600"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <polygon points="5 3 19 12 5 21 5 3" /></svg>
                        {:else}
                        <svg class="h-8 w-8 text-violet-500 hover:text-purple-600"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <rect x="6" y="4" width="4" height="16" />  <rect x="14" y="4" width="4" height="16" /></svg>
                        {/if}
                    </button>
                    <button on:click={seekBackward}><svg class="h-8 w-8 text-violet-500 hover:text-purple-600"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <polygon points="19 20 9 12 19 4 19 20" />  <line x1="5" y1="19" x2="5" y2="5" /></svg></button>
                    <button on:click={seekForward}><svg class="h-8 w-8 text-violet-500 hover:text-purple-600"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <polygon points="5 4 15 12 5 20 5 4" />  <line x1="19" y1="5" x2="19" y2="19" /></svg></button>
                    <button on:click={skipNext}><svg class="h-8 w-8 text-violet-500 hover:text-purple-600"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <polygon points="13 19 22 12 13 5 13 19" />  <polygon points="2 19 11 12 2 5 2 19" /></svg></button>
                </div>
                <div class="flex flex-row gap-4 px-3 pt-5">
                    <p class="font-medium flex flex-row items-center gap-2"><svg class="h-8 w-8 text-violet-500"  width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">  <path stroke="none" d="M0 0h24v24H0z"/>  <path d="M9 4.55a8 8 0 0 1 6 14.9m0 -4.45v5h5" />  <path d="M11 19.95a8 8 0 0 1 -5.3 -12.8" stroke-dasharray=".001 4.13" /></svg> {skipTimeArray[skipTimeIdx]}s</p>
                    <p class="font-medium flex flex-row items-center gap-2"><svg class="h-8 w-8 text-violet-500"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <circle cx="12" cy="12" r="10" />  <polyline points="12 6 12 12 16 14" /></svg> <span class="font-mono">{formatAsTs(Math.floor(currentTime))}</span></p>
                </div>
                <div class="flex flex-col px-3">
                    <p class="text-slate-300 font-light italic">Recording</p>
                    <div class="flex flex-row gap-2 px-3">
                        <button on:click={toggleRecord}>
                            {#if isRecording}
                            <svg class="h-8 w-8 text-red-500 hover:text-red-600"  viewBox="0 0 24 24"  fill="currentColor"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <circle cx="12" cy="12" r="8" /></svg>
                            {:else}
                            <svg class="h-8 w-8 text-red-500 hover:text-red-600"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <circle cx="12" cy="12" r="8" /></svg>
                            {/if}
                        </button>
                        <p class="flex flex-row items-center gap-2"><svg class="h-8 w-8 text-violet-500"  width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">  <path stroke="none" d="M0 0h24v24H0z"/>  <circle cx="7" cy="12" r="3" />  <circle cx="17" cy="12" r="3" />  <line x1="7" y1="15" x2="17" y2="15" /></svg>{recordedTimes.length}</p>
                        <button on:click={copyRecordings}><svg class="h-8 w-8 text-violet-500 hover:text-violet-600"  width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">  <path stroke="none" d="M0 0h24v24H0z"/>  <rect x="8" y="8" width="12" height="12" rx="2" />  <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2" /></svg></button>
                        <button on:click={clearRecording}><svg class="h-8 w-8 text-violet-500 hover:text-violet-600"  width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">  <path stroke="none" d="M0 0h24v24H0z"/>  <path d="M9 13l-4 -4l4 -4m-4 4h11a4 4 0 0 1 0 8h-1" /></svg></button>
                        <button on:click={clearAllRecordings}><svg class="h-8 w-8 text-violet-500 hover:text-violet-600"  width="24" height="24" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">  <path stroke="none" d="M0 0h24v24H0z"/>  <line x1="4" y1="7" x2="20" y2="7" />  <line x1="10" y1="11" x2="10" y2="17" />  <line x1="14" y1="11" x2="14" y2="17" />  <path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2 -2l1 -12" />  <path d="M9 7v-3a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3" /></svg></button>
                        {#if isRecording}
                        <p class="flex flex-row items-center gap-3 font-mono px-1"><svg class="h-8 w-8 text-violet-500"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round">  <polygon points="23 7 16 12 23 17 23 7" />  <rect x="1" y="5" width="15" height="14" rx="2" ry="2" /></svg>{formatAsTs(Math.floor(recordStart))} ({Math.floor(currentTime - recordStart)}s)</p>
                        {/if}
                    </div>
                </div>
                <!-- <p class="font-medium px-4 py-1">current time: <span class="font-mono">{formatAsTs(Math.floor(currentTime))}</span></p> -->
            </div>
        </div>
    </div>
    <div id="times" class="w-min">
        <ol bind:this={scrollContainer} id="time-list" class="px-2 divide-y divide-violet-100 flex-1 overflow-auto h-full">
        {#each items as item (item.id) }
            <div id={"time-" + item.id}>
                <TimeEntry {...item} currentTime={currentTime} on:click={() => seek(item.start)}  />
            </div>
        {/each}
        </ol>
    </div>
</div>