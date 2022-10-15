<script>
import { loop_guard } from "svelte/internal";


import { formatAsTs } from '$lib/util'

export let id;
export let start;
export let end;
export let currentTime = 0;
export let nextTime = false;
const timeDiff = end - start;
let className=false;
const baseClasses = 'p-3 border-solid rounded-none pointer-events-auto hover:bg-violet-100 cursor-pointer border-l-8';
$: className = baseClasses + ((currentTime>=start && currentTime<=end) ? ' border-violet-500' : ' border-transparent');
</script>

<li on:click class="whitespace-nowrap">
    <div class={className}>
        {formatAsTs(start)} - {formatAsTs(end)} ({Math.floor(timeDiff/60)}m {timeDiff%60}s)
        {#if nextTime}
        <span class="text-red-500 font-extrabold">!</span>
        {/if}
    </div>
</li>