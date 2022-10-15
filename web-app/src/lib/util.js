
export function formatAsTs(num) {
    return `${String(Math.floor(num / 60 / 60)).padStart(2, '0')}:${String(Math.floor((num/60)%60)).padStart(2, '0')}:${String(num%60).padStart(2, '0')}`
}