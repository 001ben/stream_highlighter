import YAML from 'yaml';
import fs from 'fs';
import path from 'path';

function convertTimestamp(ts, i) {
    let parts = ts.split('-').map(e => e.split(':').map(parseFloat))
    let [start, end] = parts.map(e => e[0]*60*60 + e[1]*60 + e[2])
    let final_obj = {id: i, start: start, end: end}
    return final_obj
}

export async function load() {
  // let vid_folder = 'E:\\code\\stream_highlighter\\output\\stream_downloads\\original_video'
  // let vid_folder = 'E:\\code\\stream_highlighter\\output\\stream_downloads\\timmy_valo_twitch_rivals_exp2'
  let vid_folder = 'E:\\code\\stream_highlighter\\output\\stream_downloads\\playing_apex_with_elephante'
  let in_d = {
    'timestamps': JSON.parse(fs.readFileSync(path.join(vid_folder, 'timestamps.json'), 'utf-8')),
    'prediction_array': JSON.parse(fs.readFileSync(path.join(vid_folder, 'predictions.json'), 'utf-8')),
    'video_id': JSON.parse(fs.readFileSync(path.join(vid_folder, 'metadata.json'), 'utf-8')).video_id
  }
  return in_d
}