import { error } from '@sveltejs/kit';
import { json } from '@sveltejs/kit';
import YAML from 'yaml';
import fs from 'fs';
 
export function GET({ url }) {
//   const d = max - min;
 
//   if (isNaN(d) || d < 0) {
//     throw error(400, 'min and max must be numbers, and min must be less than max');
//   }
 
//   const random = min + Math.random() * d;
 
  return json({a:1});
}