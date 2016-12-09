// gamelife.cl
//
// OpenCL kernel for Conway's cellular automata game of life on parallel compute

#define NEIGHBORS(a,b,c,d,e,f,g,h) ((a)?1:0)+((b)?1:0)+((c)?1:0)+((d)?1:0)+((e)?1:0)+((f)?1:0)+((g)?1:0)+((h)?1:0)
#define LIVE(sl,sh,rl,rh,a,n) (a) ? (n)>=(sl)&&(n)<=(sh) : (n)>=(rl)&&(n)<=(rh)

__kernel void k(global int* grid, int odd_Neven, int sl, int sh, int rl, int rh)
{
  int row_global, rows_per_strip, row_strips;
  int col, col_left, col_strip_offset, col_my_strip;
  int lsb_old, lsb_new, row_leftbits, row_old, row_new;
  int above, level, below, alive_status, next_value;
  int mask, mask_left, mask_right;

  // figure out the grid index of the pertinent 30+2 bit word
  rows_per_strip = get_local_size(0);
  row_strips = get_num_groups(0);
  col_my_strip = get_group_id(1);
  col_strip_offset = col_my_strip * (row_strips * (2+rows_per_strip));
  row_global = get_global_id(0);
  row_leftbits = (col_strip_offset + row_global) << 1;

  // convert to an interleaved offset from the beginning of grid and read words
  lsb_new = (lsb_old = odd_Neven & 1) ? 0 : 1;
  row_old = row_leftbits | lsb_old;
  above = grid[row_old - 2]; // skip interleaved (_new) row
  level = grid[row_old]
  below = grid[row_old + 2]; // skip interleaved (_new) row
  row_new = row_leftbits | lsb_new;
  
  // establish whether this element lives, dies, or gets resurrected
  col = get_local_id(1);
  mask = 1 << (col + 1); // col can be 0..30, but bits indexed 1..31
  mask_left = mask << 1;
  mask_right = mask >> 1;
  alive_status = LIVE(sl, sh, rl, rh, mask & level,
		      NEIGHBORS(mask_left & above,
				mask & above,
				mask_right & above,
				mask_left & level,
				mask_right & level,
				mask_left & below,
				mask & below,
				mask_right & below));

  // write into the new array (requires a barrier for atomicity?!?)
  grid[row_new] = (grid[row_new] & ~mask) | (alive_status ? mask : 0);
}
