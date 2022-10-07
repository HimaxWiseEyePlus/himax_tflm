/*
* Copyright 2020-2021, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#include <arc/arc_intrinsics.h>
extern unsigned long long
_uldivmod_normbs(unsigned long long a, unsigned long long b);

// ----------------------------------------------------------------
// Knuth's efficient division algorithm
// Works iff higher order 32 bits of the dividend < divisor
// ----------------------------------------------------------------
static
unsigned long long divlu2(unsigned u1, unsigned u0, unsigned v)
{
  const unsigned b = 65536; // Number base (16 bits).
  unsigned un1, un0,        // Norm. dividend LSD's.
    vn1, vn0,        // Norm. divisor digits.
    q1, q0,          // Quotient digits.
    un32, un21, un10,// Dividend digit pairs.
    rhat;            // A remainder.

  int s; 
  //  s = nlz(v);               // 0 <= s <= 31.
  s = v & (1 << 31) ? 0 :_norm(v) + 1;
  v = v << s;               // Normalize divisor.
  vn1 = v >> 16;            // Break divisor up into
  vn0 = v & 0xFFFF;         // two 16-bit digits.

  un32 = (u1 << s) | ((u0 >> (32 - s)) & (-s >> 31));
  un10 = u0 << s;           // Shift dividend left.

  un1 = un10 >> 16;         // Break right half of
  un0 = un10 & 0xFFFF;      // dividend into two digits.

  q1 = un32/vn1;            // Compute the first
  rhat = un32 - q1*vn1;     // quotient digit, q1.
  do{
	  if (q1 >= b || q1 * vn0 > b*rhat + un1) {
		  q1 = q1 - 1;
		  rhat = rhat + vn1;
	  }
	  else break;
  }while(rhat < b);


  un21 = un32*b + un1 - q1*v;  // Multiply and subtract.

  q0 = un21/vn1;            // Compute the second
  rhat = un21 - q0*vn1;     // quotient digit, q0.
  do{
	  if (q0 >= b || q0 * vn0 > b*rhat + un0) {
		  q0 = q0 - 1;
		  rhat = rhat + vn1;
	  }
	  else break;
  }while(rhat < b);
  return (unsigned long long)q1*b + q0;
}

// x = dividend, y = divisor
unsigned long long 
_uldivmod_normbs_opt(unsigned long long x, unsigned long long y)
{
  // a = higher 32 bits of x (dividend)
  unsigned int a = x >> 32; 
  // b = lower 32 bits of y (divisor)
  unsigned int b = y&0xFFFFFFFFLL;
  //  = higher 32 bits of x (divisor)
  unsigned int c = y >> 32;
  if ((a<b) && (c==0)) {
    unsigned int a2 = x&0xFFFFFFFFLL;
    return divlu2(a,a2,b);
  }
  else
    return _uldivmod_normbs(x,y);
}

// x = dividend, y = divisor
signed long long 
_ldivmod_normbs_opt(signed long long x, signed long long y)
{
  int sx = x < 0 ? 1 : 0;
  int sy = y < 0 ? 1 : 0;
  if(sx) x = -x;
  if(sy) y = -y;
  // a = higher 32 bits of x (dividend)
  unsigned int a = x >> 32; 
  // b = lower 32 bits of y (divisor)
  unsigned int b = y & 0xFFFFFFFFLL;
  //  = higher 32 bits of x (divisor)
  unsigned int c = y >> 32;
  if ((a<b) && (c == 0)) {
    unsigned int a2 = x&0xFFFFFFFFLL;
    unsigned long long t = divlu2(a,a2,b);
    return (sx ^ sy) ? -t : t;
  }
  else
    {
      unsigned long long answer = _uldivmod_normbs(x,y);
      return (sx ^ sy) && (y!=0) ? -answer: answer;
    }
}
