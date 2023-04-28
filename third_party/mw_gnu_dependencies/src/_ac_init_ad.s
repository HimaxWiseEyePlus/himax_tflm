;
; Copyright 2020-2021, Synopsys, Inc.
; All rights reserved.
;
; This source code is licensed under the BSD-3-Clause license found in
; the LICENSE file in the root directory of this source tree.
; 
	; initialize the bit in status32 that allows unaligned accesses
_init_ad::
	.if $is_arcv2
	lr	r0, [status32]
	bset	r0, r0, 19 	;or	r0, r0, 0x80000
	flag	r0
	.endif
	j	[blink]
