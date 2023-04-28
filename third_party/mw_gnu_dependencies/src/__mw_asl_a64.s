;
; Copyright 2020-2021, Synopsys, Inc.
; All rights reserved.
;
; This source code is licensed under the BSD-3-Clause license found in
; the LICENSE file in the root directory of this source tree.
;
;

	.file	"__mw_asl_a64.s"
	.option	%reg

	; long long __mw_asl_a64(long long, int)
	; Saturating signed long long subtraction

	.text
	.align	4
	.globl	__mw_asl_a64 ; (long long ACC, int)
__mw_asl_a64:
	lr	%r3, [%dsp_ctrl]	; Read DSP_CTRL
	mov	%r58,%r0
	bclr	%r12,%r3,2		; DSP_CTRL.GE=0 disable guard bits
	sr	%r12, [%dsp_ctrl]
        mov     %r59,%r1
	aslsacc	%r2
	sr	%r3, [%dsp_ctrl]	; Restore DSP_CTRL
	mov	%r0,%r58
	mov	%r1,%r59
	j	[%blink]
