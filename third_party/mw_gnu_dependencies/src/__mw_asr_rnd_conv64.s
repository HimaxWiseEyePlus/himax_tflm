;
; Copyright 2020-2021, Synopsys, Inc.
; All rights reserved.
;
; This source code is licensed under the BSD-3-Clause license found in
; the LICENSE file in the root directory of this source tree.
;
;

	.file	"__mw_asr_rnd_conv64.s"
	.option	%reg

	; long long __mw_asr_rnd_conv64(long long, int)
	; Saturating/rounded signed long long arithmetic shift

	; CAVEAT: we know the shift amount in %r2 is negative and that this
	; function is performing a right shift because it's called only when
	; the shift amount is negative from the function fx_asl_rnd_q63().

	.text
	.align	4
	.globl	__mw_asr_rnd_conv64 ; (long long, int)
__mw_asr_rnd_conv64:
	.ifdef	_LE
	.define	LHS_L,	%r0
	.define	LHS_H,	%r1
	.else
	.define	LHS_L,	%r1
	.define	LHS_H,	%r0
	.endif

	cmp	%r2,-63			; if shift amount > 63 then return zero
	bge	1f
	mov	%r0,0
	mov	%r1,0
	j	[%blink]

1:
	add	%r10,%r2,64		; Compute convergent rounding bits
	setacc	0,LHS_H,0x301		; Init ACC0_HI and ACC0_GHI
	mov	%accl,LHS_L		; Init ACC0_LO
	aslacc	%r10
	mov	%r10,%accl		; (R11,R10) bits lost by shift
	mov	%r11,%acch

	setacc	0,LHS_H,0x301		; Init ACC0_HI and ACC0_GHI
	mov	%accl,LHS_L		; Init ACC0_LO
	extb	%r2,%r2			; Select wide accumulator and mask shift amount
	aslsacc	%r2			; Compute shift right amount
	mov	LHS_L,%accl
	mov	LHS_H,%acch

	cmp	%r11,0			; If last_deleted_mask is zero return
	jge	[%blink]

	btst	LHS_L,0			; If LSB is non-zero and last_deleted_mask is
	bne	2f			; non-zero then round up

	bxor	%r11,%r11,31		; Clear last_deleted_mask bit
	or	%r11,%r11,%r10
	cmp	%r11,0			; If last_deleted_mask is zero return
	jeq	[%blink]

2:
	add.f	LHS_L,LHS_L,1		; Rounded LSB
	adc	LHS_H,LHS_H,0		; Rounded MSB
	j	[%blink]
