;
; Copyright 2020-2021, Synopsys, Inc.
; All rights reserved.
;
; This source code is licensed under the BSD-3-Clause license found in
; the LICENSE file in the root directory of this source tree.
;
; Routines for performing double-word  logical shift left for ARC
; ARC5+ (w/ barrel-shift)
; Inputs:  r0,r1 = operand
;          r2 = shift amount
; Outputs: r0,r1 = result
;
	.file	"lsllbs.s"
	; .include "miscmacro.i"
    .if $isa == "ARCompact"
	.option isa_ac, assume_short
	.define DF,
    .else
	.option isa_arc
	.off	emit_cfa
	.define DF, .f
    .endif
    .if $isa == "ARC" && $core_version <= 0x05
	.macro nop_a4
	    nop
	.endm
    .else
	.macro nop_a4
	.endm
    .endif
	.text

	.ifdef __ARC64
		.define ARC64_SUFFIX,32
	.else
		.define ARC64_SUFFIX,
	.endif

	.ifdef _ALIGN_32BIT_INSTRS
		.define delay_slots, 0
	.elif $off("delay_slots") || $on("bank_conflict_opt_hazard")
		.define delay_slots, 0
	.else
		.define delay_slots, 1
	.endif

	.ifdef _BE
	.define rlo,r1
	.define rhi,r0
	.else
	.define rlo,r0
	.define rhi,r1
	.endif


	.global _Lsllbs
	.cfa_bf	_Lsllbs
	.align 4
_Lsllbs:
    .if $isa == "ARCompact"
	bmsk.f	r2,r2,5
	jz	[blink]
	rsub.f	r3,r2,32
    .else
	and.f	r2, r2, 0x3f
	nop_a4
	jz\&DF	[blink]
	sub.f	r3,32,r2
	nop_a4
    .endif
	ble	.L1
	// 32 bit shift
	lsr	r12,rlo,r3
	asl	rhi,rhi,r2
	asl	rlo,rlo,r2
    .if delay_slots
	j.d\&DF	[blink]
	or	rhi,rhi,r12
    .else
	or	rhi,rhi,r12
	j\&DF	[blink]
    .endif
.L1:
	// 64 bit shift
    .if $isa == "ARCompact"
	neg	r3,r3
    .else
	sub	r3, 0, r3
    .endif
	asl	rhi,rlo,r3
    .if delay_slots
	j.d\&DF	[blink]
	sub	rlo,rlo,rlo
    .else
	sub	rlo,rlo,rlo
	j\&DF	[blink]
    .endif
	.cfa_ef
