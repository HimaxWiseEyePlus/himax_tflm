;
; Copyright 2020-2021, Synopsys, Inc.
; All rights reserved.
;
; This source code is licensed under the BSD-3-Clause license found in
; the LICENSE file in the root directory of this source tree.
;
;
; Routines for performing 64-bit integer divide for ARC
; This version is optimized for the barrel shifter and NORM extension,
; and is invoked by the compiler for integer division if both of those
; extensions are turned on.
;
; Inputs:  ahi,alo = dividend
;          bhi,blo = divisor
; Outputs: qhi,qlo = quotient
;          rhi,rlo = remainder
;
	.file	"ldiv.s"
;	.include "miscmacro.i"
	.option	%reg
	.text
	.global _ldivmod_normbs
	.global	_uldivmod_normbs
	
	.if $isa == "ARC" && $core_version <= 0x05
	    .macro nop_a4
		nop
	    .endm
	.else
	    .macro nop_a4
	    .endm
	.endif
	
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
	
	.if $lpc_width == 0 || $arc_family == "ARC64" || $defined(__ZD_SIZE_THRESHOLD)
	    .define removed_lp, 1
	.else
	    .define removed_lp, 0
	.endif
	
	.if $isa == "ARC"
	    .option isa_arc
	    .off emit_cfa
	    .define	 _S,  ""
	    .err	"This function is not yet ported for A4"
	    .end
	.else
	    .define DF,
	    .align	4
	    .cfa_bf	_ldivmod_normbs
	    .option isa_ac, assume_short
	    .define	 _S,  _s
	.endif
	
	.ifdef	_BE
    	    .define     qlo,%r1
	    .define     qhi,%r0
	    .define     rlo,%r3
	    .define     rhi,%r2
	    .define     alo,%r1
	    .define     ahi,%r0
	    .define     blo,%r3
	    .define     bhi,%r2
	.else
    	    .define     qlo,%r0
	    .define     qhi,%r1
	    .define     rlo,%r2
	    .define     rhi,%r3
	    .define     alo,%r0
	    .define     ahi,%r1
	    .define     blo,%r2
	    .define     bhi,%r3
	.endif
	.ifdef _ARC_RF16
	    .define     signs,%r10
	    .define     temp,%r11
	    .define     tmp1,%r12
	    .define     tmp2,%r13
	    .define     xhi,%r14
	    .define     xlo,%r15
	.else
	    .define     signs,%r4
	    .define     temp,%r5
	    .define     tmp1,%r7
	    .define     tmp2,%r10
	    .define     xhi,%r11
	    .define     xlo,%r12
	.endif

.ifdef __ARC64
	arc64_divrem_thunk _uldivmod_normbs
	arc64_divrem_thunk _ldivmod_normbs
.endif

	.align	4
_uldivmod_normbs\&ARC64_SUFFIX:
    .if delay_slots
	b.d	_divide_normbs
	mov	signs, 0	; signs reg used to store sign of results
    .else
	mov	signs, 0	; signs reg used to store sign of results
	b	_divide_normbs
    .endif

	.align	4
_ldivmod_normbs\&ARC64_SUFFIX:
	xor	signs,ahi,bhi
	asl.f	0,ahi
	mov     temp,0
    .if delay_slots
	bcc.d  .Labsvsr
	rrc	signs,signs
    .else
	rrc	signs,signs
	bcc	.Labsvsr
    .endif
	sub.f   alo,temp,alo	; abs value of divisor
	sbc     ahi,temp,ahi	
.Labsvsr:		
	cmp	bhi,0
	nop_a4
	bpl	.Labsvnd
	sub.f   blo,temp,blo	; abs value of dividend
	sbc     bhi,temp,bhi
.Labsvnd:		
    .if removed_lp
	; we use lower byte of signs as an extra temp
	bic	signs, signs, 0xff
    .endif
		                 ;	signs[1]=sign of r
_divide_normbs:
	.ifdef	_ARC_RF16
	push	tmp2		; Save registers
	push	xhi
	push	xlo
	.cfa_push	12
	.endif
	or.f	temp,bhi,blo
	beq	.Ldiv_zero	; if divide by zero
	mov	xhi,bhi		; copy divisor
	mov	xlo,blo
	mov	rlo,alo		; copy dividend to remainder
	mov	rhi,ahi		; 
	; Note that since unsigned div shares this code, norm must
	; keep track of whether leading bit is set.
	mov.f	temp,xhi 	; handle norm of divisor
	mov	tmp2,0
	bmi	.Ldvover
	add.eq	tmp2,tmp2,32	; short divisor
	mov.eq.f  temp,xlo
	norm	tmp1,temp
	add.pl  tmp2,tmp2,1
	add.pl	tmp2,tmp2,tmp1
.Ldvover:
	mov	tmp1,0		; handle dividend now
	mov.f	temp,rhi
	add.eq	tmp1,tmp1,32    ; short dividend
	bmi	.Lddover
	mov.eq.f  temp,rlo
	norm	alo,temp
	add.pl  tmp1,tmp1,1
	add.pl	tmp1,tmp1,alo
.Lddover:
	sub.f	tmp1,tmp2,tmp1
	mov.mi	tmp1,0
	add	tmp1,tmp1,1     ;  shift
	mov	temp,tmp1       ; loop trip
	cmp	tmp1,32		; test shift >= 32
	mov	qhi,0		; zero quotient
    .if delay_slots
	bmi.d	.Lsftok		; if shift <32
	mov	qlo,0
    .else
	mov	qlo,0
	bmi	.Lsftok		; if shift <32
    .endif
	mov	qhi,rlo
	mov	rlo,rhi
	mov	rhi,0		
	sub.f	tmp1,tmp1,32
.Lsftok:
    .if removed_lp
	; temp has loop count value.  since we can't save in lp_count,
	; we need another temp reg but instruction block below uses up
	; every available reg.  so we use low-order byte of "signs", since
	; signs only cares about two high-order bits.
	or	signs, signs, temp
    .else
	mov	%lp_count,temp  ; number of bits in quotient
    .endif
	beq	.Lsftzr		; if no shift of dividend
	rsub.f	tmp2,tmp1,32    
        beq	.Lwords	   	; if special case of shift of 32
	lsl	qlo,qhi,tmp2    ; shift divident bit into hold area
	lsr	qhi,qhi,tmp1    ; this if a four word right shift
	lsl	temp,rlo,tmp2
	add	qhi,qhi,temp
	lsr	rlo,rlo,tmp1
	lsl	temp,rhi,tmp2
	lsr	rhi,rhi,tmp1
	add	rlo,rlo,temp	
.Lsftzr:		
    .if removed_lp
	extb	temp, signs
	bic	signs, signs, 0xff
	; temp now contains the loop count when ZD loops are removed
    .endif
	cmp	tmp2,32
	nop_a4
	bgt	.Lshort
    .if removed_lp
	.Ltop1:
    .else
	lp	divloop_end
    .endif
	
; loop for large divisor trip count must be small
; divloop:
	add.f	qlo,qlo,qlo	; position next quotient bit
	adc.f	qhi,qhi,qhi
	adc.f	rlo,rlo,rlo 	; shift in next remainder bit
	adc	rhi,rhi,rhi
	sub.f	rlo,rlo,xlo	; remainder-divisor
	sbc.f   rhi,rhi,xhi
	nop_a4
	bcc     .Lnorestore     ; and skip restore
	add.f	rlo,rlo,xlo     ; restore remainder
	adc.f   rhi,rhi,xhi
.Lnorestore:	
	add.cc  qlo,qlo,1       ; if no carry add one to quotient
    .if removed_lp
	sub.f	temp, temp, 1
	nop_a4
	bne	.Ltop1
    .endif
	
; Place signs on result and exit
divloop_end:
	sub.f	0,signs,0
	mov	temp,0
	beq	.Lout
        bge	.Lnocng
	sub.f	rlo,temp,rlo
	sbc     rhi,temp,rhi
.Lnocng:		
	asl.f	signs,signs
	nop_a4
	bpl	.Lout
	sub.f	qlo,temp,qlo
	sbc	qhi,temp,qhi
.Lout:	
	.ifdef	_ARC_RF16
	pop	xlo		; restore registers
	pop	xhi
	pop	tmp2
	.cfa_pop	12
	.endif
	j\&DF	[%blink]
	
;  Handle case where shift is exactly 32 
.Lwords:	
	mov	qlo,qhi
	mov	qhi,rlo
	mov	rlo,rhi
	mov	rhi,0	
	b	.Lsftzr
	
; Handle case where divisor is single word, trip count likely larger
.Lshort:		
    .if removed_lp
	.Ltop2:
    .else
	lp	divloop_nd2
    .endif
; divloop:
	add.f	qlo,qlo,qlo	; position next quotient bit
	adc.f	qhi,qhi,qhi
	adc.f	rlo,rlo,rlo	; shift next bit into remainder
	.if	$isa == "ARCompact"
	cmp	rlo,xlo
	.else
	sub.f	0,rlo,xlo	; remainder-divisor
	.endif
	sub.hs	rlo,rlo,xlo	; subtract divisor from remainder
	or.hs	qlo,qlo,1	; enter bit into quotient
    .if removed_lp
	sub.f	temp, temp, 1
	nop_a4
	bne	.Ltop2
    .endif
divloop_nd2:
	b	divloop_end	
	
//////////////////////////////////////

;  Handle divide by zero
.Ldiv_zero:
	mov	rlo,0		; zero remainder
	sub	qlo,rlo,1	; 0x7fffff... quotient
	mov	rhi,0
	lsr	qhi,qlo
	b	.Lout

	.type  _ldivmod_normbs, @function
	.type  _uldivmod_normbs, @function

	.if $isa == "ARCompact"
	.cfa_ef
	.endif
	.end

