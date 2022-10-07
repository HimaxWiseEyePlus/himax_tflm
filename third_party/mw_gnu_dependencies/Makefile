#
# Copyright 2020-2021, Synopsys, Inc.
# All rights reserved.
# 
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#
#

# OS-specific definitions
BACKSLASH=\$(nullstring)
ifneq ($(ComSpec)$(COMSPEC),)
	O_SYS=Windows
	RM=del /F /Q
	MKDIR=mkdir 
	CP=copy /Y
	TYPE=type
	PS=$(BACKSLASH)
	Q=
	coQ=\$(nullstring)
	fix_platform_path = $(subst /,$(PS), $(1))
	DEV_NULL = nul
else
	O_SYS=Unix
	RM=rm -rf
	MKDIR=mkdir -p
	CP=cp 
	TYPE=cat
	PS=/
	Q=$(BACKSLASH)
	coQ=
	fix_platform_path=$(1)
	DEV_NULL=/dev/null
endif


# Specify tools
CC = ccac
AR_TOOL = arac
CXX = ccac
LD := ccac

# em11d parameters
CCAC=ccac
OPT_LEV=
CC_OPT= -O3 -Hsdata0 -Hnoxcheck -av2em -core4 -Mm -Xbs -Xbitstream -Xcd -Xdiv_rem -Xmpy -Xmpy16 -Xxy -Xnorm -Xsa -Xswap -Xdsp -Xunaligned -Hharvard -Hreg_reserve=30
CC_INC= -I$(MWLIB)/mw/inc/internal -I$(MWLIB)/mw/inc
cflags=

AS=$(CCAC)
AS_OPT=-c -Hnoxcheck -av2em 
AS_INC=
aflags=

# Specify the target file and the install directory
OUTPUTFILE  = libmwdepend.a
INSTALLDIR  = gnu_depend_lib
OBJDIR = obj
SRCDIR = src

# Build libmwdepend.a from objects
$(INSTALLDIR)/$(OUTPUTFILE): $(OBJDIR)/l64x32divnorm.o \
							 $(OBJDIR)/_ac_init_ad.o $(OBJDIR)/ldivnorm.o $(OBJDIR)/lsllbs.o \
							 $(OBJDIR)/__mw_asl_a64.o $(OBJDIR)/__mw_asr_rnd_conv64.o | $(INSTALLDIR)
	arac -r $@ $^
	arac -s $@

$(OBJDIR):
	$(MKDIR) $(OBJDIR)
	
$(INSTALLDIR):
	$(MKDIR) $(INSTALLDIR)

# Default target
.PHONY: all
all: $(INSTALLDIR)/$(OUTPUTFILE)

.PHONY: clean 
clean:
	@$(RM) $(INSTALLDIR)
	@$(RM) $(OBJDIR)
	@$(RM) *.o
	@$(RM) *.a

# Indicate dependencies
$(OBJDIR)/l64x32divnorm.o: $(SRCDIR)/l64x32divnorm.c | $(OBJDIR)
	$(CXX) $(CC_OPT) $(cflags) -o $@ -I. -c $(OPT_LEV) -Xnorm -Xbarrel_shifter -Xmpy src/l64x32divnorm.c

$(OBJDIR)/_ac_init_ad.o: $(SRCDIR)/_ac_init_ad.s  | $(OBJDIR)
	$(AS) $(AS_OPT) $(aflags) $(AS_INC) -o $@ src/_ac_init_ad.s

$(OBJDIR)/ldivnorm.o: $(SRCDIR)/ldivnorm.s  | $(OBJDIR)
	$(AS) $(AS_OPT) $(aflags) -Xbarrel_shifter -Xnorm $(AS_INC) -o $@ src/ldivnorm.s

$(OBJDIR)/lsllbs.o: $(SRCDIR)/lsllbs.s  | $(OBJDIR)
	$(AS) $(AS_OPT) $(aflags) -Xbarrel_shifter $(AS_INC) -o $@ src/lsllbs.s

$(OBJDIR)/__mw_asl_a64.o: $(SRCDIR)/__mw_asl_a64.s  | $(OBJDIR)
	$(AS) $(AS_OPT) -Xdsp $(aflags) -Hon=incompat_option_warn $(AS_INC) -o $@ src/__mw_asl_a64.s
	
$(OBJDIR)/__mw_asr_rnd_conv64.o: $(SRCDIR)/__mw_asr_rnd_conv64.s  | $(OBJDIR)
	$(AS) $(AS_OPT) -Xdsp $(aflags) -Hon=incompat_option_warn $(AS_INC) -o $@ src/__mw_asr_rnd_conv64.s