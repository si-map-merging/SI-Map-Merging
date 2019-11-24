.KEEP_STATE:

all: fmc

DFLAGS      = -Wall -D_DEBUG
OPTFLAGS    =
OPTFLAGS    = -O3
INCFLAGS    = 
CFLAGS      = $(OPTFLAGS) $(DFLAGS) $(INCFLAGS)
LDFLAGS     = $(OPTFLAGS)

CC          = g++
H_FILES     = findClique.h graphIO.h

.cpp.o:
	$(CC) $(CFLAGS) -c $<

SEQ_SRC_FMC	= main_fmc.cpp 
OTH_SRC		= findClique.cpp findCliqueHeu.cpp utils.cpp graphIO.cpp

OTH_OBJ 		= $(OTH_SRC:%.cpp=%.o)

SEQ_OBJ_FMC	= $(SEQ_SRC_FMC:%.cpp=%.o) $(OTH_SRC)

$(SEQ_OBJ_FMC): $(H_FILES) Makefile

fmc: $(SEQ_OBJ_FMC) $(H_FILES)
	$(CC) $(LDFLAGS) -o fmc $(SEQ_OBJ_FMC)

clean:
	rm -rf *.o fmc 
