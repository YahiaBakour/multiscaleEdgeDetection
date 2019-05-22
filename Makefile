.SUFFIXES:
.SUFFIXES: .o .cpp

#============================================================
TARGET1	=  edgeDetect
C_OBJS1     =  main.o utilities.o

TARGET2	=  edgeDetectFFT
C_OBJS2     =  mainFFT.o utilities.o fft.o

C_SOURCES = main.cpp utilities.cpp fft.cpp mainFFT.cpp
MY_INCLUDES = stb_image.h stb_image_write.h utilities.h fft.h

CCX = g++
CXXFLAGS = $(INC)

#============================================================
all: $(TARGET1) $(TARGET2)

.o:.cpp	$(MY_INCLUDES)
	$(CCX)  -c  $(CXXFLAGS) $<  

$(TARGET1) :   $(C_OBJS1)
	$(CCX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@

$(TARGET2) :   $(C_OBJS2)
	$(CCX) $(CXXFLAGS)  $^ $(LIBDIRS)  -o $@

# Implicit rules: $@ = target name, $< = first prerequisite name, $^ = name of all prerequisites 
#============================================================

ALL_SOURCES = Makefile $(C_SOURCES) $(MY_INCLUDES)

NOTES =

clean:
	rm -f $(TARGET1) $(TARGET2) $(C_OBJS1) $(C_OBJS2) *~



