CC = g++
CFLAGS = -std=c++11 -fopenmp
INCLUDES =
LDFLAGS = -lopencv_core -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs 
SOURCES = blurImageThreads.cpp
OUTF = blurImageThreads.exe
OBJS = blurImageThreads.o

$(OUTF): $(OBJS)
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o $(OUTF)
