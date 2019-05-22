set term png
set output "runtimePlot.png"
set title "Runtime vs Implementation"
set xlabel "Image Size (MPix)"
set ylabel "Multiscale Edge Runtime (s)"
set key top left


plot "dataForPlot.txt" using 1:3:7 title "SERIAL" with errorlines, \
     "dataForPlot.txt" using 1:4:8 title "MPI" with errorlines, \
     "dataForPlot.txt" using 1:2:6  title "ACC" with errorlines
     

pause -1 "Hit any key to continue"
