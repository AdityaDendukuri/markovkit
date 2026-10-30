set terminal pdfcairo enhanced monochrome font 'Times,10' size 6.8in,4.8in
set output "/Users/adityadendukuri/DiscStochSim-multigrid/paper/figures/rdme_transfer_population_block.pdf"
set datafile separator ','
set border 15 lw 0.5
set tics nomirror out scale 0.4
set format x '%g'
set format y '%g'
set xtics font ',8'
set ytics font ',8'
unset key
set multiplot layout 2,2 margins 0.10,0.98,0.12,0.91 spacing 0.14,0.18
unset object
set title "(a) R_4: solver ordering"
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:34.5]
set yrange [4.5:-0.5]
set xtics 0,17,34
set ytics 0,2,4
plot "/Users/adityadendukuri/DiscStochSim-multigrid/paper/data/transfer_sparsity/schlogl_k40_population4.csv" every ::1 using 1:2 with points pt 7 ps 0.4 lc rgb 'black' notitle
unset object
set title "(b) P_4: solver ordering"
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:4.5]
set yrange [34.5:-0.5]
set xtics 0,2,4
set ytics 0,17,34
plot "/Users/adityadendukuri/DiscStochSim-multigrid/paper/data/transfer_sparsity/schlogl_k40_population4.csv" every ::1 using 2:1 with points pt 7 ps 0.4 lc rgb 'black' notitle
unset object
set object 1 rect from -0.5,-0.5 to 4.5,0.5 front fs empty border lc rgb 'black' lw 0.5
set object 2 rect from 4.5,0.5 to 12.5,1.5 front fs empty border lc rgb 'black' lw 0.5
set object 3 rect from 12.5,1.5 to 21.5,2.5 front fs empty border lc rgb 'black' lw 0.5
set object 4 rect from 21.5,2.5 to 29.5,3.5 front fs empty border lc rgb 'black' lw 0.5
set object 5 rect from 29.5,3.5 to 34.5,4.5 front fs empty border lc rgb 'black' lw 0.5
set title "(c) R_4: fine states grouped by coarse counts"
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:34.5]
set yrange [4.5:-0.5]
set xtics 0,17,34
set ytics 0,2,4
plot "/Users/adityadendukuri/DiscStochSim-multigrid/paper/data/transfer_sparsity/schlogl_k40_population4_grouped.csv" every ::1 using 1:2 with points pt 7 ps 0.4 lc rgb 'black' notitle
unset object
set object 1 rect from -0.5,-0.5 to 0.5,4.5 front fs empty border lc rgb 'black' lw 0.5
set object 2 rect from 0.5,4.5 to 1.5,12.5 front fs empty border lc rgb 'black' lw 0.5
set object 3 rect from 1.5,12.5 to 2.5,21.5 front fs empty border lc rgb 'black' lw 0.5
set object 4 rect from 2.5,21.5 to 3.5,29.5 front fs empty border lc rgb 'black' lw 0.5
set object 5 rect from 3.5,29.5 to 4.5,34.5 front fs empty border lc rgb 'black' lw 0.5
set title "(d) P_4: fine states grouped by coarse counts"
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:4.5]
set yrange [34.5:-0.5]
set xtics 0,2,4
set ytics 0,17,34
plot "/Users/adityadendukuri/DiscStochSim-multigrid/paper/data/transfer_sparsity/schlogl_k40_population4_grouped.csv" every ::1 using 2:1 with points pt 7 ps 0.4 lc rgb 'black' notitle
unset multiplot

unset output
