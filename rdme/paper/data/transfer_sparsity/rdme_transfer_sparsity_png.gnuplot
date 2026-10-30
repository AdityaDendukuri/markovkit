set terminal pngcairo enhanced monochrome font 'Times,17' size 1800,1270
set output "/Users/adityadendukuri/DiscStochSim-multigrid/paper/figures/rdme_transfer_sparsity.png"
set datafile separator ','
set border 15 lw 0.5
set tics nomirror out scale 0.4
set format x '%g'
set format y '%g'
set xtics font ',8'
set ytics font ',8'
unset key
set multiplot layout 2,2 margins 0.12,0.98,0.12,0.91 spacing 0.16,0.18
set title "(a) R: 4 to 2 voxels (861 x 135,751)"
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:135750.5]
set yrange [860.5:-0.5]
set xtics 0,67875,135750
set ytics 0,430,860
plot "/Users/adityadendukuri/DiscStochSim-multigrid/paper/data/transfer_sparsity/schlogl_k40_level0.csv" every ::1 using 1:2 with dots lc rgb 'black' notitle
set title "(b) P: 4 to 2 voxels (135,751 x 861)"
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:860.5]
set yrange [135750.5:-0.5]
set xtics 0,430,860
set ytics 0,67875,135750
plot "/Users/adityadendukuri/DiscStochSim-multigrid/paper/data/transfer_sparsity/schlogl_k40_level0.csv" every ::1 using 2:1 with dots lc rgb 'black' notitle
set title "(c) R: 2 to 1 voxels (41 x 861)"
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:860.5]
set yrange [40.5:-0.5]
set xtics 0,430,860
set ytics 0,20,40
plot "/Users/adityadendukuri/DiscStochSim-multigrid/paper/data/transfer_sparsity/schlogl_k40_level1.csv" every ::1 using 1:2 with dots lc rgb 'black' notitle
set title "(d) P: 2 to 1 voxels (861 x 41)"
set xlabel 'Column index'
set ylabel 'Row index'
set xrange [-0.5:40.5]
set yrange [860.5:-0.5]
set xtics 0,20,40
set ytics 0,430,860
plot "/Users/adityadendukuri/DiscStochSim-multigrid/paper/data/transfer_sparsity/schlogl_k40_level1.csv" every ::1 using 2:1 with dots lc rgb 'black' notitle
unset multiplot

unset output
