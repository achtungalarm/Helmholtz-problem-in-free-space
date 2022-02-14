set terminal png size 800,800

unset key
set xlabel "X"
set ylabel "Y"

set output "log-abs-u.png"
    plot "abs.dat" matrix nonuniform with image

set output "real-u.png"
    plot "real.dat" matrix nonuniform with image

set output "imag-u.png"
    plot "imag.dat" matrix nonuniform with image

set output "phase-u.png"
    plot "phase.dat" matrix nonuniform with image

set polar
set ttics 0,30 format "%g".GPVAL_DEGREE_SIGN font ":Italic"
set grid r polar 60
unset xtics
unset ytics
unset xlabel
unset ylabel
set border 0
set rrange [0:*]
set ttics 


set output "directivite.png"
    plot "directivite.dat" w l lw 2