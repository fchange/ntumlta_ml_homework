# 潘石屹的代码，2020年1月16日。
# 在运行前猜测一下运行结果吧。
import turtle

p = turtle.Pen()
p.speed(0)
p.pencolor("red")
p.fillcolor("yellow")
p.begin_fill()
for i in range(36):
    for j in range(4):
        p.forward(100)
        p.left(90)
    p.right(10)
p.end_fill()